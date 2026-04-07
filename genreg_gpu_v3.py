# ================================================================
# GENREG 2048 — GPU Evolver V3 (Evolved Encoder)
# ================================================================
# Extends GPUEvolver with an evolved perception layer:
#   Raw signals (22) → Encoder (22→enc_dim, evolved activation) → Controller (enc_dim→32→4)
#
# Each genome evolves:
#   - Encoder weights + bias
#   - Activation function selection (index into catalog of 8)
#   - Activation parameters (4 floats that tune the activation shape)
#   - Controller weights + bias (as before)
#   - Protein parameters (as before)
# ================================================================

import torch
import math
import random as py_random

from genreg_gpu import (
    GPUEvolver, BatchGame2048, _controller_forward_compiled,
    _get_tables, _HAS_COMPILE, MAX_POWER, BASE,
)
from genreg_encoder_gpu import (
    apply_evolved_activations, DEFAULT_PARAMS, PARAM_BOUNDS, NUM_ACTIVATIONS,
)


class GPUEvolverV3(GPUEvolver):
    """
    GPU Evolver with evolved encoder perception layer.

    Forward pass per step:
      1. Raw signals (22 dims) from game
      2. Encoder: linear(22 → enc_dim) + evolved activation
      3. Controller: linear(enc_dim → hidden) + tanh → linear(hidden → 4) → argmax

    The encoder's activation function is selected per-genome from a catalog
    of 8 diverse nonlinearities. Each genome literally sees the board
    through a different mathematical lens.
    """

    def __init__(self, population, config, device="cuda", encoder_dim=32):
        self.raw_input_size = 22  # fixed: 16 board cells + 6 meta

        # Detect encoder_dim from existing genomes if they have encoders
        g0_enc = getattr(population.genomes[0], 'encoder', None)
        if g0_enc is not None:
            encoder_dim = g0_enc.encoder_dim
        self.encoder_dim = encoder_dim

        # Don't call super().__init__ yet — we need to set up the encoder first
        # then pack the population manually
        self.dev = torch.device(device)
        self.B = population.size
        self.cfg = config

        g0 = population.genomes[0].controller
        self.input_size = g0.input_size  # original input size (22)
        self.hidden_size = g0.hidden_size
        self.output_size = g0.output_size

        self._pack_from_population_v3(population)
        self._build_sig_index_tensors()

    def _pack_from_population_v3(self, population):
        """Pack population to GPU including encoder weights."""
        genomes = population.genomes
        d = self.dev
        B = len(genomes)

        # --- Encoder weights ---
        # Check if genomes have encoders (new v3) or not (legacy)
        has_encoder = hasattr(genomes[0], 'encoder') and genomes[0].encoder is not None

        if has_encoder:
            self.enc_w = torch.tensor(
                [g.encoder.enc_w for g in genomes], dtype=torch.float32, device=d)
            self.enc_b = torch.tensor(
                [g.encoder.enc_b for g in genomes], dtype=torch.float32, device=d)
            self.act_ids = torch.tensor(
                [g.encoder.act_id for g in genomes], dtype=torch.long, device=d)

            # Pack activation params into 4 uniform tensors (p1-p4)
            # Map from named params to positional
            self.act_p1 = torch.zeros(B, self.encoder_dim, device=d)
            self.act_p2 = torch.zeros(B, self.encoder_dim, device=d)
            self.act_p3 = torch.zeros(B, self.encoder_dim, device=d)
            self.act_p4 = torch.zeros(B, self.encoder_dim, device=d)

            for i, g in enumerate(genomes):
                # Prefer per-neuron params if available (preserves evolved state)
                per_neuron = getattr(g.encoder, "act_params_per_neuron", None)
                if per_neuron is not None and len(per_neuron) == self.encoder_dim:
                    for n in range(self.encoder_dim):
                        pn = per_neuron[n]
                        keys = list(pn.keys())
                        if len(keys) > 0: self.act_p1[i, n] = pn[keys[0]]
                        if len(keys) > 1: self.act_p2[i, n] = pn[keys[1]]
                        if len(keys) > 2: self.act_p3[i, n] = pn[keys[2]]
                        if len(keys) > 3: self.act_p4[i, n] = pn[keys[3]]
                else:
                    # Fallback: broadcast scalar params across all neurons
                    p = g.encoder.act_params
                    keys = list(p.keys())
                    if len(keys) > 0: self.act_p1[i] = p[keys[0]]
                    if len(keys) > 1: self.act_p2[i] = p[keys[1]]
                    if len(keys) > 2: self.act_p3[i] = p[keys[2]]
                    if len(keys) > 3: self.act_p4[i] = p[keys[3]]
        else:
            # Initialize fresh encoders for legacy populations
            self.enc_w = torch.randn(B, self.encoder_dim, self.raw_input_size, device=d) * 0.5
            self.enc_b = torch.randn(B, self.encoder_dim, device=d) * 0.1
            self.act_ids = torch.randint(0, NUM_ACTIVATIONS, (B,), device=d)

            # Initialize activation params from defaults based on act_id
            self.act_p1 = torch.zeros(B, self.encoder_dim, device=d)
            self.act_p2 = torch.zeros(B, self.encoder_dim, device=d)
            self.act_p3 = torch.zeros(B, self.encoder_dim, device=d)
            self.act_p4 = torch.zeros(B, self.encoder_dim, device=d)
            for i in range(B):
                aid = self.act_ids[i].item()
                dp = DEFAULT_PARAMS[aid]
                self.act_p1[i] = dp[0]
                self.act_p2[i] = dp[1]
                self.act_p3[i] = dp[2]
                self.act_p4[i] = dp[3]

        # --- Controller weights ---
        # Controller now takes encoder_dim inputs, not raw 22
        # For legacy populations with 22-input controllers, we need to re-init
        ctrl_input = genomes[0].controller.input_size
        if ctrl_input != self.encoder_dim:
            # Re-initialize controller weights for the new input size
            self.w1 = torch.randn(B, self.hidden_size, self.encoder_dim, device=d) * 0.5
            self.b1 = torch.randn(B, self.hidden_size, device=d) * 0.1
        else:
            self.w1 = torch.tensor(
                [g.controller.w1 for g in genomes], dtype=torch.float32, device=d)
            self.b1 = torch.tensor(
                [g.controller.b1 for g in genomes], dtype=torch.float32, device=d)

        self.w2 = torch.tensor(
            [g.controller.w2 for g in genomes], dtype=torch.float32, device=d)
        self.b2 = torch.tensor(
            [g.controller.b2 for g in genomes], dtype=torch.float32, device=d)

        # --- Protein params (same as base) ---
        g0 = genomes[0]
        self._trend_indices = [i for i, p in enumerate(g0.proteins) if p.type == "trend"]
        self._trust_indices = [i for i, p in enumerate(g0.proteins) if p.type == "trust_modifier"]
        n_trends = len(self._trend_indices)
        n_trusts = len(self._trust_indices)

        if n_trends > 0:
            self.trend_momentum = torch.tensor(
                [[g.proteins[i].params.get("momentum", 0.5)
                  for i in self._trend_indices] for g in genomes],
                dtype=torch.float32, device=d)
        else:
            self.trend_momentum = torch.zeros(B, 1, dtype=torch.float32, device=d)

        if n_trusts > 0:
            self.trust_gain = torch.tensor(
                [[g.proteins[i].params.get("gain", 1.0)
                  for i in self._trust_indices] for g in genomes],
                dtype=torch.float32, device=d)
            self.trust_scale = torch.tensor(
                [[g.proteins[i].params.get("scale", 1.0)
                  for i in self._trust_indices] for g in genomes],
                dtype=torch.float32, device=d)
            self.trust_decay = torch.tensor(
                [[g.proteins[i].params.get("decay", 0.999)
                  for i in self._trust_indices] for g in genomes],
                dtype=torch.float32, device=d)
        else:
            self.trust_gain = torch.ones(B, 1, dtype=torch.float32, device=d)
            self.trust_scale = torch.ones(B, 1, dtype=torch.float32, device=d)
            self.trust_decay = torch.full((B, 1), 0.999, dtype=torch.float32, device=d)

        # Wiring (same as base)
        self._trend_signal_keys = []
        self._trend_names = []
        for i in self._trend_indices:
            p = g0.proteins[i]
            self._trend_signal_keys.append(p.inputs[0] if p.inputs else None)
            self._trend_names.append(p.name)

        trend_name_to_col = {name: col for col, name in enumerate(self._trend_names)}
        self._trust_input_map = []
        for i in self._trust_indices:
            p = g0.proteins[i]
            key = p.inputs[0] if p.inputs else None
            if key in trend_name_to_col:
                self._trust_input_map.append(("trend", trend_name_to_col[key]))
            else:
                self._trust_input_map.append(("signal", key))

    @torch.no_grad()
    def run_generation(self, n_games=1):
        """Run generation with encoder perception layer."""
        B = self.B
        cfg = self.cfg
        n_trends = len(self._trend_indices)
        n_trusts = len(self._trust_indices)

        if n_games > 1:
            BG = B * n_games
            idx = torch.arange(B, device=self.dev).repeat_interleave(n_games)
            enc_w = self.enc_w[idx]
            enc_b = self.enc_b[idx]
            act_ids = self.act_ids[idx]
            ap1, ap2, ap3, ap4 = self.act_p1[idx], self.act_p2[idx], self.act_p3[idx], self.act_p4[idx]
            w1 = self.w1[idx]; b1 = self.b1[idx]
            w2 = self.w2[idx]; b2 = self.b2[idx]
            tm = self.trend_momentum[idx]
            tg = self.trust_gain[idx]; ts = self.trust_scale[idx]; td = self.trust_decay[idx]
        else:
            BG = B
            enc_w, enc_b = self.enc_w, self.enc_b
            act_ids = self.act_ids
            ap1, ap2, ap3, ap4 = self.act_p1, self.act_p2, self.act_p3, self.act_p4
            w1, b1, w2, b2 = self.w1, self.b1, self.w2, self.b2
            tm = self.trend_momentum
            tg, ts, td = self.trust_gain, self.trust_scale, self.trust_decay

        env = BatchGame2048(BG, cfg["target_tile"], cfg["starting_energy"], device=str(self.dev),
                            invalid_move_penalty=cfg.get("invalid_move_penalty", 2.0))

        trend_vel = torch.zeros(BG, max(n_trends, 1), dtype=torch.float32, device=self.dev)
        trend_last = torch.zeros(BG, max(n_trends, 1), dtype=torch.float32, device=self.dev)
        trust_running = torch.zeros(BG, max(n_trusts, 1), dtype=torch.float32, device=self.dev)
        trust_total = torch.zeros(BG, dtype=torch.float32, device=self.dev)
        trust_in = torch.zeros(BG, max(n_trusts, 1), dtype=torch.float32, device=self.dev) if n_trusts > 0 else None
        trend_init = False

        # Pre-transpose controller weights
        w1t = w1.transpose(1, 2).contiguous()
        w2t = w2.transpose(1, 2).contiguous()

        # Pre-transpose encoder weights
        enc_wt = enc_w.transpose(1, 2).contiguous()  # (BG, input_dim, enc_dim)

        sig = env.reset()
        # Move cap: allow long games. Energy is the primary limiter anyway
        # (games die from energy exhaustion far before hitting this cap).
        # Old cap of starting_energy + 500 = 530 moves, which is too short
        # for 1024 runs that average 1000+ moves at inference.
        cap = cfg.get("max_moves_per_game", 3000)

        for step_i in range(cap):
            if step_i % 16 == 0 and step_i > 0 and not env.alive.any():
                break
            alive_f = env.alive.float()

            # --- ENCODER: raw signals → evolved activation → encoded features ---
            # Linear: (BG, 1, 22) @ (BG, 22, enc_dim) = (BG, 1, enc_dim)
            encoded = torch.bmm(sig.unsqueeze(1), enc_wt).squeeze(1) + enc_b
            # Evolved activation (per-genome function selection)
            encoded = apply_evolved_activations(encoded, act_ids, ap1, ap2, ap3, ap4)

            # --- CONTROLLER: encoded → hidden → action ---
            h = torch.bmm(encoded.unsqueeze(1), w1t).squeeze(1) + b1
            actions = (torch.bmm(torch.tanh(h).unsqueeze(1), w2t).squeeze(1) + b2).argmax(1)

            sig = env.step(actions)

            # --- Proteins (use raw signals, not encoded) ---
            if n_trends > 0:
                trend_in = sig[:, self._trend_sig_cols]
                if not trend_init:
                    trend_last.copy_(trend_in)
                    trend_init = True
                else:
                    delta = trend_in - trend_last
                    trend_last.copy_(trend_in)
                    trend_vel.mul_(tm).add_((1.0 - tm) * delta)

            if n_trusts > 0:
                trust_in.zero_()
                if self._trust_trend_js_t is not None:
                    trust_in[:, self._trust_trend_js_t] = trend_vel[:, self._trust_trend_cols_t]
                if self._trust_sig_js_t is not None:
                    trust_in[:, self._trust_sig_js_t] = sig[:, self._trust_sig_cols_t]
                trust_running.mul_(td).add_((1.0 - td) * trust_in)
                trust_delta = (tg * ts * trust_running).sum(1)
                trust_total.add_(trust_delta * alive_f)

        results = env.get_results()

        if n_games > 1:
            all_tiles = [r[0] for r in results]
            all_scores = [r[1] for r in results]
            all_moves = [r[2] for r in results]
            all_invalid = [r[3] for r in results]
            trust_2d = trust_total.reshape(B, n_games)
            trust_avg = trust_2d.mean(dim=1)
            tiles, scores, efficiencies = [], [], []
            for i in range(B):
                chunk_t = all_tiles[i * n_games:(i + 1) * n_games]
                chunk_s = all_scores[i * n_games:(i + 1) * n_games]
                chunk_m = all_moves[i * n_games:(i + 1) * n_games]
                chunk_i = all_invalid[i * n_games:(i + 1) * n_games]
                tiles.append(max(chunk_t))
                scores.append(sum(chunk_s) // n_games)
                total_m = sum(chunk_m)
                total_i = sum(chunk_i)
                efficiencies.append(1.0 - (total_i / max(total_m, 1)))
            return tiles, scores, trust_avg, efficiencies
        else:
            tiles = [r[0] for r in results]
            scores = [r[1] for r in results]
            moves = [r[2] for r in results]
            invalids = [r[3] for r in results]
            efficiencies = [1.0 - (iv / max(mv, 1)) for mv, iv in zip(moves, invalids)]
            return tiles, scores, trust_total, efficiencies

    @torch.no_grad()
    def evolve(self, trust, survival_pct=20, trust_inherit=0.1,
               child_mutation_rate=0.05, tiles=None, scores=None,
               efficiencies=None):
        """Evolve with encoder weights + activation params.

        Adaptive mutation: rate scales from 0.05 (early) down to 0.01
        as the population's best tile approaches 2048.  The caller's
        child_mutation_rate is treated as the *max* rate; the floor is
        always 0.01.
        """
        B = self.B
        ratchet_strength = self.cfg.get("ratchet_strength", 2.0)
        proximity_strength = self.cfg.get("proximity_strength", 1.0)

        # --- ADAPTIVE MUTATION RATE ---
        mut_max = child_mutation_rate  # 0.05 default
        mut_min = 0.01
        if tiles is not None and len(tiles) > 0:
            best_tile = max(tiles)
            # log2 progress: tile 4 → 2, tile 2048 → 11
            progress = (math.log2(max(best_tile, 4)) - 2.0) / (11.0 - 2.0)
            progress = max(0.0, min(1.0, progress))  # clamp [0, 1]
            child_mutation_rate = mut_max - (mut_max - mut_min) * progress
        # else: keep the passed-in rate as-is

        # --- NORMALIZE TRUST ---
        t_mean = trust.mean()
        t_std = trust.std().clamp(min=1e-6)
        trust = (trust - t_mean) / t_std

        # --- TILE RATCHET ---
        if tiles is not None and ratchet_strength > 0:
            tiles_t = torch.tensor(tiles, dtype=torch.float32, device=self.dev)
            log_tiles = torch.log2(tiles_t.clamp(min=1))
            pop_mean = log_tiles.mean()
            pop_std = log_tiles.std().clamp(min=0.5)
            z_scores = ((log_tiles - pop_mean) / pop_std).clamp(min=0)
            ratchet_bonus = z_scores.pow(2) * ratchet_strength
            trust = trust + ratchet_bonus

        # --- SCORE PROXIMITY BONUS ---
        # Use _NEXT_TILE_SCORE (not _TILE_SCORE_THRESHOLDS) so the bonus
        # pushes toward the NEXT tile's threshold, not the current one.
        # Without this, a 512 genome with score 6900 gets the same bonus
        # as one with score 3500 — no gradient toward 1024.
        from genreg_gpu import GPUEvolver
        if tiles is not None and scores is not None and proximity_strength > 0:
            tiles_t = torch.tensor(tiles, dtype=torch.float32, device=self.dev)
            scores_t = torch.tensor(scores, dtype=torch.float32, device=self.dev)
            proximity_bonus = torch.zeros(B, dtype=torch.float32, device=self.dev)
            for tile_val, next_score in GPUEvolver._NEXT_TILE_SCORE.items():
                at_tile = (tiles_t == tile_val)
                if not at_tile.any():
                    continue
                ratio = (scores_t / max(next_score, 1.0)).clamp(0, 1.5)
                bonus = ratio.pow(2) * proximity_strength
                proximity_bonus += bonus * at_tile.float()
            trust = trust + proximity_bonus

        # --- EFFICIENCY BONUS ---
        # Reward genomes that waste fewer moves on invalid actions.
        # A genome with 60% valid moves has more energy headroom to reach
        # higher tiles than one at 40% valid. This creates selection pressure
        # toward energy-efficient play, which is required for 2048.
        if efficiencies is not None:
            eff_t = torch.tensor(efficiencies, dtype=torch.float32, device=self.dev)
            # Efficiency ranges from ~0.5 (bad) to ~0.9 (good)
            # Bonus scales quadratically: 0.5 → 0, 0.7 → 0.04, 0.9 → 0.16
            eff_bonus = (eff_t - 0.5).clamp(min=0).pow(2) * 5.0
            trust = trust + eff_bonus

        sorted_trust, indices = trust.sort(descending=True)
        st = sorted_trust.cpu()
        best_trust = st[0].item()
        median_trust = st[B // 2].item()
        lowest_trust = st[-1].item()

        # --- SELECTIVE REPLACEMENT ---
        k = max(1, int(B * survival_pct / 100.0))
        cull_start = B - k

        elite_idx = indices[:k]
        middle_idx = indices[k:cull_start]

        elite_trust = trust[elite_idx]
        weights = elite_trust - elite_trust.min() + 1.0
        parent_choices = torch.multinomial(weights, k, replacement=True)
        replacement_parent_idx = elite_idx[parent_choices]

        keep_idx = torch.cat([elite_idx, middle_idx])
        new_idx = torch.cat([keep_idx, replacement_parent_idx])

        # Reorder all tensors (encoder + controller + proteins)
        all_attrs = [
            "enc_w", "enc_b", "act_p1", "act_p2", "act_p3", "act_p4",
            "w1", "b1", "w2", "b2",
            "trend_momentum", "trust_gain", "trust_scale", "trust_decay",
        ]
        for attr in all_attrs:
            setattr(self, attr, getattr(self, attr)[new_idx].contiguous())
        self.act_ids = self.act_ids[new_idx].contiguous()

        # --- MUTATE only the bottom k (replacements) ---
        n_keep = len(keep_idx)
        rate = child_mutation_rate

        # Encoder + controller weights
        for param in [self.enc_w, self.enc_b, self.w1, self.b1, self.w2, self.b2]:
            bottom = param[n_keep:]
            mask = torch.rand_like(bottom) < rate
            bottom.add_(torch.randn_like(bottom) * 0.1 * mask)

        # Activation params
        bounds = [
            (self.act_p1, 0, 1), (self.act_p2, 2, 3),
            (self.act_p3, 4, 5), (self.act_p4, 6, 7),
        ]
        for param, lo_idx, hi_idx in bounds:
            bottom = param[n_keep:]
            mask = torch.rand_like(bottom) < rate
            bottom.add_(torch.randn_like(bottom) * 0.2 * mask)
            # Clamp per activation type (approximate: use wide bounds)
            bottom.clamp_(-5.0, 10.0)

        # Protein params
        for param, scale, lo, hi in [
            (self.trend_momentum, 0.2, 0.0, 0.99),
            (self.trust_gain, 0.2, 0.1, 10.0),
            (self.trust_scale, 0.2, -5.0, 5.0),
            (self.trust_decay, 0.2, 0.0, 0.999),
        ]:
            bottom = param[n_keep:]
            mask = torch.rand_like(bottom) < rate
            noise = torch.randn_like(bottom) * scale * (bottom.abs() + 1e-9)
            bottom.add_(noise * mask).clamp_(lo, hi)

        # Small chance to switch activation for replacements.
        # When we switch, we MUST reset the activation params to the new
        # activation's defaults — otherwise the params are meaningless
        # (they were tuned for a different activation function).
        switch_mask = torch.rand(k, device=self.dev) < rate * 0.1
        if switch_mask.any():
            new_acts = torch.randint(0, NUM_ACTIVATIONS, (k,), device=self.dev)
            old_acts = self.act_ids[n_keep:].clone()
            self.act_ids[n_keep:] = torch.where(switch_mask, new_acts, old_acts)
            # For any genome that actually switched, reset its params to defaults
            for local_i in range(k):
                if switch_mask[local_i].item():
                    global_i = n_keep + local_i
                    new_aid = self.act_ids[global_i].item()
                    dp = DEFAULT_PARAMS[new_aid]
                    self.act_p1[global_i] = dp[0]
                    self.act_p2[global_i] = dp[1]
                    self.act_p3[global_i] = dp[2]
                    self.act_p4[global_i] = dp[3]

        self.effective_mutation_rate = child_mutation_rate
        return best_trust, median_trust, lowest_trust

    def sync_to_cpu(self, population, tiles=None, scores=None, trust=None):
        """Sync everything including encoder back to CPU genomes."""
        # First sync controller + proteins via parent
        super().sync_to_cpu(population, tiles=tiles, scores=scores, trust=trust)

        # Now sync encoder
        enc_w = self.enc_w.cpu().tolist()
        enc_b = self.enc_b.cpu().tolist()
        act_ids = self.act_ids.cpu().tolist()
        ap1 = self.act_p1.cpu().tolist()
        ap2 = self.act_p2.cpu().tolist()
        ap3 = self.act_p3.cpu().tolist()
        ap4 = self.act_p4.cpu().tolist()

        from genreg_encoder import GENREGEncoder, ACTIVATION_CATALOG

        for i, g in enumerate(population.genomes):
            if not hasattr(g, 'encoder') or g.encoder is None:
                g.encoder = GENREGEncoder(self.raw_input_size, self.encoder_dim)

            g.encoder.enc_w = enc_w[i]
            g.encoder.enc_b = enc_b[i]
            g.encoder.act_id = act_ids[i]

            # Map positional params back to named params
            _, defaults, bounds = ACTIVATION_CATALOG[act_ids[i]]
            keys = list(defaults.keys())
            # Scalar act_params uses neuron 0 (legacy)
            vals_0 = [ap1[i][0], ap2[i][0], ap3[i][0], ap4[i][0]]
            g.encoder.act_params = {}
            g.encoder.act_bounds = dict(bounds)
            for j, key in enumerate(keys):
                g.encoder.act_params[key] = vals_0[j] if j < len(vals_0) else 0.0
            # Per-neuron act_params (preserves evolved diversity)
            enc_dim = len(ap1[i])
            g.encoder.act_params_per_neuron = []
            for n in range(enc_dim):
                vals_n = [ap1[i][n], ap2[i][n], ap3[i][n], ap4[i][n]]
                params_n = {}
                for j, key in enumerate(keys):
                    params_n[key] = vals_n[j] if j < len(vals_n) else 0.0
                g.encoder.act_params_per_neuron.append(params_n)
