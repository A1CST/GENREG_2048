# ================================================================
# GENREG 2048 — GPU-Accelerated Training Module
# ================================================================
# ENTIRE training pipeline on GPU: game sim, controller, proteins,
# evolution, mutation. CPU only touches checkpoints and prints.
#
# - Lookup-table vectorized 2048 (fused all-4-directions)
# - Fast valid-move check (adjacent comparison, not 4-dir recompute)
# - Batched controller forward (bmm)
# - Vectorized protein cascade (tensor ops)
# - GPU evolution: selection, cloning, mutation as tensor ops
# - Weights stay on GPU across generations (no pack/unpack)
# ================================================================

import torch
import math
import random as py_random

# Enable TF32 tensor cores for float32 matmul (huge speedup on Ampere+)
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

MAX_POWER = 15
BASE = 16
_TABLE_CACHE = {}
_HAS_COMPILE = hasattr(torch, "compile")


def _get_tables(device):
    key = str(device)
    if key not in _TABLE_CACHE:
        _TABLE_CACHE[key] = _build_table().to(device)
    return _TABLE_CACHE[key]


def _build_table():
    size = BASE ** 4
    tbl = torch.zeros(size, 6, dtype=torch.long)
    for enc in range(size):
        c = [(enc >> s) & 0xF for s in (12, 8, 4, 0)]
        nz = [x for x in c if x > 0]
        res, sc, mg, i = [], 0, 0, 0
        while i < len(nz):
            if i + 1 < len(nz) and nz[i] == nz[i + 1]:
                v = min(nz[i] + 1, MAX_POWER)
                res.append(v); sc += (1 << v); mg += 1; i += 2
            else:
                res.append(nz[i]); i += 1
        res += [0] * (4 - len(res))
        tbl[enc, :4] = torch.tensor(res[:4])
        tbl[enc, 4] = sc
        tbl[enc, 5] = mg
    return tbl


def _enc(boards):
    return (boards[:, :, 0] << 12) | (boards[:, :, 1] << 8) | (boards[:, :, 2] << 4) | boards[:, :, 3]


def _all_dirs(boards, tbl):
    B = boards.shape[0]
    t = boards.transpose(1, 2).contiguous()
    mega = torch.cat([t, t.flip(2), boards, boards.flip(2)], dim=0)
    result = tbl[_enc(mega).reshape(-1)]
    nr = result[:, :4].reshape(4 * B, 4, 4)
    sc = result[:, 4].reshape(4 * B, 4).sum(1)
    mg = result[:, 5].reshape(4 * B, 4).sum(1)
    return (
        torch.stack([
            nr[:B].transpose(1, 2).contiguous(),
            nr[B:2*B].flip(2).transpose(1, 2).contiguous(),
            nr[2*B:3*B],
            nr[3*B:].flip(2),
        ]),
        sc.reshape(4, B),
        mg.reshape(4, B),
    )


# ================================================================
# BATCHED GAME
# ================================================================
class BatchGame2048:
    def __init__(self, B, target_tile=2048, max_energy=50, device="cuda",
                 invalid_move_penalty=2.0):
        self.B = B
        self.max_energy = max_energy
        self.invalid_move_penalty = invalid_move_penalty
        self.dev = torch.device(device)
        self.tbl = _get_tables(self.dev)

    def reset(self):
        self.boards = torch.zeros(self.B, 4, 4, dtype=torch.long, device=self.dev)
        self.scores = torch.zeros(self.B, dtype=torch.long, device=self.dev)
        self.max_tiles = torch.zeros(self.B, dtype=torch.long, device=self.dev)
        self.moves_made = torch.zeros(self.B, dtype=torch.long, device=self.dev)
        self.invalid_moves = torch.zeros(self.B, dtype=torch.long, device=self.dev)
        self.last_merges = torch.zeros(self.B, dtype=torch.long, device=self.dev)
        self.energy = torch.full((self.B,), self.max_energy, dtype=torch.float32, device=self.dev)
        self.alive = torch.ones(self.B, dtype=torch.bool, device=self.dev)
        self._bi = torch.arange(self.B, device=self.dev)  # cached index
        self._spawn_fill = torch.full((self.B, 16), 2.0, device=self.dev)
        self._spawn_one = torch.ones(self.B, dtype=torch.long, device=self.dev)
        self._spawn_two = torch.full((self.B,), 2, dtype=torch.long, device=self.dev)
        ones = torch.ones(self.B, dtype=torch.bool, device=self.dev)
        self._spawn(ones); self._spawn(ones)
        self.max_tiles = self.boards.reshape(self.B, -1).max(1).values
        return self._signals()

    def _spawn(self, mask):
        # Branchless: skip .any() GPU→CPU syncs; ops are no-ops on zero masks
        flat_boards = self.boards.reshape(self.B, 16)
        rand = torch.rand(self.B, 16, device=self.dev)
        # Set occupied cells and masked-out boards to 2.0 so they can't be selected
        rand = torch.where((flat_boards == 0) & mask.unsqueeze(1), rand, self._spawn_fill)
        mv, mi = rand.min(1)
        valid = mv < 2.0
        # Value: 1 (90%) or 2 (10%)
        vals = torch.where(torch.rand(self.B, device=self.dev) < 0.9,
                           self._spawn_one, self._spawn_two)
        # Only update valid boards — read current value for invalid ones
        current_at_mi = flat_boards.gather(1, mi.unsqueeze(1)).squeeze(1)
        flat_boards.scatter_(1, mi.unsqueeze(1),
                             torch.where(valid, vals, current_at_mi).unsqueeze(1))

    def step(self, actions):
        all_b, all_s, all_m = _all_dirs(self.boards, self.tbl)
        bi = self._bi
        nb = all_b[actions, bi]
        sg = all_s[actions, bi]
        mc = all_m[actions, bi]
        moved = (nb != self.boards).any(dim=(1, 2)) & self.alive
        moved_l = moved.long()
        self.boards = torch.where(moved[:, None, None].expand_as(self.boards), nb, self.boards)
        self.scores += sg * moved_l
        self.last_merges = mc * moved_l
        self._spawn(moved)
        flat = self.boards.reshape(self.B, 16)
        self.max_tiles = flat.max(1).values
        alive_f = self.alive.float()
        moved_f = moved.float()
        # Match CPU energy mechanics exactly:
        #   if moved:  energy += 3*merges - 1
        #   if !moved: energy -= 2
        #   always:    energy -= 0.01  (passive decay)
        merge_reward = 3.0 * mc.float() * moved_f       # +3 per merge (only if moved)
        valid_cost = -1.0 * moved_f                       # -1 for valid move
        invalid_cost = -self.invalid_move_penalty * (1.0 - moved_f) * alive_f
        passive_decay = -0.01 * alive_f                   # -0.01 every step
        self.energy += merge_reward + valid_cost + invalid_cost + passive_decay
        self.moves_made += self.alive.long()
        self.invalid_moves += ((1.0 - moved_f) * alive_f).long()
        dead = (self.energy <= 0) & self.alive
        has_empty = (flat == 0).any(1)
        b = self.boards
        h_merge = ((b[:, :, :-1] == b[:, :, 1:]) & (b[:, :, :-1] != 0)).reshape(self.B, -1).any(1)
        v_merge = ((b[:, :-1, :] == b[:, 1:, :]) & (b[:, :-1, :] != 0)).reshape(self.B, -1).any(1)
        no_moves = ~(has_empty | h_merge | v_merge) & self.alive & ~dead
        self.alive &= ~dead & ~no_moves
        return self._signals()

    def _signals(self):
        # Pre-allocated buffer avoids cat/stack overhead each step
        if not hasattr(self, '_sig_buf'):
            self._sig_buf = torch.zeros(self.B, 22, dtype=torch.float32, device=self.dev)
        buf = self._sig_buf
        flat = self.boards.reshape(self.B, 16)
        buf[:, :16] = flat.float() * (1.0 / 11.0)
        buf[:, 16] = self.max_tiles.float() * (1.0 / 11.0)
        buf[:, 17] = (flat == 0).sum(1).float()
        buf[:, 18] = self.scores.float()
        buf[:, 19] = self.moves_made.float()
        buf[:, 20] = self.last_merges.float()
        buf[:, 21] = self.alive.float()
        return buf

    def get_results(self):
        mt = self.max_tiles.cpu().tolist()
        sc = self.scores.cpu().tolist()
        mv = self.moves_made.cpu().tolist()
        iv = self.invalid_moves.cpu().tolist()
        return [((1 << mt[i]) if mt[i] > 0 else 0, sc[i], mv[i], iv[i]) for i in range(self.B)]


# ================================================================
# GPU EVOLVER — Entire population lifecycle on GPU
# ================================================================
# Weights, protein params, evolution, mutation all stay on GPU.
# CPU is only used for checkpoint sync and printing.
# ================================================================
def _controller_forward(sig, w1t, b1, w2t, b2):
    """Batched controller: sig (B, I) -> actions (B,)"""
    h = torch.bmm(sig.unsqueeze(1), w1t).squeeze(1) + b1
    return (torch.bmm(torch.tanh(h).unsqueeze(1), w2t).squeeze(1) + b2).argmax(1)


# Compile the controller forward if possible (fuses bmm + tanh + argmax)
if _HAS_COMPILE:
    try:
        _controller_forward_compiled = torch.compile(
            _controller_forward, mode="reduce-overhead", fullgraph=True)
    except Exception:
        _controller_forward_compiled = _controller_forward
else:
    _controller_forward_compiled = _controller_forward


class GPUEvolver:
    def __init__(self, population, config, device="cuda"):
        self.dev = torch.device(device)
        self.B = population.size
        self.cfg = config

        g0 = population.genomes[0].controller
        self.input_size = g0.input_size
        self.hidden_size = g0.hidden_size
        self.output_size = g0.output_size

        self._pack_from_population(population)
        self._build_sig_index_tensors()

    def _pack_from_population(self, population):
        """One-time pack from CPU population to GPU tensors.
        Discovers protein layout dynamically — works with any protein template.
        """
        genomes = population.genomes
        d = self.dev

        # Controller weights
        self.w1 = torch.tensor([g.controller.w1 for g in genomes], dtype=torch.float32, device=d)
        self.b1 = torch.tensor([g.controller.b1 for g in genomes], dtype=torch.float32, device=d)
        self.w2 = torch.tensor([g.controller.w2 for g in genomes], dtype=torch.float32, device=d)
        self.b2 = torch.tensor([g.controller.b2 for g in genomes], dtype=torch.float32, device=d)

        # Discover protein layout from first genome
        g0 = genomes[0]
        self._trend_indices = [i for i, p in enumerate(g0.proteins) if p.type == "trend"]
        self._trust_indices = [i for i, p in enumerate(g0.proteins) if p.type == "trust_modifier"]
        n_trends = len(self._trend_indices)
        n_trusts = len(self._trust_indices)

        # Protein params (evolved per-genome) — dynamic layout
        if n_trends > 0:
            self.trend_momentum = torch.tensor(
                [[g.proteins[i].params.get("momentum", 0.5) for i in self._trend_indices] for g in genomes],
                dtype=torch.float32, device=d)
        else:
            self.trend_momentum = torch.zeros(len(genomes), 1, dtype=torch.float32, device=d)

        if n_trusts > 0:
            self.trust_gain = torch.tensor(
                [[g.proteins[i].params.get("gain", 1.0) for i in self._trust_indices] for g in genomes],
                dtype=torch.float32, device=d)
            self.trust_scale = torch.tensor(
                [[g.proteins[i].params.get("scale", 1.0) for i in self._trust_indices] for g in genomes],
                dtype=torch.float32, device=d)
            self.trust_decay = torch.tensor(
                [[g.proteins[i].params.get("decay", 0.999) for i in self._trust_indices] for g in genomes],
                dtype=torch.float32, device=d)
        else:
            self.trust_gain = torch.ones(len(genomes), 1, dtype=torch.float32, device=d)
            self.trust_scale = torch.ones(len(genomes), 1, dtype=torch.float32, device=d)
            self.trust_decay = torch.full((len(genomes), 1), 0.999, dtype=torch.float32, device=d)

        # Discover wiring: what signal feeds each trend, what feeds each trust modifier
        # Trends read raw signals by input name
        self._trend_signal_keys = []
        self._trend_names = []
        for i in self._trend_indices:
            p = g0.proteins[i]
            self._trend_signal_keys.append(p.inputs[0] if p.inputs else None)
            self._trend_names.append(p.name)

        # Trust modifiers: input is either a trend output (by name) or a raw signal
        trend_name_to_col = {name: col for col, name in enumerate(self._trend_names)}
        self._trust_input_map = []  # list of ("trend", col) or ("signal", key)
        for i in self._trust_indices:
            p = g0.proteins[i]
            key = p.inputs[0] if p.inputs else None
            if key in trend_name_to_col:
                self._trust_input_map.append(("trend", trend_name_to_col[key]))
            else:
                self._trust_input_map.append(("signal", key))

    # --- Signal column mapping (sig tensor layout from _signals): ---
    # cols 0-15: board cells, 16: max_tile, 17: empty_count, 18: score,
    # 19: moves_made, 20: last_merges, 21: alive
    _SIGNAL_COL = {
        "max_tile": 16, "empty_count": 17, "score": 18,
        "moves_made": 19, "steps_alive": 19, "last_merge_count": 20,
    }

    def _build_sig_index_tensors(self):
        """Pre-compute integer column indices for trend/trust inputs.
        Called once at init so the hot loop does pure tensor indexing.
        """
        # Trend signal columns
        self._trend_sig_cols = torch.tensor(
            [self._SIGNAL_COL.get(k, 0) for k in self._trend_signal_keys],
            dtype=torch.long, device=self.dev) if self._trend_signal_keys else None

        # Trust modifier inputs: separate lists for trend-sourced vs sig-sourced
        n_trusts = len(self._trust_indices)
        # Pre-build the column arrays for each source type
        self._trust_trend_js = []  # output column indices that read from trend_vel
        self._trust_trend_cols = []  # which trend_vel column to read
        self._trust_sig_js = []  # output column indices that read from sig
        self._trust_sig_cols_list = []  # which sig column to read
        for j, (src_type, src_val) in enumerate(self._trust_input_map):
            if src_type == "trend":
                self._trust_trend_js.append(j)
                self._trust_trend_cols.append(src_val)
            else:
                self._trust_sig_js.append(j)
                self._trust_sig_cols_list.append(self._SIGNAL_COL.get(src_val, 0))
        # Convert to tensors for fast indexing
        self._trust_trend_js_t = torch.tensor(self._trust_trend_js, dtype=torch.long, device=self.dev) if self._trust_trend_js else None
        self._trust_trend_cols_t = torch.tensor(self._trust_trend_cols, dtype=torch.long, device=self.dev) if self._trust_trend_cols else None
        self._trust_sig_js_t = torch.tensor(self._trust_sig_js, dtype=torch.long, device=self.dev) if self._trust_sig_js else None
        self._trust_sig_cols_t = torch.tensor(self._trust_sig_cols_list, dtype=torch.long, device=self.dev) if self._trust_sig_cols_list else None
        self._n_trusts = n_trusts

    @torch.no_grad()
    def run_generation(self, n_games=1):
        """Run one generation entirely on GPU.
        n_games: number of games per genome (>1 = multi-game ensemble).
                 Each genome plays n_games independent boards.  Trust, tiles,
                 and scores are averaged across games per genome.
        Returns: tiles (list), scores (list), trust (GPU tensor)
        """
        B = self.B
        cfg = self.cfg
        n_trends = len(self._trend_indices)
        n_trusts = len(self._trust_indices)

        # --- Multi-game expansion ---
        # Duplicate weights so genome i occupies slots [i*N, i*N+1, ..., i*N+N-1]
        if n_games > 1:
            BG = B * n_games  # total parallel games
            # Repeat each genome's weights N times: [0,0,0,1,1,1,2,2,2,...]
            idx = torch.arange(B, device=self.dev).repeat_interleave(n_games)
            w1_exp = self.w1[idx]
            b1_exp = self.b1[idx]
            w2_exp = self.w2[idx]
            b2_exp = self.b2[idx]
            tm_exp = self.trend_momentum[idx]
            tg_exp = self.trust_gain[idx]
            ts_exp = self.trust_scale[idx]
            td_exp = self.trust_decay[idx]
        else:
            BG = B
            w1_exp, b1_exp, w2_exp, b2_exp = self.w1, self.b1, self.w2, self.b2
            tm_exp, tg_exp, ts_exp, td_exp = (
                self.trend_momentum, self.trust_gain, self.trust_scale, self.trust_decay)

        env = BatchGame2048(BG, cfg["target_tile"], cfg["starting_energy"], device=str(self.dev),
                            invalid_move_penalty=cfg.get("invalid_move_penalty", 2.0))

        # Protein state (fresh each generation)
        trend_vel = torch.zeros(BG, max(n_trends, 1), dtype=torch.float32, device=self.dev)
        trend_last = torch.zeros(BG, max(n_trends, 1), dtype=torch.float32, device=self.dev)
        trust_running = torch.zeros(BG, max(n_trusts, 1), dtype=torch.float32, device=self.dev)
        trust_total = torch.zeros(BG, dtype=torch.float32, device=self.dev)
        trust_in = torch.zeros(BG, max(n_trusts, 1), dtype=torch.float32, device=self.dev) if n_trusts > 0 else None
        trend_init = False

        # Pre-transpose weights
        w1t = w1_exp.transpose(1, 2).contiguous()
        w2t = w2_exp.transpose(1, 2).contiguous()

        sig = env.reset()
        # Allow long games (1024+ runs average 1000+ moves at inference).
        # Energy is the real limiter; most games die well before this cap.
        cap = cfg.get("max_moves_per_game", 3000)

        for step_i in range(cap):
            if step_i % 16 == 0 and step_i > 0 and not env.alive.any():
                break
            alive_f = env.alive.float()

            actions = _controller_forward_compiled(sig, w1t, b1_exp, w2t, b2_exp)
            sig = env.step(actions)

            if n_trends > 0:
                trend_in = sig[:, self._trend_sig_cols]
                if not trend_init:
                    trend_last.copy_(trend_in)
                    trend_init = True
                else:
                    delta = trend_in - trend_last
                    trend_last.copy_(trend_in)
                    trend_vel.mul_(tm_exp).add_((1.0 - tm_exp) * delta)

            if n_trusts > 0:
                trust_in.zero_()
                if self._trust_trend_js_t is not None:
                    trust_in[:, self._trust_trend_js_t] = trend_vel[:, self._trust_trend_cols_t]
                if self._trust_sig_js_t is not None:
                    trust_in[:, self._trust_sig_js_t] = sig[:, self._trust_sig_cols_t]

                trust_running.mul_(td_exp).add_((1.0 - td_exp) * trust_in)
                trust_delta = (tg_exp * ts_exp * trust_running).sum(1)
                trust_total.add_(trust_delta * alive_f)

        results = env.get_results()

        # --- Aggregate multi-game results back to per-genome ---
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

    # Approximate score thresholds to REACH each tile.
    # tile 512 -> 3500 means "you need ~3500 score to hit 512".
    _TILE_SCORE_THRESHOLDS = {
        32: 150, 64: 400, 128: 900, 256: 1800,
        512: 3500, 1024: 7000, 2048: 15000,
    }

    # For a genome currently AT tile T, the proximity bonus should push
    # toward the NEXT tile's threshold, not the current tile's.
    # Otherwise a 512 genome with score 6900 (almost at 1024) gets the
    # same bonus as one with score 3500 — no gradient toward 1024.
    _NEXT_TILE_SCORE = {
        2: 150, 4: 150, 8: 150, 16: 150,  # anything small → push toward 32
        32: 400, 64: 900, 128: 1800, 256: 3500,
        512: 7000, 1024: 15000, 2048: 30000,  # keep pushing even past win
    }

    @torch.no_grad()
    def evolve(self, trust, survival_pct=20, trust_inherit=0.1, child_mutation_rate=0.05,
               tiles=None, scores=None):
        """Selection + cloning + mutation entirely on GPU.
        tiles: list of max tile values per genome (for ratchet bonus).
        scores: list of game scores per genome (for proximity bonus).
        Returns: (best_trust, median_trust, lowest_trust)
        """
        B = self.B
        ratchet_strength = self.cfg.get("ratchet_strength", 2.0)
        proximity_strength = self.cfg.get("proximity_strength", 1.0)

        # --- NORMALIZE TRUST ---
        t_mean = trust.mean()
        t_std = trust.std().clamp(min=1e-6)
        trust = (trust - t_mean) / t_std  # now ~ N(0, 1)

        # --- TILE RATCHET ---
        # Protects breakthrough genomes that hit rare high tiles.
        if tiles is not None and ratchet_strength > 0:
            tiles_t = torch.tensor(tiles, dtype=torch.float32, device=self.dev)
            log_tiles = torch.log2(tiles_t.clamp(min=1))
            pop_mean = log_tiles.mean()
            pop_std = log_tiles.std().clamp(min=0.5)
            z_scores = ((log_tiles - pop_mean) / pop_std).clamp(min=0)
            ratchet_bonus = z_scores.pow(2) * ratchet_strength
            trust = trust + ratchet_bonus

        # --- SCORE PROXIMITY BONUS ---
        # Gives a continuous "you're getting warmer" signal toward the NEXT tile.
        # A 256 genome with score 3000 (almost at 512) gets a bigger bonus than
        # one with score 1800 (just barely reached 256). This provides a dense
        # gradient across the gap between tile milestones.
        if tiles is not None and scores is not None and proximity_strength > 0:
            tiles_t = torch.tensor(tiles, dtype=torch.float32, device=self.dev)
            scores_t = torch.tensor(scores, dtype=torch.float32, device=self.dev)
            proximity_bonus = torch.zeros(B, dtype=torch.float32, device=self.dev)

            for tile_val, next_score in self._NEXT_TILE_SCORE.items():
                # Find genomes at this tile level
                at_tile = (tiles_t == tile_val)
                if not at_tile.any():
                    continue
                # How close is their score to the NEXT tile's threshold?
                # ratio: 0.0 = just reached current, 1.0 = at next, >1.0 = overshooting
                ratio = (scores_t / max(next_score, 1.0)).clamp(0, 1.5)
                # Quadratic bonus rewards proximity to the next breakthrough
                bonus = ratio.pow(2) * proximity_strength
                proximity_bonus += bonus * at_tile.float()

            trust = trust + proximity_bonus

        # Sort by trust
        sorted_trust, indices = trust.sort(descending=True)

        # Stats for printing (small CPU transfer)
        st = sorted_trust.cpu()
        best_trust = st[0].item()
        median_trust = st[B // 2].item()
        lowest_trust = st[-1].item()

        # --- SELECTIVE REPLACEMENT ---
        # Top 20%: elite — kept as-is, also source for clones
        # Middle 60%: stable reservoir — kept untouched
        # Bottom 20%: culled — replaced with mutated clones from elite
        k = max(1, int(B * survival_pct / 100.0))  # elite count
        cull_start = B - k  # bottom N% starts here

        elite_idx = indices[:k]          # top 20% indices
        middle_idx = indices[k:cull_start]  # middle 60% indices
        # indices[cull_start:] = bottom 20%, discarded

        # Sample parents from elite for the replacements
        elite_trust = trust[elite_idx]
        weights = elite_trust - elite_trust.min() + 1.0
        parent_choices = torch.multinomial(weights, k, replacement=True)
        replacement_parent_idx = elite_idx[parent_choices]

        # Build new ordering: elite (unchanged) + middle (unchanged) + clones
        keep_idx = torch.cat([elite_idx, middle_idx])
        new_idx = torch.cat([keep_idx, replacement_parent_idx])

        # Reorder all tensors
        all_params = ["w1", "b1", "w2", "b2",
                      "trend_momentum", "trust_gain", "trust_scale", "trust_decay"]
        for attr in all_params:
            setattr(self, attr, getattr(self, attr)[new_idx].contiguous())

        # Mutate ONLY the bottom k (replacement clones), not elite or middle
        rate = child_mutation_rate
        n_keep = len(keep_idx)
        for param in [self.w1, self.b1, self.w2, self.b2]:
            bottom = param[n_keep:]
            mask = torch.rand_like(bottom) < rate
            bottom.add_(torch.randn_like(bottom) * 0.1 * mask)

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

        return best_trust, median_trust, lowest_trust

    def sync_to_cpu(self, population, tiles=None, scores=None, trust=None):
        """Sync GPU tensors back to CPU genome objects (for checkpointing).
        Optionally writes last generation's tiles/scores/trust to genomes.
        """
        w1 = self.w1.cpu().tolist()
        b1 = self.b1.cpu().tolist()
        w2 = self.w2.cpu().tolist()
        b2 = self.b2.cpu().tolist()
        tm = self.trend_momentum.cpu().tolist()
        tg = self.trust_gain.cpu().tolist()
        ts = self.trust_scale.cpu().tolist()
        td = self.trust_decay.cpu().tolist()

        # Convert trust from GPU if provided
        trust_cpu = trust.cpu().tolist() if trust is not None else None

        # Ensure population has the right number of genomes
        while len(population.genomes) < self.B:
            population.genomes.append(population.genomes[0].clone())
        population.genomes = population.genomes[:self.B]
        population.size = self.B

        for i, g in enumerate(population.genomes):
            g.controller.w1 = w1[i]
            g.controller.b1 = b1[i]
            g.controller.w2 = w2[i]
            g.controller.b2 = b2[i]
            # Update input_size to match actual w1 shape (V3 encoder changes this)
            g.controller.input_size = len(w1[i][0])
            g.controller.hidden_size = len(b1[i])
            # Game results from last evaluation
            if tiles is not None:
                g.max_tile = tiles[i]
            if scores is not None:
                g.game_score = scores[i]
            if trust_cpu is not None:
                g.trust = trust_cpu[i]
            # Trend proteins (dynamic indices)
            for j, idx in enumerate(self._trend_indices):
                g.proteins[idx].params["momentum"] = tm[i][j]
            # Trust modifier proteins (dynamic indices)
            for j, idx in enumerate(self._trust_indices):
                g.proteins[idx].params["gain"] = tg[i][j]
                g.proteins[idx].params["scale"] = ts[i][j]
                g.proteins[idx].params["decay"] = td[i][j]

    def resize(self, new_size, trust_inherit=0.1, child_mutation_rate=0.05):
        """Resize population on GPU. Trims worst or pads from elite."""
        old = self.B
        if new_size == old:
            return

        if new_size < old:
            # Keep first new_size (assumes already sorted by fitness or doesn't matter)
            self.w1 = self.w1[:new_size].clone()
            self.b1 = self.b1[:new_size].clone()
            self.w2 = self.w2[:new_size].clone()
            self.b2 = self.b2[:new_size].clone()
            self.trend_momentum = self.trend_momentum[:new_size].clone()
            self.trust_gain = self.trust_gain[:new_size].clone()
            self.trust_scale = self.trust_scale[:new_size].clone()
            self.trust_decay = self.trust_decay[:new_size].clone()
        else:
            # Pad by cloning from top 20% with mutation
            needed = new_size - old
            elite_k = max(1, int(old * 0.2))
            elite_idx = torch.randint(0, elite_k, (needed,), device=self.dev)
            for attr in ["w1", "b1", "w2", "b2", "trend_momentum", "trust_gain", "trust_scale", "trust_decay"]:
                existing = getattr(self, attr)
                new_rows = existing[elite_idx].clone()
                # Mutate the new rows
                mask = torch.rand_like(new_rows) < child_mutation_rate
                new_rows.add_(torch.randn_like(new_rows) * 0.1 * mask)
                setattr(self, attr, torch.cat([existing, new_rows], dim=0))
            # Re-clamp protein params
            self.trend_momentum.clamp_(0.0, 0.99)
            self.trust_gain.clamp_(0.1, 10.0)
            self.trust_scale.clamp_(-5.0, 5.0)
            self.trust_decay.clamp_(0.0, 0.999)

        self.B = new_size


# ================================================================
# Legacy wrapper (for GUI app compatibility)
# ================================================================
def run_generation_gpu(population, config, device="cuda"):
    """Run one generation on GPU using legacy per-call interface."""
    evolver = GPUEvolver(population, config, device=device)
    tiles, scores, trust = evolver.run_generation()
    trust_cpu = trust.cpu().tolist()
    for i, g in enumerate(population.genomes):
        g.trust = trust_cpu[i]
        g.max_tile = tiles[i]
        g.game_score = scores[i]
    return tiles, scores
