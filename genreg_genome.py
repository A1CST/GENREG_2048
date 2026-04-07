# ================================================================
# GENREG v1.1 — Genome + Evolution Loop (2048 Edition)
# ================================================================
# - "Cull & Replace" Evolution
# - Inherited Trust with DECAY (Prevents Inflation)
# - Split Mutation Rates
# - Tracks max_tile and game_score instead of food_eaten
# ================================================================

import random
import copy
import math
from genreg_proteins import run_protein_cascade
from genreg_controller import GENREGController


# ================================================================
# GENOME CLASS
# ================================================================
class GENREGGenome:
    def __init__(self, proteins, controller):
        self.proteins = proteins            # list[Protein]
        self.controller = controller        # GENREGController
        self.trust = 0.0                    # fitness scalar
        self.max_tile = 0                   # highest tile achieved in last game
        self.game_score = 0                 # score in last game
        self.step_count = 0                 # moves made in last game
        self.id = random.randint(1000, 9999)

    # ------------------------------------------------------------
    def reset_trust(self):
        self.trust = 0.0

    # ------------------------------------------------------------
    def clone(self):
        """
        Deep copy genome - genetic information only.
        Protein states (perception) reset, but PARAMETERS persist.
        """
        new_genome = GENREGGenome(
            proteins=[copy.deepcopy(p) for p in self.proteins],
            controller=self.controller.clone()
        )

        # NOTE: Trust inheritance is handled in evolve()
        new_genome.trust = 0.0

        # Inherit parent's game stats (overwritten after next evaluation)
        new_genome.max_tile = self.max_tile
        new_genome.game_score = self.game_score
        new_genome.step_count = self.step_count

        # Reset internal biological states (perception is individual)
        for p in new_genome.proteins:
            # Reset numeric states to starting values
            for key in p.state:
                if isinstance(p.state[key], (int, float)):
                    if key == "running_mean": p.state[key] = 0.0
                    elif key == "running_max": p.state[key] = 1.0
                    elif key == "count": p.state[key] = 0
                    elif key == "accum": p.state[key] = 0.0
                    elif key == "velocity": p.state[key] = 0.0
                    elif key == "running": p.state[key] = 0.0
                    else: p.state[key] = 0.0
                elif isinstance(p.state[key], bool):
                    p.state[key] = False
                elif p.state[key] is None:
                    p.state[key] = None

        return new_genome

    # ------------------------------------------------------------
    def mutate(self, rate=0.1):
        """Mutate both proteins AND their parameters."""
        for p in self.proteins:
            # Mutate hyperparameters with bounds
            for k in p.params:
                if random.random() < rate:
                    p.mutate_param(k, scale=0.2)

                    # Universal parameter bounds (numerical stability)
                    if k == "scale":
                        p.params[k] = max(min(p.params[k], 5.0), -5.0)

                    elif k == "gain":
                        p.params[k] = max(min(p.params[k], 10.0), 0.1)

                    elif k == "decay":
                        p.params[k] = max(min(p.params[k], 0.999), 0.0)

                    elif k == "threshold":
                        p.params[k] = max(min(p.params[k], 10.0), -10.0)

                    elif k == "momentum":
                        p.params[k] = max(min(p.params[k], 0.99), 0.0)

        # Mutate controller weights
        self.controller.mutate(rate)

        return self

    # ------------------------------------------------------------
    def forward(self, signals):
        _, trust_delta = run_protein_cascade(self.proteins, signals)
        self.trust += trust_delta
        return trust_delta


# ================================================================
# POPULATION MANAGER
# ================================================================
class GENREGPopulation:
    def __init__(self, template_proteins, input_size, hidden_size=32, output_size=4, size=20, mutation_rate=0.1):
        self.size = size

        # Init population
        self.genomes = []
        for _ in range(size):
            controller = GENREGController(input_size, hidden_size, output_size)
            g = GENREGGenome(
                proteins=[copy.deepcopy(p) for p in template_proteins],
                controller=controller
            )
            self.genomes.append(g)

        self.active = 0

    def get_active(self):
        return self.genomes[self.active]

    def next_genome(self):
        self.active = (self.active + 1) % self.size
        return self.get_active()

    # ------------------------------------------------------------
    # Evolution with Partial Trust Inheritance
    # ------------------------------------------------------------
    # Score thresholds for proximity bonus (same as GPU path)
    _TILE_SCORE_THRESHOLDS = {
        32: 150, 64: 400, 128: 900, 256: 1800,
        512: 3500, 1024: 7000, 2048: 15000,
    }

    def evolve(self, survival_pct=20, trust_inherit=0.1, child_mutation_rate=0.05,
                ratchet_strength=2.0, proximity_strength=1.0):
        """
        Evolution with tile ratchet and score proximity bonus.

        Args:
            survival_pct: Top N% are elite (1-100)
            trust_inherit: Children inherit this fraction of parent trust (0-1)
            child_mutation_rate: Mutation rate for offspring (0-1)
            ratchet_strength: Protection for high-tile genomes (0-10)
            proximity_strength: Bonus for scores approaching next tile (0-5)
        """

        # --- NORMALIZE TRUST ---
        import math
        trust_vals = [g.trust for g in self.genomes]
        t_mean = sum(trust_vals) / len(trust_vals)
        t_std = max(1e-6, (sum((t - t_mean)**2 for t in trust_vals) / len(trust_vals)) ** 0.5)
        for g in self.genomes:
            g.trust = (g.trust - t_mean) / t_std

        # --- TILE RATCHET ---
        if ratchet_strength > 0:
            tile_values = [g.max_tile for g in self.genomes]
            log_tiles = [math.log2(max(t, 1)) for t in tile_values]
            pop_mean = sum(log_tiles) / len(log_tiles)
            pop_std = max(0.5, (sum((x - pop_mean)**2 for x in log_tiles) / len(log_tiles)) ** 0.5)

            for i, g in enumerate(self.genomes):
                z = max(0.0, (log_tiles[i] - pop_mean) / pop_std)
                g.trust += z * z * ratchet_strength

        # --- SCORE PROXIMITY BONUS ---
        if proximity_strength > 0:
            for g in self.genomes:
                tile = g.max_tile
                score = g.game_score
                if tile in self._TILE_SCORE_THRESHOLDS:
                    target_score = self._TILE_SCORE_THRESHOLDS[tile]
                    ratio = min(2.0, max(0.0, score / max(target_score, 1.0)))
                    g.trust += ratio * ratio * proximity_strength

        # Sort by trust (highest first)
        self.genomes.sort(key=lambda g: g.trust, reverse=True)

        # Calculate statistics
        trust_values = [g.trust for g in self.genomes]
        best_trust = trust_values[0]
        median_trust = trust_values[len(trust_values)//2]
        lowest_trust = trust_values[-1]

        tile_values = [g.max_tile for g in self.genomes]
        best_tile = max(tile_values) if tile_values else 0
        median_tile = sorted(tile_values)[len(tile_values)//2]

        score_values = [g.game_score for g in self.genomes]
        best_score = max(score_values) if score_values else 0

        print(f"  > Evolution: Trust[Best={best_trust:.1f} | Med={median_trust:.1f} | Low={lowest_trust:.1f}] MaxTile[Best={best_tile} | Med={median_tile}] Score[Best={best_score}]")

        # --- SELECTIVE REPLACEMENT ---
        # Top 20%: elite parents — kept as-is, also source clones
        # Middle 60%: stable reservoir — kept untouched, no mutation
        # Bottom 20%: culled — replaced with mutated clones from top 20%
        elite_cutoff = max(1, int(self.size * survival_pct / 100.0))
        cull_cutoff = self.size - elite_cutoff  # bottom N% start here

        elite = self.genomes[:elite_cutoff]
        middle = self.genomes[elite_cutoff:cull_cutoff]
        # bottom = self.genomes[cull_cutoff:]  — discarded

        # Fitness-proportional weights for parent selection
        min_trust = min(g.trust for g in elite)
        fitness_weights = [g.trust - min_trust + 1.0 for g in elite]

        # Generate replacements for the culled bottom
        replacements = []
        for _ in range(self.size - elite_cutoff - len(middle)):
            parent = random.choices(elite, weights=fitness_weights, k=1)[0]
            child = parent.clone()
            child.trust = parent.trust * trust_inherit
            child.mutate(rate=child_mutation_rate)
            replacements.append(child)

        # Reassemble: elite (unchanged) + middle (unchanged) + fresh clones
        self.genomes = elite + middle + replacements
        self.active = 0
