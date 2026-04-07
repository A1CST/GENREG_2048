# ================================================================
# GENREG Training Logger
# ================================================================
# Structured logging for long training runs.
# Designed to diagnose evolutionary plateaus and track breakthroughs.
#
# Log structure:
#   - Per-generation one-liner (every gen)
#   - Population snapshot (every 100 gens)
#   - Milestone events (new best tile, first time reaching a tile)
#   - Cull reports (when a high-tile genome drops to bottom 20%)
#
# Files rotate every 10,000 generations to keep each file manageable.
# ================================================================

import os
import time
from datetime import datetime
from collections import defaultdict


class TrainingLogger:
    def __init__(self, log_dir="logs", rotate_every=10000):
        self.log_dir = log_dir
        self.rotate_every = rotate_every
        os.makedirs(log_dir, exist_ok=True)

        self._file = None
        self._file_gen_start = 0
        self._session_start = datetime.now()
        self._session_id = self._session_start.strftime("%Y%m%d_%H%M%S")

        # Tracking state for diagnostics
        self._prev_top_ids = set()      # genome IDs in top 20% last gen
        self._prev_genome_tiles = {}    # {genome_id: max_tile} from last gen
        self._best_tile_ever = 0
        self._milestone_tiles = set()
        self._snapshot_interval = 100

    def _get_file(self, generation):
        """Get the current log file, rotating if needed."""
        chunk = (generation // self.rotate_every) * self.rotate_every
        if self._file is not None and chunk == self._file_gen_start:
            return self._file

        # Close old file
        if self._file is not None:
            self._file.close()

        self._file_gen_start = chunk
        filename = f"train_{self._session_id}_gen{chunk:07d}.log"
        path = os.path.join(self.log_dir, filename)
        self._file = open(path, "a", buffering=1)  # line-buffered

        # Write header for new file
        self._file.write(f"# GENREG Training Log — Session {self._session_id}\n")
        self._file.write(f"# Started: {self._session_start.isoformat()}\n")
        self._file.write(f"# Generations: {chunk} — {chunk + self.rotate_every - 1}\n")
        self._file.write(f"# Format: GEN | best_tile avg_tile med_tile | best_score avg_score | trust_best trust_med trust_low | time_s | dist\n")
        self._file.write("#\n")
        return self._file

    def log_generation(self, generation, tiles, scores, trust_best, trust_med,
                        trust_low, gen_time, elapsed_total, genome_ids=None):
        """Write a compact one-liner per generation."""
        f = self._get_file(generation)
        n = len(tiles)
        best_tile = max(tiles)
        avg_tile = sum(tiles) / n
        sorted_t = sorted(tiles)
        med_tile = sorted_t[n // 2]
        best_score = max(scores)
        avg_score = sum(scores) / n

        # Tile distribution compact: "512:2 256:15 128:20 64:10 32:3"
        dist = defaultdict(int)
        for t in tiles:
            dist[t] += 1
        dist_str = " ".join(f"{t}:{c}" for t, c in sorted(dist.items(), reverse=True))

        # Hit rate for key thresholds
        rates = []
        for thresh in [128, 256, 512, 1024]:
            count = sum(1 for t in tiles if t >= thresh)
            if count > 0:
                rates.append(f"≥{thresh}:{count/n*100:.0f}%")
        rate_str = " ".join(rates) if rates else ""

        f.write(
            f"GEN {generation:>7} | "
            f"tile {best_tile:>5} avg:{avg_tile:>6.0f} med:{med_tile:>5} | "
            f"score {best_score:>6} avg:{avg_score:>6.0f} | "
            f"trust {trust_best:>10.1f} {trust_med:>10.1f} {trust_low:>10.1f} | "
            f"{gen_time:.2f}s {elapsed_total:>8.0f}s | "
            f"{dist_str} | {rate_str}\n"
        )

        # --- Milestone detection ---
        if best_tile > self._best_tile_ever:
            self._log_milestone(f, generation, best_tile, tiles, scores,
                                trust_best, elapsed_total, genome_ids)
            self._best_tile_ever = best_tile

        # First time a tile count is reached
        for tile_val in [64, 128, 256, 512, 1024, 2048]:
            if tile_val not in self._milestone_tiles and any(t >= tile_val for t in tiles):
                self._milestone_tiles.add(tile_val)
                count = sum(1 for t in tiles if t >= tile_val)
                f.write(
                    f"  *** FIRST {tile_val} at gen {generation} "
                    f"({elapsed_total:.0f}s) — {count}/{n} genomes ({count/n*100:.1f}%)\n"
                )

        # --- Population snapshot every N gens ---
        if generation % self._snapshot_interval == 0:
            self._log_snapshot(f, generation, tiles, scores, trust_best,
                                trust_med, trust_low, elapsed_total, genome_ids)

        # --- Cull detection: did a high-tile genome drop out? ---
        if genome_ids is not None:
            self._check_culls(f, generation, tiles, genome_ids, n)

    def _log_milestone(self, f, generation, new_best, tiles, scores,
                        trust_best, elapsed, genome_ids):
        """Detailed log when a new all-time best tile is reached."""
        n = len(tiles)
        f.write(f"\n{'='*80}\n")
        f.write(f"  NEW BEST TILE: {new_best} at generation {generation} ({elapsed:.0f}s)\n")
        f.write(f"  Previous best: {self._best_tile_ever}\n")
        f.write(f"  Population: {n} genomes\n")

        # Which genomes hit it
        hit_indices = [i for i, t in enumerate(tiles) if t == new_best]
        f.write(f"  Genomes that hit {new_best}: {len(hit_indices)}\n")
        for idx in hit_indices[:5]:  # first 5
            gid = genome_ids[idx] if genome_ids else f"#{idx}"
            f.write(f"    Genome {gid}: score={scores[idx]}\n")

        # Tile distribution at this moment
        dist = defaultdict(int)
        for t in tiles:
            dist[t] += 1
        f.write(f"  Distribution: {dict(sorted(dist.items(), reverse=True))}\n")
        f.write(f"  Trust: best={trust_best:.1f}\n")
        f.write(f"{'='*80}\n\n")

    def _log_snapshot(self, f, generation, tiles, scores, trust_best,
                       trust_med, trust_low, elapsed, genome_ids):
        """Population health snapshot every N generations."""
        n = len(tiles)
        avg_tile = sum(tiles) / n
        avg_score = sum(scores) / n
        sorted_tiles = sorted(tiles)

        dist = defaultdict(int)
        for t in tiles:
            dist[t] += 1

        f.write(f"\n--- SNAPSHOT gen {generation} ({elapsed:.0f}s) ---\n")
        f.write(f"  Avg tile: {avg_tile:.0f} | Avg score: {avg_score:.0f}\n")
        f.write(f"  Tile range: {sorted_tiles[0]} — {sorted_tiles[-1]}\n")
        f.write(f"  Trust: best={trust_best:.1f} med={trust_med:.1f} low={trust_low:.1f} "
                f"spread={trust_best-trust_low:.1f}\n")

        # Distribution
        f.write(f"  Tiles: {dict(sorted(dist.items(), reverse=True))}\n")

        # Hit rates
        for thresh in [128, 256, 512, 1024, 2048]:
            count = sum(1 for t in tiles if t >= thresh)
            if count > 0:
                f.write(f"  ≥{thresh}: {count}/{n} ({count/n*100:.1f}%)\n")

        # Percentiles
        p10 = sorted_tiles[n // 10]
        p25 = sorted_tiles[n // 4]
        p75 = sorted_tiles[3 * n // 4]
        p90 = sorted_tiles[9 * n // 10]
        f.write(f"  Percentiles: p10={p10} p25={p25} med={sorted_tiles[n//2]} p75={p75} p90={p90}\n")

        # Diversity: how many distinct tile values in the population
        unique_tiles = len(set(tiles))
        f.write(f"  Diversity: {unique_tiles} distinct tile outcomes\n")

        # Top 5 genomes
        if genome_ids:
            paired = sorted(zip(tiles, scores, genome_ids), reverse=True)
            f.write(f"  Top 5: ")
            for tile, score, gid in paired[:5]:
                f.write(f"[ID:{gid} tile:{tile} score:{score}] ")
            f.write("\n")

        f.write(f"---\n\n")

    def _check_culls(self, f, generation, tiles, genome_ids, n):
        """Detect when a genome that previously hit a high tile disappears."""
        # Current top 20% IDs
        cutoff = max(1, n // 5)
        paired = sorted(zip(tiles, genome_ids), reverse=True)
        current_top_ids = set(gid for _, gid in paired[:cutoff])
        current_tile_map = {gid: tile for tile, gid in zip(tiles, genome_ids)}

        # Check if any previous high-tile genomes were culled
        for gid, prev_tile in self._prev_genome_tiles.items():
            if prev_tile >= 512 and gid not in current_tile_map:
                # This genome is completely gone (not even in population)
                f.write(
                    f"  ! CULLED: genome {gid} (prev tile={prev_tile}) "
                    f"no longer in population at gen {generation}\n"
                )
            elif prev_tile >= 512 and gid in current_tile_map:
                curr_tile = current_tile_map[gid]
                was_top = gid in self._prev_top_ids
                is_top = gid in current_top_ids
                if was_top and not is_top:
                    f.write(
                        f"  ! DEMOTED: genome {gid} dropped from top 20% "
                        f"(prev={prev_tile}, now={curr_tile}) at gen {generation}\n"
                    )

        # Update tracking for next generation
        self._prev_top_ids = current_top_ids
        self._prev_genome_tiles = dict(zip(genome_ids, tiles))

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
