# ================================================================
# GENREG 2048 Environment
# ================================================================
# 4x4 grid, 4 actions (UP/DOWN/LEFT/RIGHT).
# Signals ONLY. Proteins (GENREG) handle all trust-based learning.
#
# Death conditions:
#   - No valid moves remaining (board full, no merges possible)
#   - Energy depletion (prevents infinite invalid-move loops)
#
# Optimizations:
#   - Pre-computed log2 lookup table
#   - Fast valid-move check (empty count first)
#   - Incremental empty_count tracking
#
# Goal: Reach the 2048 tile.
# ================================================================

import random

# Pre-computed log2 lookup: avoids math.log2() per signal per step
_LOG2_TABLE = {0: 0.0}
for _p in range(1, 18):
    _LOG2_TABLE[1 << _p] = _p / 11.0  # normalized by log2(2048)


class Game2048Env:
    def __init__(self, target_tile=2048):
        self.target_tile = target_tile
        self.reset()

    # ------------------------------------------------------------
    def reset(self):
        """Reset board to starting state with 2 random tiles."""
        self.board = [[0] * 4 for _ in range(4)]
        self.score = 0
        self.max_tile = 0
        self.moves_made = 0
        self.last_merge_count = 0
        self.alive = True
        self.won = False
        self.empty_count = 16

        # Energy system
        self.max_energy = 50
        self.energy = self.max_energy

        # Spawn two starting tiles
        self._spawn_tile()
        self._spawn_tile()

        self._update_max_tile()
        return self.get_signals()

    # ------------------------------------------------------------
    def _spawn_tile(self):
        """Spawn a 2 (90%) or 4 (10%) on a random empty cell."""
        empty = []
        for r in range(4):
            for c in range(4):
                if self.board[r][c] == 0:
                    empty.append((r, c))
        if not empty:
            return
        r, c = random.choice(empty)
        self.board[r][c] = 2 if random.random() < 0.9 else 4
        self.empty_count -= 1

    # ------------------------------------------------------------
    def _update_max_tile(self):
        """Update max tile on the board."""
        self.max_tile = max(self.board[r][c] for r in range(4) for c in range(4))

    # ------------------------------------------------------------
    @staticmethod
    def _slide_and_merge(line):
        """Slide a single row/column left and merge.
        Returns: (new_line, merge_score, merge_count)
        """
        non_zero = [x for x in line if x != 0]
        merged = []
        score = 0
        merges = 0
        i = 0
        while i < len(non_zero):
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                val = non_zero[i] * 2
                merged.append(val)
                score += val
                merges += 1
                i += 2
            else:
                merged.append(non_zero[i])
                i += 1
        while len(merged) < 4:
            merged.append(0)
        return merged, score, merges

    # ------------------------------------------------------------
    def _apply_move(self, direction):
        """Apply a move and return (new_board, score_gained, merge_count).
        direction: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        """
        board = self.board
        total_score = 0
        total_merges = 0

        if direction == 2:  # LEFT
            new_board = []
            for row in board:
                merged, s, m = self._slide_and_merge(row)
                new_board.append(merged)
                total_score += s
                total_merges += m
            return new_board, total_score, total_merges

        elif direction == 3:  # RIGHT
            new_board = []
            for row in board:
                merged, s, m = self._slide_and_merge(row[::-1])
                new_board.append(merged[::-1])
                total_score += s
                total_merges += m
            return new_board, total_score, total_merges

        elif direction == 0:  # UP — extract columns, slide, put back
            new_board = [row[:] for row in board]
            for c in range(4):
                col = [board[r][c] for r in range(4)]
                merged, s, m = self._slide_and_merge(col)
                for r in range(4):
                    new_board[r][c] = merged[r]
                total_score += s
                total_merges += m
            return new_board, total_score, total_merges

        elif direction == 1:  # DOWN — extract columns reversed
            new_board = [row[:] for row in board]
            for c in range(4):
                col = [board[3 - r][c] for r in range(4)]
                merged, s, m = self._slide_and_merge(col)
                for r in range(4):
                    new_board[3 - r][c] = merged[r]
                total_score += s
                total_merges += m
            return new_board, total_score, total_merges

    # ------------------------------------------------------------
    def _has_valid_moves(self):
        """Check if any move would change the board."""
        # Fast path: if there are empty cells, there's always a valid move
        if self.empty_count > 0:
            return True
        # Board is full — check for adjacent equal cells
        for r in range(4):
            for c in range(4):
                v = self.board[r][c]
                if c + 1 < 4 and v == self.board[r][c + 1]:
                    return True
                if r + 1 < 4 and v == self.board[r + 1][c]:
                    return True
        return False

    # ------------------------------------------------------------
    def step(self, action):
        """
        action: int (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
        Returns: (signals_dict, done)
        """
        if not self.alive:
            return self.get_signals(), True

        # Apply the move
        new_board, score_gained, merge_count = self._apply_move(action)
        moved = (new_board != self.board)

        if moved:
            # Count empties from the new board
            self.empty_count = sum(1 for r in range(4) for c in range(4) if new_board[r][c] == 0)
            self.board = new_board
            self.score += score_gained
            self.last_merge_count = merge_count
            self._spawn_tile()
            self._update_max_tile()

            # Energy reward: +3 per merge created
            self.energy += 3 * merge_count
            # Valid move costs 1 energy
            self.energy -= 1
        else:
            # Invalid move - double energy penalty
            self.last_merge_count = 0
            self.energy -= 2

        # Passive energy decay every step
        self.energy -= 0.01
        self.moves_made += 1

        # Check win condition
        if self.max_tile >= self.target_tile:
            self.won = True

        # Check death conditions
        if self.energy <= 0:
            self.alive = False
            return self.get_signals(), True

        if not self._has_valid_moves():
            self.alive = False
            return self.get_signals(), True

        return self.get_signals(), False

    # ------------------------------------------------------------
    def get_signals(self):
        """Return a GENREG-friendly signal dictionary of pure floats.
        Uses pre-computed log2 lookup table for speed.
        """
        signals = {}

        # Board cells: log2(value) / 11.0
        for r in range(4):
            for c in range(4):
                idx = r * 4 + c
                signals[f"cell_{idx}"] = _LOG2_TABLE.get(self.board[r][c], 0.0)

        # Meta signals
        signals["max_tile"] = _LOG2_TABLE.get(self.max_tile, 0.0)
        signals["empty_count"] = float(self.empty_count)
        signals["score"] = float(self.score)
        signals["moves_made"] = float(self.moves_made)
        signals["last_merge_count"] = float(self.last_merge_count)
        signals["alive"] = 1.0 if self.alive else 0.0

        return signals

    # ------------------------------------------------------------
    def _count_empty(self):
        """Count empty cells on the board."""
        return self.empty_count

    # ------------------------------------------------------------
    def get_board_copy(self):
        """Return a copy of the board for rendering."""
        return [row[:] for row in self.board]
