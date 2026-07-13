"""2048 game logic + expectimax AI.

Board representation: a tuple of 16 ints in row-major order.
0 means an empty cell, other values are the actual tile numbers (2, 4, 8, ...).

The engine is deliberately independent from vision/ADB so it can be unit
tested and benchmarked headlessly (see tests/test_solver.py).
"""

from __future__ import annotations

import math
import random
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

Board = Tuple[int, ...]
MOVES = ("up", "down", "left", "right")


# --------------------------------------------------------------------------- #
# Core mechanics
# --------------------------------------------------------------------------- #
def _compress_and_merge(line: Sequence[int]) -> Tuple[Tuple[int, ...], int]:
    """Slide a single row to the LEFT, merging equal neighbours once.

    Returns the new 4-length row and the score gained by merges.
    """
    tiles = [v for v in line if v]
    result: List[int] = []
    gained = 0
    i = 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            merged = tiles[i] * 2
            result.append(merged)
            gained += merged
            i += 2
        else:
            result.append(tiles[i])
            i += 1
    result.extend([0] * (4 - len(result)))
    return tuple(result), gained


def _rows(board: Board) -> List[Tuple[int, ...]]:
    return [board[r * 4:r * 4 + 4] for r in range(4)]


def _from_rows(rows: Sequence[Sequence[int]]) -> Board:
    return tuple(v for row in rows for v in row)


def _transpose(board: Board) -> Board:
    return tuple(board[c * 4 + r] for r in range(4) for c in range(4))


@lru_cache(maxsize=None)
def apply_move(board: Board, move: str) -> Tuple[Board, bool, int]:
    """Apply a move. Returns (new_board, moved, gained_score)."""
    if move == "left":
        new_rows, gained = zip(*[_compress_and_merge(r) for r in _rows(board)])
        new = _from_rows(new_rows)
        return new, new != board, sum(gained)
    if move == "right":
        new_rows, gained = zip(
            *[
                (tuple(reversed(m)), g)
                for m, g in (_compress_and_merge(tuple(reversed(r))) for r in _rows(board))
            ]
        )
        new = _from_rows(new_rows)
        return new, new != board, sum(gained)
    if move == "up":
        t = _transpose(board)
        moved_board, moved, gained = apply_move(t, "left")
        return _transpose(moved_board), moved, gained
    if move == "down":
        t = _transpose(board)
        moved_board, moved, gained = apply_move(t, "right")
        return _transpose(moved_board), moved, gained
    raise ValueError(f"unknown move: {move}")


def legal_moves(board: Board) -> List[str]:
    return [m for m in MOVES if apply_move(board, m)[1]]


def is_game_over(board: Board) -> bool:
    return len(legal_moves(board)) == 0


def empty_cells(board: Board) -> List[int]:
    return [i for i, v in enumerate(board) if v == 0]


def spawn_tile(board: Board, distribution: Dict[int, float], rng: random.Random) -> Board:
    """Place one random tile according to `distribution` (value -> probability)."""
    empties = empty_cells(board)
    if not empties:
        return board
    idx = rng.choice(empties)
    values, weights = zip(*distribution.items())
    value = rng.choices(values, weights=weights, k=1)[0]
    lst = list(board)
    lst[idx] = value
    return tuple(lst)


# --------------------------------------------------------------------------- #
# Heuristic evaluation
# --------------------------------------------------------------------------- #
# Weights follow the well-tested ov3y/Yiyuan-Lee 2048 heuristic, which balances
# keeping the board open, smooth (easy to merge) and monotonic (values ordered
# toward one corner). A hand-tuned "snake" matrix was tried and made deeper
# search *worse* -- it gave the search something to exploit -- so it was dropped.
_W_EMPTY = 2.7
_W_SMOOTH = 0.1
_W_MONO = 1.0
_W_MAX = 1.0


def _log(v: int) -> float:
    return math.log2(v) if v else 0.0


def _monotonicity(board: Board) -> float:
    """Reward rows/columns that are monotonic (all increasing or decreasing)."""
    total = 0.0
    rows = _rows(board)
    cols = _rows(_transpose(board))
    for line in rows + cols:
        inc = dec = 0.0
        for a, b in zip(line, line[1:]):
            la, lb = _log(a), _log(b)
            if la > lb:
                dec += lb - la
            elif lb > la:
                inc += la - lb
        total += max(inc, dec)
    return total


def _smoothness(board: Board) -> float:
    """Penalise big differences between neighbouring tiles (easier to merge)."""
    total = 0.0
    for r in range(4):
        for c in range(4):
            v = board[r * 4 + c]
            if not v:
                continue
            lv = _log(v)
            if c + 1 < 4 and board[r * 4 + c + 1]:
                total -= abs(lv - _log(board[r * 4 + c + 1]))
            if r + 1 < 4 and board[(r + 1) * 4 + c]:
                total -= abs(lv - _log(board[(r + 1) * 4 + c]))
    return total


def evaluate(board: Board) -> float:
    empties = empty_cells(board)
    open_term = math.log(len(empties)) if empties else 0.0
    return (
        _W_EMPTY * open_term
        + _W_SMOOTH * _smoothness(board)
        + _W_MONO * _monotonicity(board)
        + _W_MAX * _log(max(board))
    )


# --------------------------------------------------------------------------- #
# Expectimax search
# --------------------------------------------------------------------------- #
class Solver:
    def __init__(
        self,
        spawn_distribution: Optional[Dict[int, float]] = None,
        max_depth: int = 4,
        chance_branch_cap: int = 6,
    ):
        # Default matches classic 2048; override for variants (e.g. that spawn 8s).
        self.spawn = spawn_distribution or {2: 0.9, 4: 0.1}
        self.max_depth = max_depth
        self.chance_branch_cap = chance_branch_cap

    def _adaptive_depth(self, board: Board) -> int:
        n = len(empty_cells(board))
        if n >= 6:
            return max(2, self.max_depth - 2)
        if n >= 3:
            return max(2, self.max_depth - 1)
        return self.max_depth

    def rank_moves(self, board: Board) -> List[Tuple[str, float]]:
        """Return legal moves sorted best-first with their expectimax values."""
        depth = self._adaptive_depth(board)
        scored = []
        for move in MOVES:
            new_board, moved, gained = apply_move(board, move)
            if not moved:
                continue
            value = gained * 0.01 + self._expect(new_board, depth - 1)
            scored.append((move, value))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def best_move(self, board: Board) -> Optional[str]:
        ranked = self.rank_moves(board)
        return ranked[0][0] if ranked else None

    def _max_node(self, board: Board, depth: int) -> float:
        if depth <= 0:
            return evaluate(board)
        best = -math.inf
        any_move = False
        for move in MOVES:
            new_board, moved, gained = apply_move(board, move)
            if not moved:
                continue
            any_move = True
            best = max(best, gained * 0.01 + self._expect(new_board, depth - 1))
        return best if any_move else evaluate(board)

    def _expect(self, board: Board, depth: int) -> float:
        empties = empty_cells(board)
        if not empties or depth <= 0:
            return evaluate(board)

        # Limit branching in wide-open boards: sample a subset of empty cells.
        if len(empties) > self.chance_branch_cap:
            empties = random.sample(empties, self.chance_branch_cap)

        total = 0.0
        for idx in empties:
            for value, prob in self.spawn.items():
                lst = list(board)
                lst[idx] = value
                total += prob * self._max_node(tuple(lst), depth - 1)
        return total / len(empties)
