"""Headless tests + a full self-play benchmark for the engine.

Run:
    python -m pytest tests/ -q
    python tests/test_solver.py            # play benchmark games, print stats
"""

from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.solver import (  # noqa: E402
    Solver,
    apply_move,
    empty_cells,
    is_game_over,
    legal_moves,
    spawn_tile,
)


# --------------------------------------------------------------------------- #
# Mechanics
# --------------------------------------------------------------------------- #
def test_merge_left():
    board = (
        2, 2, 0, 0,
        4, 0, 4, 0,
        8, 8, 8, 8,
        0, 0, 0, 2,
    )
    new, moved, gained = apply_move(board, "left")
    assert moved
    assert new[0:4] == (4, 0, 0, 0)
    assert new[4:8] == (8, 0, 0, 0)
    assert new[8:12] == (16, 16, 0, 0)
    assert new[12:16] == (2, 0, 0, 0)
    assert gained == 4 + 8 + 16 + 16


def test_no_move_is_detected():
    board = (
        2, 4, 8, 16,
        4, 8, 16, 2,
        8, 16, 2, 4,
        16, 2, 4, 8,
    )
    for m in ("up", "down", "left", "right"):
        _, moved, _ = apply_move(board, m)
        assert not moved
    assert is_game_over(board)
    assert legal_moves(board) == []


def test_right_merge_order():
    board = (
        2, 2, 2, 2,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
    )
    new, moved, gained = apply_move(board, "right")
    assert moved
    assert new[0:4] == (0, 0, 4, 4)
    assert gained == 8


def test_solver_returns_legal_move():
    board = (
        2, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 4,
    )
    solver = Solver(spawn_distribution={2: 0.75, 8: 0.25}, max_depth=3)
    move = solver.best_move(board)
    assert move in legal_moves(board)


# --------------------------------------------------------------------------- #
# Self-play benchmark (also usable as a smoke test)
# --------------------------------------------------------------------------- #
def play_game(solver: Solver, rng: random.Random) -> dict:
    board = (0,) * 16
    board = spawn_tile(board, solver.spawn, rng)
    board = spawn_tile(board, solver.spawn, rng)
    moves = 0
    while not is_game_over(board):
        move = solver.best_move(board)
        if move is None:
            break
        board, moved, _ = apply_move(board, move)
        if not moved:
            break
        board = spawn_tile(board, solver.spawn, rng)
        moves += 1
    return {"max_tile": max(board), "moves": moves, "sum": sum(board)}


def test_self_play_reaches_reasonable_tile():
    solver = Solver(spawn_distribution={2: 0.9, 4: 0.1}, max_depth=3)
    rng = random.Random(0)
    result = play_game(solver, rng)
    # Even at shallow depth on classic rules the AI should comfortably pass 256.
    assert result["max_tile"] >= 256


if __name__ == "__main__":
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    solver = Solver(spawn_distribution={2: 0.9, 4: 0.1}, max_depth=4)
    rng = random.Random(42)
    maxes, sums, moves = [], [], []
    for i in range(games):
        r = play_game(solver, rng)
        maxes.append(r["max_tile"])
        sums.append(r["sum"])
        moves.append(r["moves"])
        print(f"game {i+1:>3}: max_tile={r['max_tile']:>5} moves={r['moves']:>4} sum={r['sum']}")
    print("\n--- summary over", games, "games (standard 2/4 spawn) ---")
    print("best max_tile:", max(maxes))
    print("median max_tile:", statistics.median(maxes))
    print("mean moves:", round(statistics.mean(moves), 1))
