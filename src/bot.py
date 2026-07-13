"""Main orchestration loop: read -> think -> swipe -> repeat.

The bot never simulates the game blindly; every decision is based on a fresh
screenshot, so random spawns (including this variant's 8s) are always accounted
for. Illegal moves are detected by comparing the board before/after a swipe.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .adb import ADB
from .humanize import SwipeProfile, TimingProfile
from .solver import Board, Solver, is_game_over
from .vision import BoardReader


def _max_tile(board: Board) -> int:
    return max(board) if board else 0


def _score_estimate(board: Board) -> int:
    """Not the app's exact score, but a stable proxy for comparing games."""
    return sum(board)


@dataclass
class GameResult:
    max_tile: int
    moves: int
    board_sum: int
    board: Board


@dataclass
class BotConfig:
    new_game_button: Tuple[int, int]
    max_tries: int = 50
    stuck_threshold: int = 4
    read_settle_s: float = 0.12


class Bot:
    def __init__(
        self,
        adb: ADB,
        reader: BoardReader,
        solver: Solver,
        timing: TimingProfile,
        swipe: SwipeProfile,
        config: BotConfig,
        rng: Optional[random.Random] = None,
        on_move=None,
        on_game_end=None,
    ):
        self.adb = adb
        self.reader = reader
        self.solver = solver
        self.timing = timing
        self.swipe = swipe
        self.config = config
        self.rng = rng or random.Random()
        self.on_move = on_move
        self.on_game_end = on_game_end

    # -- one game ----------------------------------------------------------- #
    def play_one_game(self) -> GameResult:
        moves = 0
        stuck = 0
        board = self._read()
        best_max = _max_tile(board)

        while True:
            ranked = self.solver.rank_moves(board)
            if not ranked or is_game_over(board):
                break

            moved = False
            for move, _value in ranked:
                self._do_swipe(move)
                time.sleep(self.config.read_settle_s)
                new_board = self._read()
                if new_board != board:
                    board = new_board
                    moved = True
                    stuck = 0
                    break
                # Board unchanged -> that direction was illegal on-screen; try next.
                stuck += 1
                if stuck >= self.config.stuck_threshold:
                    break

            if not moved:
                break

            moves += 1
            best_max = max(best_max, _max_tile(board))
            if self.on_move:
                self.on_move(moves, board)
            self.timing.sleep(self.rng)

        result = GameResult(
            max_tile=best_max, moves=moves, board_sum=_score_estimate(board), board=board
        )
        if self.on_game_end:
            self.on_game_end(result)
        return result

    # -- many games --------------------------------------------------------- #
    def run(self) -> List[GameResult]:
        results: List[GameResult] = []
        for attempt in range(1, self.config.max_tries + 1):
            result = self.play_one_game()
            results.append(result)
            best = max(results, key=lambda r: (r.max_tile, r.board_sum))
            print(
                f"[try {attempt}/{self.config.max_tries}] "
                f"max_tile={result.max_tile} moves={result.moves} "
                f"sum={result.board_sum} | best so far max_tile={best.max_tile}"
            )
            if attempt < self.config.max_tries:
                self._start_new_game()
        return results

    # -- primitives --------------------------------------------------------- #
    def _read(self) -> Board:
        return self.reader.read(self.adb.screencap())

    def _do_swipe(self, direction: str) -> None:
        x1, y1, x2, y2, dur = self.swipe.gesture(direction, self.rng)
        self.adb.swipe(x1, y1, x2, y2, dur)

    def _start_new_game(self) -> None:
        bx, by = self.config.new_game_button
        # tiny human jitter on the button tap too
        self.adb.tap(bx + self.rng.randint(-6, 6), by + self.rng.randint(-6, 6))
        time.sleep(0.6)
