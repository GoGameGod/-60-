#!/usr/bin/env python3
"""Entry point: load config, wire everything up, and play N games.

Usage:
    python main.py                 # play using config.yaml
    python main.py --tries 100     # override number of games
    python main.py --dry-run       # read the board once and print it (no swipes)
"""

from __future__ import annotations

import argparse
import random
import sys

import yaml

from src.adb import ADB
from src.bot import Bot, BotConfig
from src.humanize import SwipeProfile, TimingProfile
from src.solver import Solver
from src.vision import BoardReader, Geometry


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_geometry(cfg: dict) -> Geometry:
    g = cfg["geometry"]
    return Geometry(
        first_center_x=g["first_center_x"],
        first_center_y=g["first_center_y"],
        pitch_x=g["pitch_x"],
        pitch_y=g["pitch_y"],
        cell_size=g["cell_size"],
    )


def build_reader(cfg: dict, geo: Geometry) -> BoardReader:
    v = cfg["vision"]
    palette = {int(k): tuple(val) for k, val in (v.get("color_palette") or {}).items()}
    return BoardReader(
        geometry=geo,
        reader=v.get("reader", "ocr"),
        color_palette=palette,
        empty_luma_max=v.get("empty_luma_max", 55),
    )


def build_swipe(cfg: dict, geo: Geometry) -> SwipeProfile:
    s = cfg["swipe"]
    cx = s.get("center_x")
    cy = s.get("center_y")
    if cx is None:
        cx = geo.first_center_x + geo.pitch_x * 3 // 2
    if cy is None:
        cy = geo.first_center_y + geo.pitch_y * 3 // 2
    return SwipeProfile(
        center_x=cx,
        center_y=cy,
        distance=s.get("distance", 320),
        distance_jitter=s.get("distance_jitter", 0.18),
        start_jitter=s.get("start_jitter", 26),
        angle_jitter_deg=s.get("angle_jitter_deg", 6.0),
        duration_min=s.get("duration_min", 45),
        duration_max=s.get("duration_max", 95),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Native ADB bot for 2048.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--tries", type=int, default=None, help="override max_tries")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="read board once, print, exit")
    args = parser.parse_args()

    cfg = load_config(args.config)
    geo = build_geometry(cfg)

    adb = ADB(serial=cfg["device"].get("serial"), adb_path=cfg["device"].get("adb_path", "adb"))
    adb.wait_for_device()
    reader = build_reader(cfg, geo)

    if args.dry_run:
        board = reader.read(adb.screencap())
        print("Detected board:")
        for r in range(4):
            print("  " + " ".join(f"{board[r*4+c]:>5}" for c in range(4)))
        return 0

    spawn = {int(k): float(v) for k, v in cfg["spawn_distribution"].items()}
    solver = Solver(
        spawn_distribution=spawn,
        max_depth=cfg["solver"].get("max_depth", 4),
        chance_branch_cap=cfg["solver"].get("chance_branch_cap", 6),
    )
    t = cfg["timing"]
    timing = TimingProfile(
        base_min=t.get("base_min", 0.09),
        base_max=t.get("base_max", 0.22),
        long_pause_chance=t.get("long_pause_chance", 0.06),
        long_pause_min=t.get("long_pause_min", 0.5),
        long_pause_max=t.get("long_pause_max", 1.4),
    )
    swipe = build_swipe(cfg, geo)

    game = cfg["game"]
    bot_cfg = BotConfig(
        new_game_button=tuple(game["new_game_button"]),
        max_tries=args.tries if args.tries is not None else game.get("max_tries", 50),
        stuck_threshold=game.get("stuck_threshold", 4),
        read_settle_s=game.get("read_settle_s", 0.12),
    )

    bot = Bot(
        adb=adb,
        reader=reader,
        solver=solver,
        timing=timing,
        swipe=swipe,
        config=bot_cfg,
        rng=random.Random(args.seed),
    )

    print(f"Starting run: up to {bot_cfg.max_tries} games. Ctrl-C to stop.\n")
    try:
        results = bot.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130

    best = max(results, key=lambda r: (r.max_tile, r.board_sum))
    print(f"\nDone. Best game: max_tile={best.max_tile}, board_sum={best.board_sum}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
