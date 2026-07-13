#!/usr/bin/env python3
"""Calibration helper.

  python calibrate.py                 # capture screen, draw the grid overlay
  python calibrate.py --image shot.png# use an existing screenshot instead
  python calibrate.py --learn-colors  # print mean RGB of each filled cell

Look at overlay.png: every red dot must sit in the centre of a tile and each
green square must stay inside one cell. Adjust the `geometry` block in
config.yaml until it lines up, then verify with `python main.py --dry-run`.
"""

from __future__ import annotations

import argparse

import yaml
from PIL import Image, ImageDraw

from main import build_geometry, build_reader
from src.adb import ADB


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--image", default=None, help="use this screenshot instead of the device")
    parser.add_argument("--out", default="overlay.png")
    parser.add_argument("--learn-colors", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    geo = build_geometry(cfg)

    if args.image:
        img = Image.open(args.image).convert("RGB")
    else:
        adb = ADB(serial=cfg["device"].get("serial"), adb_path=cfg["device"].get("adb_path", "adb"))
        adb.wait_for_device()
        img = adb.screencap()

    reader = build_reader({**cfg, "vision": {**cfg["vision"], "reader": "color"}}, geo)

    if args.learn_colors:
        board = reader.read(img) if cfg["vision"].get("color_palette") else None
        samples = reader.sample_colors(img)
        print("Filled cells (row,col) -> mean RGB:")
        for (r, c), rgb in samples.items():
            print(f"  ({r},{c}): {list(rgb)}")
        print(
            "\nTip: match these RGBs to the digits you see on screen and fill in\n"
            "vision.color_palette in config.yaml, e.g.  8: [40, 130, 230]"
        )
        if board:
            print(f"\nCurrent palette decodes board as: {board}")
        return 0

    draw = ImageDraw.Draw(img)
    for r in range(4):
        for c in range(4):
            box = geo.cell_box(r, c)
            draw.rectangle(box, outline=(0, 255, 0), width=3)
            cx, cy = geo.cell_center(r, c)
            draw.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), fill=(255, 0, 0))
    img.save(args.out)
    print(f"Saved overlay to {args.out}. Check that dots are centred on tiles.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
