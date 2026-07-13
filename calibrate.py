#!/usr/bin/env python3
"""Calibration helper.

Screenshot mode (source: screencap):
  python calibrate.py                 # capture screen, draw the grid overlay
  python calibrate.py --image shot.png# use an existing screenshot instead
  python calibrate.py --learn-colors  # print mean RGB of each filled cell

Camera mode (source: camera -- for the screenshot-blocked app):
  python calibrate.py --camera        # click the 4 board corners, save preview
  python calibrate.py --camera --index 1

In camera mode a window opens on the live feed: click the board corners in the
order TL, TR, BR, BL, then press Enter. The corners are printed for you to paste
into `camera.corners` in config.yaml, and warp_preview.png shows the flattened
board with the 4x4 grid drawn on top -- every green box must sit on one tile.
"""

from __future__ import annotations

import argparse

import numpy as np
import yaml
from PIL import Image, ImageDraw

from main import build_reader, build_screen_geometry
from src.adb import ADB


def _draw_grid(img: Image.Image, geo) -> Image.Image:
    draw = ImageDraw.Draw(img)
    for r in range(4):
        for c in range(4):
            box = geo.cell_box(r, c)
            draw.rectangle(box, outline=(0, 255, 0), width=3)
            cx, cy = geo.cell_center(r, c)
            draw.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), fill=(255, 0, 0))
    return img


def calibrate_camera(cfg: dict, index: int) -> int:
    from src.camera import canonical_geometry, pick_corners_interactive, warp_board

    corners = pick_corners_interactive(index)
    print("\nPaste this into config.yaml under camera.corners:")
    print("  corners: " + str([[round(x, 1), round(y, 1)] for x, y in corners]))

    warp_size = cfg["camera"].get("warp_size", 640)
    # Re-grab one frame to render the preview.
    from src.camera import CameraSource
    cam = CameraSource(index=index, corners=corners, warp_size=warp_size, flush_frames=3)
    try:
        warped = np.asarray(cam.grab().convert("RGB"))
    finally:
        cam.release()

    geo = canonical_geometry(warp_size, cfg["camera"].get("cell_fill", 0.6))
    preview = _draw_grid(Image.fromarray(warped), geo)
    preview.save("warp_preview.png")
    print("Saved warp_preview.png. Each green box must land on exactly one tile.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--camera", action="store_true", help="calibrate camera corners")
    parser.add_argument("--index", type=int, default=None, help="camera index (camera mode)")
    parser.add_argument("--image", default=None, help="use this screenshot instead of the device")
    parser.add_argument("--out", default="overlay.png")
    parser.add_argument("--learn-colors", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    if args.camera:
        index = args.index if args.index is not None else cfg["camera"].get("index", 0)
        return calibrate_camera(cfg, index)

    geo = build_screen_geometry(cfg)
    if args.image:
        img = Image.open(args.image).convert("RGB")
    else:
        adb = ADB(serial=cfg["device"].get("serial"), adb_path=cfg["device"].get("adb_path", "adb"))
        adb.wait_for_device()
        img = adb.screencap()

    reader = build_reader({**cfg, "vision": {**cfg["vision"], "reader": "color"}}, geo)

    if args.learn_colors:
        samples = reader.sample_colors(img)
        print("Filled cells (row,col) -> mean RGB:")
        for (r, c), rgb in samples.items():
            print(f"  ({r},{c}): {list(rgb)}")
        print(
            "\nTip: match these RGBs to the digits you see on screen and fill in\n"
            "vision.color_palette in config.yaml, e.g.  8: [40, 130, 230]"
        )
        return 0

    _draw_grid(img, geo).save(args.out)
    print(f"Saved overlay to {args.out}. Check that dots are centred on tiles.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
