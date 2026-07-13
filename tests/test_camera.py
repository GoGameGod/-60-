"""Verify the camera perspective-correction maps a tilted board to a flat grid.

Runs headless: we synthesize a board drawn into a quad, warp it back, and check
that the BoardReader (color mode) decodes the tiles from the corrected image.
Skipped automatically if opencv-python is not installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

cv2 = pytest.importorskip("cv2")

from PIL import Image  # noqa: E402

from src.camera import canonical_geometry, warp_board  # noqa: E402
from src.vision import BoardReader  # noqa: E402


# Distinct colours so the "color" reader can classify each value unambiguously.
PALETTE = {2: (60, 60, 60), 4: (230, 200, 60), 8: (40, 130, 230), 16: (200, 60, 60)}

BOARD = [
    2, 0, 4, 0,
    0, 8, 0, 16,
    16, 0, 8, 0,
    0, 4, 0, 2,
]


def _render_flat_board(size: int) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    pitch = size // 4
    pad = pitch // 6
    for r in range(4):
        for c in range(4):
            v = BOARD[r * 4 + c]
            if not v:
                continue
            y0, x0 = r * pitch + pad, c * pitch + pad
            y1, x1 = (r + 1) * pitch - pad, (c + 1) * pitch - pad
            img[y0:y1, x0:x1] = PALETTE[v]
    return img


def test_warp_recovers_board():
    size = 640
    flat = _render_flat_board(size)

    # Paint the flat board into a tilted quad inside a bigger camera frame.
    frame = np.full((900, 1200, 3), 10, dtype=np.uint8)
    dst_quad = np.float32([[200, 120], [980, 180], [1010, 760], [160, 800]])
    src_quad = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
    M = cv2.getPerspectiveTransform(src_quad, dst_quad)
    cv2.warpPerspective(flat, M, (1200, 900), dst=frame,
                        borderMode=cv2.BORDER_TRANSPARENT)

    # Now recover it using the same corners the bot would be given.
    corners = [tuple(p) for p in dst_quad.tolist()]
    recovered = warp_board(frame, corners, size)

    geo = canonical_geometry(size, cell_fill=0.5)
    reader = BoardReader(geo, reader="color", color_palette=PALETTE, empty_luma_max=25)
    decoded = reader.read(Image.fromarray(recovered))

    assert decoded == tuple(BOARD)
