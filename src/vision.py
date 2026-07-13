"""Turn a phone screenshot into a 4x4 board matrix.

Two independent readers are provided:

* ``ocr``   -- crops each cell and runs Tesseract on the digits. Works for any
               2048 variant with no per-value setup, but needs the tesseract
               binary installed.
* ``color`` -- maps each tile's background colour to a value using a calibrated
               palette in the config. Dependency-free and very fast, but the
               palette must be learned once per app/theme.

Grid geometry (where the 16 cells live on screen) comes from the config and is
tuned with ``calibrate.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

Board = Tuple[int, ...]


@dataclass
class Geometry:
    """Pixel layout of the board in the *captured* screenshot resolution."""
    first_center_x: int
    first_center_y: int
    pitch_x: int
    pitch_y: int
    cell_size: int          # side length of the sampled square inside a cell

    def cell_center(self, row: int, col: int) -> Tuple[int, int]:
        return (
            self.first_center_x + col * self.pitch_x,
            self.first_center_y + row * self.pitch_y,
        )

    def cell_box(self, row: int, col: int) -> Tuple[int, int, int, int]:
        cx, cy = self.cell_center(row, col)
        h = self.cell_size // 2
        return (cx - h, cy - h, cx + h, cy + h)


def _crop(img: Image.Image, box) -> Image.Image:
    return img.crop(box)


def _mean_rgb(patch: Image.Image) -> Tuple[int, int, int]:
    arr = np.asarray(patch).reshape(-1, 3)
    return tuple(int(v) for v in arr.mean(axis=0))


def _is_empty_cell(patch: Image.Image, empty_luma_max: int = 55, white_frac_min: float = 0.01) -> bool:
    """An empty cell is dark and has almost no bright (digit) pixels."""
    arr = np.asarray(patch).astype(np.int32)
    luma = arr @ np.array([0.299, 0.587, 0.114])
    white_frac = float((luma > 180).mean())
    mean_luma = float(luma.mean())
    return mean_luma < empty_luma_max and white_frac < white_frac_min


class BoardReader:
    def __init__(
        self,
        geometry: Geometry,
        reader: str = "ocr",
        color_palette: Optional[Dict[int, Tuple[int, int, int]]] = None,
        empty_luma_max: int = 55,
        tesseract_cmd: Optional[str] = None,
    ):
        self.geo = geometry
        self.reader = reader
        self.color_palette = color_palette or {}
        self.empty_luma_max = empty_luma_max
        self.tesseract_cmd = tesseract_cmd
        self._ocr_ready = False
        if reader == "ocr":
            self._init_ocr()

    def _init_ocr(self) -> None:
        try:
            import pytesseract  # noqa: F401
            self._pytesseract = pytesseract
            # Windows users can point straight at tesseract.exe instead of PATH,
            # e.g. C:\Program Files\Tesseract-OCR\tesseract.exe
            if self.tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
            self._ocr_ready = True
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "reader='ocr' requires pytesseract and the tesseract binary. "
                "Windows: install from https://github.com/UB-Mannheim/tesseract/wiki "
                "and set vision.tesseract_cmd in config.yaml. "
                "Linux: apt install tesseract-ocr. "
                f"Import failed: {exc}"
            )

    # -- public ------------------------------------------------------------- #
    def read(self, img: Image.Image) -> Board:
        cells: List[int] = []
        for r in range(4):
            for c in range(4):
                patch = _crop(img, self.geo.cell_box(r, c))
                if _is_empty_cell(patch, self.empty_luma_max):
                    cells.append(0)
                elif self.reader == "ocr":
                    cells.append(self._read_ocr(patch))
                else:
                    cells.append(self._read_color(patch))
        return tuple(cells)

    # -- readers ------------------------------------------------------------ #
    def _read_ocr(self, patch: Image.Image) -> int:
        arr = np.asarray(patch).astype(np.int32)
        luma = arr @ np.array([0.299, 0.587, 0.114])
        # Digits are white on a coloured tile: threshold to black text on white.
        mask = (luma > 170).astype(np.uint8) * 255
        binimg = Image.fromarray(255 - mask).resize(
            (patch.width * 3, patch.height * 3), Image.LANCZOS
        )
        txt = self._pytesseract.image_to_string(
            binimg,
            config="--psm 7 -c tessedit_char_whitelist=0123456789",
        ).strip()
        digits = "".join(ch for ch in txt if ch.isdigit())
        return int(digits) if digits else 0

    def _read_color(self, patch: Image.Image) -> int:
        if not self.color_palette:
            raise RuntimeError("reader='color' but no color_palette configured.")
        mean = np.array(_mean_rgb(patch))
        best_val, best_dist = 0, float("inf")
        for value, rgb in self.color_palette.items():
            d = float(np.linalg.norm(mean - np.array(rgb)))
            if d < best_dist:
                best_dist, best_val = d, value
        return best_val

    # -- calibration helper ------------------------------------------------- #
    def sample_colors(self, img: Image.Image) -> Dict[Tuple[int, int], Tuple[int, int, int]]:
        """Return mean RGB of every non-empty cell -- used when learning a palette."""
        out = {}
        for r in range(4):
            for c in range(4):
                patch = _crop(img, self.geo.cell_box(r, c))
                if not _is_empty_cell(patch, self.empty_luma_max):
                    out[(r, c)] = _mean_rgb(patch)
        return out
