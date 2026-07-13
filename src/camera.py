"""Read the board optically with a camera instead of a screenshot.

Why: the target app is a "secure" build (FLAG_SECURE), so `adb screencap`
returns a black frame -- the screen cannot be captured in software. A camera
pointed at the phone bypasses this completely, because it reads light, not the
framebuffer. Input still goes through normal `adb input swipe`, which secure
apps do NOT block.

Pipeline: grab a camera frame -> perspective-correct the (possibly tilted)
phone screen to a flat square -> hand that top-down board image to the same
BoardReader used for screenshots.

Requires opencv-python. The four board corners are calibrated once (interactive
picker in calibrate.py, or hard-coded in config under `camera.corners`).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .vision import Geometry

Corner = Tuple[float, float]


def _require_cv2():
    try:
        import cv2  # noqa
        return cv2
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Camera mode needs opencv-python (`pip install opencv-python`). "
            f"Import failed: {exc}"
        )


def warp_board(frame_rgb: np.ndarray, corners: Sequence[Corner], size: int) -> np.ndarray:
    """Perspective-correct the quad `corners` (TL, TR, BR, BL) to a size x size square."""
    cv2 = _require_cv2()
    src = np.array(corners, dtype=np.float32)
    dst = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame_rgb, matrix, (size, size))


def canonical_geometry(size: int, cell_fill: float = 0.6) -> Geometry:
    """Grid layout for a warped board image (evenly spaced 4x4)."""
    pitch = size // 4
    first = pitch // 2
    return Geometry(
        first_center_x=first,
        first_center_y=first,
        pitch_x=pitch,
        pitch_y=pitch,
        cell_size=int(pitch * cell_fill),
    )


class CameraSource:
    """Frame source backed by a webcam / capture device."""

    def __init__(
        self,
        index: int,
        corners: Sequence[Corner],
        warp_size: int = 640,
        flush_frames: int = 3,
        capture_width: Optional[int] = None,
        capture_height: Optional[int] = None,
    ):
        cv2 = _require_cv2()
        self._cv2 = cv2
        self.corners = list(corners)
        self.warp_size = warp_size
        self.flush_frames = flush_frames
        self.cap = cv2.VideoCapture(index)
        if capture_width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
        if capture_height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {index}.")

    def _read_frame_rgb(self) -> np.ndarray:
        cv2 = self._cv2
        frame = None
        # Flush the driver buffer so we always act on the latest frame.
        for _ in range(max(1, self.flush_frames)):
            ok, frame = self.cap.read()
            if not ok or frame is None:
                raise RuntimeError("Camera frame grab failed.")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def raw_frame(self) -> Image.Image:
        return Image.fromarray(self._read_frame_rgb())

    def grab(self) -> Image.Image:
        frame = self._read_frame_rgb()
        warped = warp_board(frame, self.corners, self.warp_size)
        return Image.fromarray(warped)

    def canonical_geometry(self, cell_fill: float = 0.6) -> Geometry:
        return canonical_geometry(self.warp_size, cell_fill)

    def release(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass


def pick_corners_interactive(index: int) -> List[Corner]:  # pragma: no cover - needs GUI+camera
    """Open a live window; click the 4 board corners in order TL, TR, BR, BL."""
    cv2 = _require_cv2()
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {index}.")

    pts: List[Corner] = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((float(x), float(y)))

    win = "Click board corners: TL, TR, BR, BL  (r=reset, Enter=done, Esc=cancel)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    labels = ["TL", "TR", "BR", "BL"]
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            for i, (px, py) in enumerate(pts):
                cv2.circle(frame, (int(px), int(py)), 6, (0, 0, 255), -1)
                cv2.putText(frame, labels[i], (int(px) + 8, int(py)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if len(pts) >= 1:
                cv2.polylines(frame, [np.array(pts, np.int32)], len(pts) == 4,
                              (0, 255, 0), 2)
            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                pts.clear()
            elif key == 13 and len(pts) == 4:  # Enter
                break
            elif key == 27:  # Esc
                pts.clear()
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if len(pts) != 4:
        raise RuntimeError("Corner selection cancelled.")
    return pts
