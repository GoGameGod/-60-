"""Thin wrapper around the `adb` command line for native device control.

Everything the bot does on the phone goes through here: capturing the screen
and injecting swipes/taps via Android's standard input pipeline. Because we use
the real touch input subsystem, the target app sees ordinary gestures -- there
is no instrumentation, root, or app patching involved.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import time
from typing import List, Optional

from PIL import Image


class ADBError(RuntimeError):
    pass


class ADB:
    def __init__(self, serial: Optional[str] = None, adb_path: str = "adb"):
        self.adb_path = adb_path
        self.serial = serial
        if shutil.which(adb_path) is None:
            raise ADBError(
                f"'{adb_path}' not found on PATH. Install Android platform-tools "
                "and enable USB debugging on the phone."
            )

    def _base_cmd(self) -> List[str]:
        cmd = [self.adb_path]
        if self.serial:
            cmd += ["-s", self.serial]
        return cmd

    def _run(self, args: List[str], capture: bool = True, timeout: float = 20.0) -> bytes:
        proc = subprocess.run(
            self._base_cmd() + args,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        if proc.returncode != 0:
            raise ADBError(
                f"adb {' '.join(args)} failed ({proc.returncode}): "
                f"{proc.stderr.decode(errors='replace').strip()}"
            )
        return proc.stdout or b""

    # -- lifecycle ---------------------------------------------------------- #
    def devices(self) -> List[str]:
        out = self._run(["devices"]).decode(errors="replace")
        serials = []
        for line in out.splitlines()[1:]:
            line = line.strip()
            if line and line.endswith("device"):
                serials.append(line.split()[0])
        return serials

    def wait_for_device(self, timeout: float = 30.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.devices():
                return
            time.sleep(0.5)
        raise ADBError("no authorized device found. Check USB and 'Allow debugging'.")

    # -- screen ------------------------------------------------------------- #
    def screencap(self) -> Image.Image:
        """Grab the framebuffer as a PIL Image (PNG over exec-out, no temp file)."""
        png = self._run(["exec-out", "screencap", "-p"])
        if not png.startswith(b"\x89PNG"):
            # Some OEM shells mangle newlines; fall back to the pull method.
            png = png.replace(b"\r\n", b"\n")
        return Image.open(io.BytesIO(png)).convert("RGB")

    # -- input -------------------------------------------------------------- #
    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 60) -> None:
        self._run(
            ["shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(duration_ms)],
            capture=False,
        )

    def tap(self, x: int, y: int) -> None:
        self._run(["shell", "input", "tap", str(x), str(y)], capture=False)
