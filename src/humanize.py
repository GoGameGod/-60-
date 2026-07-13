"""Make the bot's input look like a real thumb.

Two sources of "humanness":

* Timing  -- floating, minimal pauses between moves with the occasional longer
             "thinking" pause, so the cadence is never mechanically constant.
* Gesture -- each swipe starts from a jittered point near the board centre, has
             a randomised length, angle wobble and duration.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Tuple


@dataclass
class TimingProfile:
    base_min: float = 0.09          # minimal pause between moves (s)
    base_max: float = 0.22
    long_pause_chance: float = 0.06  # occasional human "hesitation"
    long_pause_min: float = 0.5
    long_pause_max: float = 1.4

    def next_delay(self, rng: random.Random) -> float:
        if rng.random() < self.long_pause_chance:
            return rng.uniform(self.long_pause_min, self.long_pause_max)
        return rng.uniform(self.base_min, self.base_max)

    def sleep(self, rng: random.Random) -> None:
        time.sleep(self.next_delay(rng))


@dataclass
class SwipeProfile:
    center_x: int
    center_y: int
    distance: int = 320             # nominal swipe length in px
    distance_jitter: float = 0.18   # +/- fraction
    start_jitter: int = 26          # px wobble around the centre start point
    angle_jitter_deg: float = 6.0   # wobble off the pure axis
    duration_min: int = 45          # ms
    duration_max: int = 95

    def gesture(self, direction: str, rng: random.Random) -> Tuple[int, int, int, int, int]:
        """Return (x1, y1, x2, y2, duration_ms) for a swipe in `direction`."""
        import math

        axis = {
            "up": (0, -1),
            "down": (0, 1),
            "left": (-1, 0),
            "right": (1, 0),
        }[direction]

        x1 = self.center_x + rng.randint(-self.start_jitter, self.start_jitter)
        y1 = self.center_y + rng.randint(-self.start_jitter, self.start_jitter)

        dist = self.distance * (1 + rng.uniform(-self.distance_jitter, self.distance_jitter))
        base_angle = math.atan2(axis[1], axis[0])
        angle = base_angle + math.radians(rng.uniform(-self.angle_jitter_deg, self.angle_jitter_deg))

        x2 = int(x1 + dist * math.cos(angle))
        y2 = int(y1 + dist * math.sin(angle))
        duration = rng.randint(self.duration_min, self.duration_max)
        return x1, y1, x2, y2, duration
