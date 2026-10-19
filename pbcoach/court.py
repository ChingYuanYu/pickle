"""Pickleball court model in court meters.

Coordinate frame (used everywhere downstream, independent of camera angle):
  origin = left corner of the near baseline, x across the court (0..WIDTH),
  y along the court (0..LENGTH), z up. The net sits at y = NET_Y.
"""
from __future__ import annotations

import numpy as np

FT = 0.3048
WIDTH = 20 * FT            # 6.096 m
LENGTH = 44 * FT           # 13.4112 m
NET_Y = LENGTH / 2         # 6.7056 m
NVZ_DEPTH = 7 * FT         # 2.1336 m (kitchen / non-volley zone)
NVZ_NEAR_Y = NET_Y - NVZ_DEPTH
NVZ_FAR_Y = NET_Y + NVZ_DEPTH
CENTER_X = WIDTH / 2
LINE_WIDTH = 2 * 0.0254    # 2 inch painted lines
NET_HEIGHT_POST = 36 * 0.0254
NET_HEIGHT_CENTER = 34 * 0.0254

# Families of infinite lines used to build calibration hypotheses.
ACROSS_Y = (0.0, NVZ_NEAR_Y, NVZ_FAR_Y, LENGTH)   # lines of constant y
ALONG_X = (0.0, CENTER_X, WIDTH)                  # lines of constant x

# Painted segments: ((x1, y1), (x2, y2)).
SEGMENTS = [
    ((0.0, 0.0), (WIDTH, 0.0)),                 # near baseline
    ((0.0, LENGTH), (WIDTH, LENGTH)),           # far baseline
    ((0.0, 0.0), (0.0, LENGTH)),                # left sideline
    ((WIDTH, 0.0), (WIDTH, LENGTH)),            # right sideline
    ((0.0, NVZ_NEAR_Y), (WIDTH, NVZ_NEAR_Y)),   # near kitchen line
    ((0.0, NVZ_FAR_Y), (WIDTH, NVZ_FAR_Y)),     # far kitchen line
    ((CENTER_X, 0.0), (CENTER_X, NVZ_NEAR_Y)),  # near centerline
    ((CENTER_X, NVZ_FAR_Y), (CENTER_X, LENGTH)),  # far centerline
]

# Named line intersections that physically exist on the court.
KEYPOINTS = {
    "near_left": (0.0, 0.0),
    "near_center": (CENTER_X, 0.0),
    "near_right": (WIDTH, 0.0),
    "nvz_near_left": (0.0, NVZ_NEAR_Y),
    "nvz_near_center": (CENTER_X, NVZ_NEAR_Y),
    "nvz_near_right": (WIDTH, NVZ_NEAR_Y),
    "nvz_far_left": (0.0, NVZ_FAR_Y),
    "nvz_far_center": (CENTER_X, NVZ_FAR_Y),
    "nvz_far_right": (WIDTH, NVZ_FAR_Y),
    "far_left": (0.0, LENGTH),
    "far_center": (CENTER_X, LENGTH),
    "far_right": (WIDTH, LENGTH),
}

CORNERS = np.array([(0.0, 0.0), (WIDTH, 0.0), (WIDTH, LENGTH), (0.0, LENGTH)])


def sample_segments(step: float = 0.2) -> np.ndarray:
    """Points along every painted segment, spaced about `step` meters apart."""
    pts = []
    for (x1, y1), (x2, y2) in SEGMENTS:
        n = max(2, int(np.hypot(x2 - x1, y2 - y1) / step) + 1)
        t = np.linspace(0.0, 1.0, n)
        pts.append(np.stack([x1 + (x2 - x1) * t, y1 + (y2 - y1) * t], axis=1))
    return np.concatenate(pts)


def zone(x: float, y: float, margin: float = 0.0) -> str:
    """Coarse court zone of a ground position, for coaching metrics."""
    if not (-margin <= x <= WIDTH + margin and -margin <= y <= LENGTH + margin):
        return "out"
    near = y < NET_Y
    dist_to_nvz = abs(y - (NVZ_NEAR_Y if near else NVZ_FAR_Y))
    in_kitchen = NVZ_NEAR_Y <= y <= NVZ_FAR_Y
    side = "near" if near else "far"
    if in_kitchen:
        return f"{side}_kitchen"
    if dist_to_nvz <= 1.0:
        return f"{side}_kitchen_line"
    if y < 1.5 or y > LENGTH - 1.5:
        return f"{side}_baseline"
    return f"{side}_transition"
