"""Drawing helpers: court overlay on the camera image and a top-down minimap."""
from __future__ import annotations

import cv2
import numpy as np

from . import court
from .camera import project_w


def draw_court(img: np.ndarray, H: np.ndarray, color=(0, 255, 255), thickness: int = 2) -> np.ndarray:
    for (x1, y1), (x2, y2) in court.SEGMENTS + [((0, court.NET_Y), (court.WIDTH, court.NET_Y))]:
        seg = np.linspace((x1, y1), (x2, y2), 40)
        pts, w = project_w(H, seg)
        pts = pts[w > 0]
        if len(pts) > 1:
            cv2.polylines(img, [pts.astype(np.int32)], False, color, thickness, cv2.LINE_AA)
    return img


def minimap(positions: dict, ppm: int = 30, margin: float = 2.0, trails: dict | None = None) -> np.ndarray:
    """Top-down court with players. positions: {player_id: (x, y, team)}."""
    W = int((court.WIDTH + 2 * margin) * ppm)
    Hh = int((court.LENGTH + 2 * margin) * ppm)
    img = np.full((Hh, W, 3), (60, 110, 60), np.uint8)

    def p(x, y):  # far baseline at top, near baseline at bottom
        return int((x + margin) * ppm), int((court.LENGTH - y + margin) * ppm)

    cv2.rectangle(img, p(0, 0), p(court.WIDTH, court.LENGTH), (150, 95, 40), -1)
    cv2.rectangle(img, p(0, court.NVZ_NEAR_Y), p(court.WIDTH, court.NVZ_FAR_Y), (170, 120, 70), -1)
    for (x1, y1), (x2, y2) in court.SEGMENTS:
        cv2.line(img, p(x1, y1), p(x2, y2), (240, 240, 240), 2)
    cv2.line(img, p(-0.3, court.NET_Y), p(court.WIDTH + 0.3, court.NET_Y), (30, 30, 30), 3)
    colors = [(0, 0, 255), (0, 165, 255), (255, 0, 0), (255, 0, 255), (0, 255, 0), (255, 255, 0)]
    for pid, trail in (trails or {}).items():
        if len(trail) > 1:
            cv2.polylines(img, [np.array([p(x, y) for x, y in trail], np.int32)], False,
                          colors[pid % len(colors)], 1, cv2.LINE_AA)
    for pid, (x, y, team) in positions.items():
        c = colors[pid % len(colors)]
        cv2.circle(img, p(x, y), 7, c, -1, cv2.LINE_AA)
        cv2.putText(img, str(pid), (p(x, y)[0] + 8, p(x, y)[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
    return img
