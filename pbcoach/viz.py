"""Drawing helpers: court overlay on the camera image and a top-down minimap."""
from __future__ import annotations

import cv2
import numpy as np

from . import court
from .camera import project_w


# BGR; players 1-4 get orange-red, yellow, cyan and purple, like a broadcast replay.
PALETTE = [(40, 80, 240), (30, 200, 245), (230, 200, 40), (220, 90, 180), (90, 210, 90), (240, 140, 60)]

# COCO-17 keypoint bones.
BONES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14),
         (14, 16), (0, 5), (0, 6)]


def player_color(pid: int):
    return PALETTE[(int(pid) - 1) % len(PALETTE)]


def draw_ring(img: np.ndarray, cal, xy, color, radius_m: float = 0.45, thickness: int = 2) -> np.ndarray:
    """Circle on the ground around a player's feet, drawn in perspective."""
    a = np.linspace(0, 2 * np.pi, 40)
    pts, w = project_w(cal.H, np.c_[xy[0] + radius_m * np.cos(a), xy[1] + radius_m * np.sin(a)])
    if (w > 0).all() and np.isfinite(pts).all():
        cv2.polylines(img, [pts.astype(np.int32)], True, color, thickness, cv2.LINE_AA)
    return img


def draw_skeleton(img: np.ndarray, kp: np.ndarray, color, min_conf: float = 0.3) -> np.ndarray:
    """COCO-17 keypoints (x, y, conf) as a stick figure."""
    scale = max(1, int(round(np.ptp(kp[kp[:, 2] >= min_conf, 1]) / 60))) if (kp[:, 2] >= min_conf).sum() > 1 else 1
    for a, b in BONES:
        if kp[a, 2] >= min_conf and kp[b, 2] >= min_conf:
            cv2.line(img, tuple(kp[a, :2].astype(int)), tuple(kp[b, :2].astype(int)), color, 1 + scale, cv2.LINE_AA)
    for x, y, c in kp[5:]:
        if c >= min_conf:
            cv2.circle(img, (int(x), int(y)), 1 + scale, (255, 255, 255), -1, cv2.LINE_AA)
    if kp[0, 2] >= min_conf:
        cv2.circle(img, tuple(kp[0, :2].astype(int)), 3 + 2 * scale, color, -1, cv2.LINE_AA)
    return img


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
    for pid, trail in (trails or {}).items():
        if len(trail) > 1:
            cv2.polylines(img, [np.array([p(x, y) for x, y in trail], np.int32)], False,
                          player_color(pid), 2, cv2.LINE_AA)
    for pid, (x, y, team) in positions.items():
        c = player_color(pid)
        cv2.circle(img, p(x, y), 7, c, -1, cv2.LINE_AA)
        cv2.putText(img, str(pid), (p(x, y)[0] + 8, p(x, y)[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
    return img
