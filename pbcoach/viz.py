"""Drawing helpers: court overlay on the camera image and a top-down minimap."""
from __future__ import annotations

import cv2
import numpy as np

from .court import PICKLEBALL, Court
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


def draw_court(img: np.ndarray, H: np.ndarray, color=(0, 255, 255), thickness: int = 2,
               court: Court = PICKLEBALL) -> np.ndarray:
    for (x1, y1), (x2, y2) in list(court.segments) + [((0, court.net_y), (court.width, court.net_y))]:
        seg = np.linspace((x1, y1), (x2, y2), 40)
        pts, w = project_w(H, seg)
        pts = pts[w > 0]
        if len(pts) > 1:
            cv2.polylines(img, [pts.astype(np.int32)], False, color, thickness, cv2.LINE_AA)
    return img


def map_margins(court: Court = PICKLEBALL, margin: float | None = None) -> tuple[float, float]:
    """Ground shown beyond the sidelines and baselines on top-down maps: where players can be."""
    return (court.runoff_side, court.runoff_back) if margin is None else (margin, margin)


def map_px(x, y, court: Court = PICKLEBALL, ppm: int = 30, margin: float | None = None):
    """Court meters -> top-down map pixels (far baseline at top, near baseline at bottom)."""
    mx, my = map_margins(court, margin)
    return (np.asarray(x) + mx) * ppm, (court.length - np.asarray(y) + my) * ppm


def minimap(positions: dict, ppm: int = 30, margin: float | None = None, trails: dict | None = None,
            court: Court = PICKLEBALL) -> np.ndarray:
    """Top-down court with players. positions: {player_id: (x, y, team)}."""
    mx, my = map_margins(court, margin)
    W = int((court.width + 2 * mx) * ppm)
    Hh = int((court.length + 2 * my) * ppm)
    img = np.full((Hh, W, 3), (60, 110, 60), np.uint8)

    def p(x, y):
        u, v = map_px(x, y, court, ppm, margin)
        return int(u), int(v)

    cv2.rectangle(img, p(0, 0), p(court.width, court.length), (150, 95, 40), -1)
    for x0, y0, x1, y1 in court.shade:
        cv2.rectangle(img, p(x0, y0), p(x1, y1), (170, 120, 70), -1)
    for (x1, y1), (x2, y2) in court.segments:
        cv2.line(img, p(x1, y1), p(x2, y2), (240, 240, 240), 2)
    cv2.line(img, p(-court.net_overhang, court.net_y), p(court.width + court.net_overhang, court.net_y), (30, 30, 30), 3)
    for pid, trail in (trails or {}).items():
        if len(trail) > 1:
            cv2.polylines(img, [np.array([p(x, y) for x, y in trail], np.int32)], False,
                          player_color(pid), 2, cv2.LINE_AA)
    for pid, (x, y, team) in positions.items():
        c = player_color(pid)
        cv2.circle(img, p(x, y), 7, c, -1, cv2.LINE_AA)
        cv2.putText(img, str(pid), (p(x, y)[0] + 8, p(x, y)[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
    return img
