"""Top-down figures for reports: per-player heatmaps and full-clip movement trails."""
from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np

from . import court
from .viz import minimap

PPM, MARGIN = 30, 2.0


def _px(x, y):
    return (x + MARGIN) * PPM, (court.LENGTH - y + MARGIN) * PPM


def _by_player(rows):
    by = defaultdict(list)
    for r in rows:
        by[int(r["player_id"])].append((float(r["x"]), float(r["y"])))
    return {k: np.array(v) for k, v in by.items()}


def heatmaps(rows: list[dict], min_frames: int = 5) -> np.ndarray | None:
    """One top-down heatmap per player, side by side (BGR)."""
    panels = []
    for pid, xy in sorted(_by_player(rows).items()):
        if len(xy) < min_frames:
            continue
        base = minimap({}, PPM, MARGIN)
        acc = np.zeros(base.shape[:2], np.float32)
        u, v = _px(xy[:, 0], xy[:, 1])
        ok = (u >= 0) & (u < acc.shape[1]) & (v >= 0) & (v < acc.shape[0])
        np.add.at(acc, (v[ok].astype(int), u[ok].astype(int)), 1.0)
        acc = cv2.GaussianBlur(acc, (0, 0), 0.6 * PPM)
        if acc.max() > 0:
            acc /= acc.max()
        color = cv2.applyColorMap((acc * 255).astype(np.uint8), cv2.COLORMAP_JET)
        alpha = np.clip(acc * 1.5, 0, 0.8)[..., None]
        img = (base * (1 - alpha) + color * alpha).astype(np.uint8)
        cv2.putText(img, f"Player {pid}", (8, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(img)
    return np.hstack(panels) if panels else None


def trails(rows: list[dict]) -> np.ndarray:
    """Whole-clip paths for every player on one court, with the last position marked."""
    tr = _by_player(rows)
    last = {pid: (*xy[-1], "") for pid, xy in tr.items()}
    return minimap(last, PPM, MARGIN, trails={pid: [tuple(p) for p in xy] for pid, xy in tr.items()})


def kitchen_distance_series(rows: list[dict]) -> list[dict]:
    """Long-format rows (t, player, meters behind own kitchen line) for a line chart."""
    out = []
    for r in rows:
        y = float(r["y"])
        behind = court.NVZ_NEAR_Y - y if r["team"] == "near" else y - court.NVZ_FAR_Y
        out.append({"t": float(r["t"]), "player": f"Player {r['player_id']}", "behind_kitchen_m": round(behind, 2)})
    return out
