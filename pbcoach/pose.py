"""Offline clean-up of per-frame 2D poses so skeletons stay steady.

A pose model looks at each frame on its own, so joints jitter by a few pixels from frame to
frame, and a joint hidden by a paddle, the net or a partner comes back as a confident guess in
the wrong place. Each player's joints are expressed relative to their box (which removes the
body's own movement), single-frame outliers are dropped, short dropouts are bridged, and what
is left is smoothed over time, harder for small far players whose joints are only a few
pixels wide.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter

MIN_CONF = 0.3        # below this a joint is treated as missing
FILLED_CONF = 0.35    # confidence given to bridged joints (still drawn)


def _bridge(v: np.ndarray, ok: np.ndarray, max_gap: int) -> tuple[np.ndarray, np.ndarray]:
    """Linearly fill runs of missing samples no longer than max_gap (edges hold the nearest value)."""
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        return v, ok
    filled = np.interp(np.arange(len(v)), idx, v[idx])
    can = ok.copy()
    i = 0
    while i < len(v):
        if ok[i]:
            i += 1
            continue
        j = i
        while j < len(v) and not ok[j]:
            j += 1
        if j - i <= max_gap:
            can[i:j] = True
        i = j
    return np.where(can, filled, v), can


def smooth_track(kps: list[np.ndarray | None], boxes: list, fps: float,
                 hold_s: float = 0.4, outlier: float = 0.12) -> list[np.ndarray | None]:
    """Steadied (17, 3) poses for one player's consecutive detections.

    kps: per-frame keypoints (x, y, conf) or None; boxes: per-frame (x1, y1, x2, y2).
    Positions are smoothed in body heights relative to the box's bottom center, so the
    smoothing never drags a joint behind a running player; only the pose itself is filtered.
    """
    n = len(kps)
    if n == 0:
        return kps
    b = np.asarray(boxes, dtype=np.float64)
    anchor = np.c_[(b[:, 0] + b[:, 2]) / 2, b[:, 3]]
    height = np.maximum(b[:, 3] - b[:, 1], 1.0)
    if n >= 3:   # boxes jitter too; a light smoothing keeps the frame of reference steady
        anchor = gaussian_filter1d(anchor, 1.0, axis=0, mode="nearest")
        height = gaussian_filter1d(height, 1.0, mode="nearest")
    K = np.full((n, 17, 3), np.nan)
    for i, kp in enumerate(kps):
        if kp is not None:
            K[i] = kp
    conf = np.nan_to_num(K[:, :, 2])
    rel = (K[:, :, :2] - anchor[:, None]) / height[:, None, None]
    ok = conf >= MIN_CONF
    # Smaller players (far side) get more smoothing: their joints are only a few pixels wide.
    sigma = float(np.clip(0.05 * 120.0 / np.median(height), 0.03, 0.12)) * fps
    win = max(3, int(round(0.2 * fps)) | 1)
    max_gap = int(round(hold_s * fps))
    out_rel, out_ok = np.full_like(rel, np.nan), np.zeros_like(ok)
    for j in range(17):
        r, m = rel[:, j].copy(), ok[:, j].copy()
        if m.sum() >= 3:
            # Drop single-frame jumps: compare with a running median over the valid samples.
            ref = np.stack([median_filter(_bridge(r[:, k], m, n)[0], size=win, mode="nearest") for k in (0, 1)], 1)
            m &= np.linalg.norm(r - ref, axis=1) <= outlier
        if not m.any():
            continue
        cols, can = [], None
        for k in (0, 1):
            v, can = _bridge(r[:, k], m, max_gap)
            w = m.astype(np.float64) * conf[:, j] + (can & ~m) * 0.2
            v = np.where(can, v, 0.0)
            if n >= 3:
                num = gaussian_filter1d(v * w, sigma, mode="nearest")
                den = gaussian_filter1d(w, sigma, mode="nearest")
                v = np.where(den > 1e-6, num / np.maximum(den, 1e-6), v)
            cols.append(v)
        out_rel[:, j] = np.stack(cols, 1)
        out_ok[:, j] = can
    xy = out_rel * height[:, None, None] + anchor[:, None]
    res = []
    for i in range(n):
        if not out_ok[i].any():
            res.append(None)
            continue
        c = np.where(ok[i] & out_ok[i], conf[i], np.where(out_ok[i], FILLED_CONF, 0.0))
        res.append(np.c_[np.nan_to_num(xy[i]), c])
    return res


def smooth_poses(seen: dict, fps: float) -> dict:
    """Smooth every player's poses in place. seen: {frame: {player_id: (court_xy, keypoints, box)}}."""
    by = {}
    for f in sorted(seen):
        for pid, entry in seen[f].items():
            by.setdefault(pid, []).append(f)
    for pid, frames in by.items():
        # Split at long absences so smoothing never bridges across them.
        runs, cur = [], [frames[0]]
        for a, b in zip(frames, frames[1:]):
            if b - a > 2:
                runs.append(cur)
                cur = []
            cur.append(b)
        runs.append(cur)
        for run in runs:
            entries = [seen[f][pid] for f in run]
            new = smooth_track([e[1] for e in entries], [e[2] for e in entries], fps)
            for f, e, kp in zip(run, entries, new):
                seen[f][pid] = (e[0], kp, e[2])
    return seen


def jitter(seen: dict) -> float:
    """Median frame-to-frame joint acceleration in body heights (lower is steadier)."""
    by = {}
    for f in sorted(seen):
        for pid, e in seen[f].items():
            by.setdefault(pid, []).append((f, e))
    vals = []
    for items in by.values():
        for (f0, a), (f1, b), (f2, c) in zip(items, items[1:], items[2:]):
            if f2 - f0 != 2 or a[1] is None or b[1] is None or c[1] is None:
                continue
            h = max(b[2][3] - b[2][1], 1.0)
            ok = (a[1][:, 2] >= MIN_CONF) & (b[1][:, 2] >= MIN_CONF) & (c[1][:, 2] >= MIN_CONF)
            acc = np.linalg.norm(a[1][:, :2] - 2 * b[1][:, :2] + c[1][:, :2], axis=1) / h
            vals += list(acc[ok])
    return float(np.median(vals)) if vals else float("nan")
