"""Angle-independent movement metrics computed from court-meter tracks."""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from . import court


def player_metrics(rows: list[dict], fps: float) -> dict:
    """rows: dicts with frame, player_id, team, x, y (one per tracked player per frame)."""
    by_player = defaultdict(list)
    for r in rows:
        by_player[r["player_id"]].append(r)
    out = {}
    for pid, rs in by_player.items():
        rs.sort(key=lambda r: r["frame"])
        xy = np.array([(r["x"], r["y"]) for r in rs])
        frames = np.array([r["frame"] for r in rs])
        steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        contiguous = np.diff(frames) <= 2
        dist = float(steps[contiguous].sum()) if len(steps) else 0.0
        zones = Counter(court.zone(x, y, margin=1.5) for x, y in xy)
        n = max(1, len(xy))
        nvz = np.where(xy[:, 1] < court.NET_Y, court.NVZ_NEAR_Y, court.NVZ_FAR_Y)
        out[str(pid)] = {
            "team": Counter(r["team"] for r in rs).most_common(1)[0][0],
            "frames_tracked": int(len(xy)),
            "seconds_tracked": round(len(xy) / fps, 1),
            "distance_m": round(dist, 1),
            "mean_distance_to_kitchen_line_m": round(float(np.mean(np.abs(xy[:, 1] - nvz))), 2),
            "zone_share": {k.split("_", 1)[1] if "_" in k else k: round(v / n, 3)
                           for k, v in sorted(zones.items())},
        }
    return out


def team_metrics(rows: list[dict]) -> dict:
    """Partner spacing and how often both partners are up at the kitchen line together."""
    frames = defaultdict(lambda: defaultdict(list))
    for r in rows:
        frames[r["team"]][r["frame"]].append((r["x"], r["y"]))
    out = {}
    for team, fr in frames.items():
        pairs = [np.array(v) for v in fr.values() if len(v) == 2]
        if not pairs:
            continue
        spacing = [float(np.linalg.norm(p[0] - p[1])) for p in pairs]
        nvz = court.NVZ_NEAR_Y if team == "near" else court.NVZ_FAR_Y
        both_up = [bool((np.abs(p[:, 1] - nvz) <= 1.0).all()) for p in pairs]
        out[team] = {
            "frames_with_both_partners": len(pairs),
            "mean_partner_spacing_m": round(float(np.mean(spacing)), 2),
            "share_both_at_kitchen_line": round(float(np.mean(both_up)), 3),
        }
    return out
