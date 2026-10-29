"""Angle-independent movement metrics computed from court-meter tracks."""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from .court import PICKLEBALL, TENNIS_SERVICE_FROM_NET, Court

TENNIS_UP_M = 1.0      # tennis "up" = within 1 m behind the service line, or closer to the net
TENNIS_BACK_M = 2.0    # tennis "back" = less than 2 m in front of the baseline (or behind it)


def tennis_up(court: Court, team: str, y) -> np.ndarray:
    return court.from_baseline(team, y) >= court.net_y - TENNIS_SERVICE_FROM_NET - TENNIS_UP_M


def tennis_back(court: Court, team: str, y) -> np.ndarray:
    return court.from_baseline(team, y) < TENNIS_BACK_M


def player_metrics(rows: list[dict], fps: float, court: Court = PICKLEBALL) -> dict:
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
        team = Counter(r["team"] for r in rs).most_common(1)[0][0]
        m = {
            "team": team,
            "frames_tracked": int(len(xy)),
            "seconds_tracked": round(len(xy) / fps, 1),
            "distance_m": round(dist, 1),
        }
        if court.sport == "pickleball":
            nvz = np.where(xy[:, 1] < court.net_y, court.front_line("near"), court.front_line("far"))
            m["mean_distance_to_kitchen_line_m"] = round(float(np.mean(np.abs(xy[:, 1] - nvz))), 2)
        else:
            m["mean_m_behind_baseline"] = round(float(np.mean(court.behind(team, xy[:, 1]))), 2)
            m["mean_m_from_center"] = round(float(np.mean(np.abs(xy[:, 0] - court.center_x))), 2)
        m["zone_share"] = {k.split("_", 1)[1] if "_" in k else k: round(v / n, 3) for k, v in sorted(zones.items())}
        out[str(pid)] = m
    return out


def team_metrics(rows: list[dict], court: Court = PICKLEBALL) -> dict:
    """Doubles partner spacing, and how often partners are up at the kitchen line together
    (pickleball) or in which formation they play (tennis: both back, one up one back, both up)."""
    frames = defaultdict(lambda: defaultdict(list))
    for r in rows:
        frames[r["team"]][r["frame"]].append((r["x"], r["y"]))
    out = {}
    for team, fr in frames.items():
        pairs = [np.array(v) for v in fr.values() if len(v) == 2]
        if not pairs:
            continue
        spacing = [float(np.linalg.norm(p[0] - p[1])) for p in pairs]
        m = {"frames_with_both_partners": len(pairs), "mean_partner_spacing_m": round(float(np.mean(spacing)), 2)}
        if court.sport == "pickleball":
            nvz = court.front_line(team)
            both_up = [bool((np.abs(p[:, 1] - nvz) <= 1.0).all()) for p in pairs]
            m["share_both_at_kitchen_line"] = round(float(np.mean(both_up)), 3)
        else:
            up = np.array([tennis_up(court, team, p[:, 1]).sum() for p in pairs])
            back = np.array([tennis_back(court, team, p[:, 1]).sum() for p in pairs])
            m["share_both_back"] = round(float(np.mean(back == 2)), 3)
            m["share_one_up_one_back"] = round(float(np.mean((up == 1) & (back == 1))), 3)
            m["share_both_up"] = round(float(np.mean(up == 2)), 3)
        out[team] = m
    return out
