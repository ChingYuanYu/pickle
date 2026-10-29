"""Coaching feedback from court-meter tracks.

Rule-based insights work offline and are the source of truth. When ANTHROPIC_API_KEY is
set, Claude can turn them into a coach's voice and answer follow-up questions; it only
sees the numbers computed here, never the video.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np

from . import court as courts
from .court import PICKLEBALL, TENNIS_SERVICE_FROM_NET
from .metrics import tennis_back, tennis_up

KITCHEN_BAND_M = 1.0      # "at the kitchen line" = within 1 m of it
BACK_M = 3.0              # "back" = more than 3 m behind the kitchen line
DEEP_M = 3.0              # tennis: "very deep" = more than 3 m behind the baseline
WIDE_M = 2.5              # tennis singles: "pulled wide" = more than 2.5 m from the center mark
RECOVERED_M = 1.5         # ... and "recovered" = back within 1.5 m of it
DEFAULT_MODEL = "claude-opus-5-5"


@dataclass
class Insight:
    who: str          # "Player 3", "Near team", "Camera"
    kind: str         # "strength" | "improve" | "info"
    title: str
    detail: str


def player_ids(rows) -> set[int]:
    """Track ids long enough to be real players (short tracks are usually ghosts or ID switches)."""
    counts = defaultdict(int)
    for r in rows:
        counts[int(r["player_id"])] += 1
    longest = max(counts.values(), default=0)
    return {pid for pid, n in counts.items() if n >= max(5, 0.2 * longest)}


def _tracks(rows):
    """{player_id: (team, frames, xy)} for tracks long enough to be real players."""
    keep = player_ids(rows)
    by = defaultdict(list)
    for r in rows:
        if int(r["player_id"]) in keep:
            by[int(r["player_id"])].append(r)
    out = {}
    for pid, rs in by.items():
        rs.sort(key=lambda r: int(r["frame"]))
        teams = [r["team"] for r in rs]
        out[pid] = (max(set(teams), key=teams.count), np.array([int(r["frame"]) for r in rs]),
                    np.array([(float(r["x"]), float(r["y"])) for r in rs]))
    return out


def _behind_kitchen(team, xy):
    """Meters behind the team's own kitchen line (negative = inside the kitchen)."""
    return PICKLEBALL.behind(team, xy[:, 1])


def analyze(rows: list[dict], summary: dict, fps: float) -> list[Insight]:
    """Findings for the sport in summary["sport"] (pickleball when absent)."""
    court = courts.get(summary.get("sport"))
    tracks = _tracks(rows)
    out = _camera_insights(summary, tracks)
    if not tracks:
        return out
    if court.sport == "tennis":
        return out + _tennis(tracks, fps, court)
    return out + _pickleball(tracks, fps)


def _camera_insights(summary: dict, tracks: dict) -> list[Insight]:
    out: list[Insight] = []
    cal = summary.get("calibration", {})
    if cal.get("score", 1.0) < 0.6:
        out.append(Insight("Camera", "info", "Court fit is uncertain",
                           f"Only {cal.get('score', 0):.0%} of the painted lines were confirmed, so positions may "
                           "be off. A higher camera that sees all four corners, or an empty-court photo, helps."))
    if not tracks:
        out.append(Insight("Camera", "info", "No players tracked",
                           "No one was tracked long enough to analyze. Check that players are visible on the court."))
    return out


def _pickleball(tracks: dict, fps: float) -> list[Insight]:
    out: list[Insight] = []
    for pid, (team, frames, xy) in sorted(tracks.items()):
        who = f"Player {pid} ({team} side)"
        behind = _behind_kitchen(team, xy)
        at_line = np.abs(behind) <= KITCHEN_BAND_M
        t = (frames - frames[0]) / fps
        # Moving up through the transition zone is fine; standing in it is not.
        k = max(1, int(round(0.5 * fps)))
        forward = np.gradient(-np.convolve(behind, np.ones(k) / k, mode="same"), t) if len(t) > 2 else np.zeros(len(t))
        transition = (behind > KITCHEN_BAND_M) & (behind < PICKLEBALL.front_y - 1.5) & (forward < 0.5)
        if at_line.any():
            t_up = float(t[np.argmax(at_line)])
            if t_up > 4.0:
                out.append(Insight(who, "improve", "Slow to reach the kitchen line",
                                   f"Took {t_up:.1f} s to get within 1 m of the kitchen line. After the return, "
                                   "move up behind your shot so you are set before the opponents' next ball."))
            else:
                out.append(Insight(who, "strength", "Gets to the kitchen line",
                                   f"Reached the kitchen line {t_up:.1f} s into the clip and spent "
                                   f"{at_line.mean():.0%} of the time there."))
        else:
            out.append(Insight(who, "improve", "Never reached the kitchen line",
                               f"Stayed on average {behind.mean():.1f} m behind the kitchen line. Most points in "
                               "doubles are won at the line, so look for a chance to move up."))
        if transition.mean() > 0.2:
            out.append(Insight(who, "improve", "Lingering in the transition zone",
                               f"Spent {transition.mean():.0%} of the time standing between the baseline and "
                               "the kitchen line, where balls land at your feet. Keep moving forward, and "
                               "split-step only to reset before continuing to the line."))

    by_team = defaultdict(list)
    for pid, (team, frames, xy) in tracks.items():
        by_team[team].append((pid, frames, xy))
    for team, players in sorted(by_team.items()):
        if len(players) != 2:
            continue
        (pa, fa, xa), (pb, fb, xb) = players
        common, ia, ib = np.intersect1d(fa, fb, return_indices=True)
        if len(common) < 5:
            continue
        a, b = xa[ia], xb[ib]
        spacing = np.linalg.norm(a - b, axis=1)
        ba, bb = _behind_kitchen(team, a), _behind_kitchen(team, b)
        both_up = (np.abs(ba) <= KITCHEN_BAND_M) & (np.abs(bb) <= KITCHEN_BAND_M)
        split = ((np.abs(ba) <= KITCHEN_BAND_M) & (bb > BACK_M)) | ((np.abs(bb) <= KITCHEN_BAND_M) & (ba > BACK_M))
        who = f"{team.capitalize()} team (players {pa} and {pb})"
        if split.mean() > 0.2:
            out.append(Insight(who, "improve", "Partners split up and back",
                               f"For {split.mean():.0%} of the time one partner was at the kitchen line while the "
                               "other stayed more than 3 m back. That leaves a gap opponents can hit through. "
                               "Move up together, like being connected by a rope."))
        if both_up.mean() >= 0.4:
            out.append(Insight(who, "strength", "Both partners at the line",
                               f"Both partners were at the kitchen line together {both_up.mean():.0%} of the time."))
        m = float(np.median(spacing))
        if m > 4.0:
            out.append(Insight(who, "improve", "Gap down the middle",
                               f"Partners were usually {m:.1f} m apart. On a 6.1 m wide court that leaves the middle "
                               "open; shift toward the ball together and keep about 3 m apart."))
        elif m < 1.8:
            out.append(Insight(who, "improve", "Bunched together",
                               f"Partners were usually only {m:.1f} m apart, which opens the sidelines. Each partner "
                               "should cover their half and shade toward the ball."))
        else:
            out.append(Insight(who, "strength", "Good partner spacing",
                               f"Typical spacing was {m:.1f} m, which covers the middle and both sidelines."))
    return out


def _excursions(off: np.ndarray, t: np.ndarray):
    """Recovery times (s) after each time the player was pulled wide; None if they never got back."""
    out, i = [], 0
    while i < len(off):
        if off[i] > WIDE_M:
            j = i
            while j < len(off) and off[j] > RECOVERED_M:
                j += 1
            out.append(float(t[j] - t[i]) if j < len(off) else None)
            i = j
        i += 1
    return out


def _tennis(tracks: dict, fps: float, court) -> list[Insight]:
    out: list[Insight] = []
    by_team = defaultdict(list)
    for pid, (team, frames, xy) in tracks.items():
        by_team[team].append((pid, frames, xy))
    singles = all(len(v) == 1 for v in by_team.values())
    service = court.net_y - TENNIS_SERVICE_FROM_NET       # service line, meters from the baseline
    for pid, (team, frames, xy) in sorted(tracks.items()):
        who = f"Player {pid} ({team} side)"
        t = (frames - frames[0]) / fps
        depth = court.behind(team, xy[:, 1])               # meters behind own baseline
        fwd = court.from_baseline(team, xy[:, 1])
        up = tennis_up(court, team, xy[:, 1])
        back = ~up
        if back.mean() > 0.5:
            d = float(np.median(depth[back]))
            if d > DEEP_M:
                out.append(Insight(who, "improve", "Playing far behind the baseline",
                                   f"Usually stood {d:.1f} m behind the baseline. From that deep the opponent gets "
                                   "sharper angles and short balls are hard to reach. Try holding about 1 m behind "
                                   "the baseline and stepping in when the ball lands short."))
            elif d < -1.0:
                out.append(Insight(who, "info", "Stays inside the baseline",
                                   f"Usually stood {-d:.1f} m inside the baseline. That takes time away from the "
                                   "opponent, but deep balls will jam you; take a step back against heavy hitters."))
            else:
                out.append(Insight(who, "strength", "Good baseline position",
                                   f"Usually stood {d:+.1f} m from the baseline, close enough to take the ball early "
                                   "and deep enough to handle depth."))
        k = max(1, int(round(0.5 * fps)))
        speed = np.gradient(np.convolve(fwd, np.ones(k) / k, mode="same"), t) if len(t) > 2 else np.zeros(len(t))
        no_mans = (fwd > 1.5) & (fwd < service - 1.0) & (speed < 0.5)
        if no_mans.mean() > 0.2:
            out.append(Insight(who, "improve", "Caught in no-man's land",
                               f"Spent {no_mans.mean():.0%} of the time between the baseline and the service line "
                               "without moving forward, where balls land at your feet. After a short ball, either "
                               "close in to the net or recover behind the baseline."))
        # Net approaches: getting inside the service line and staying for at least half a second.
        runs = np.diff(np.r_[0, up.astype(int), 0])
        starts, ends = np.nonzero(runs == 1)[0], np.nonzero(runs == -1)[0]
        approaches = int(sum((t[min(e, len(t)) - 1] - t[s_]) >= 0.5 for s_, e in zip(starts, ends)))
        if singles and approaches:
            out.append(Insight(who, "strength", "Comes to the net",
                               f"Moved in to the net {approaches} time(s). Finishing points at the net shortens "
                               "rallies; approach behind a deep or short-angled ball."))
        elif singles and t[-1] >= 20:
            out.append(Insight(who, "info", "Stayed back the whole clip",
                               "Never came in past the service line. Look for short balls to attack and follow in."))
        if singles:
            rec = _excursions(np.abs(xy[:, 0] - court.center_x), t)
            done = [r for r in rec if r is not None]
            if len(rec) >= 2 and (len(done) < len(rec) / 2 or np.median(done) > 2.5):
                out.append(Insight(who, "improve", "Slow to recover to the middle",
                                   f"After being pulled wide ({len(rec)} times), you got back toward the center "
                                   + (f"in {np.median(done):.1f} s on average" if done else "rarely") +
                                   ". Recover with side shuffles right after your shot, and split-step as the "
                                   "opponent hits."))
            elif len(done) >= 2:
                out.append(Insight(who, "strength", "Recovers to the middle",
                                   f"Got back toward the center {np.median(done):.1f} s after being pulled wide, "
                                   f"{len(done)} times."))

    for team, players in sorted(by_team.items()):
        if len(players) != 2:
            continue
        (pa, fa, xa), (pb, fb, xb) = players
        common, ia, ib = np.intersect1d(fa, fb, return_indices=True)
        if len(common) < 5:
            continue
        a, b = xa[ia], xb[ib]
        ua, ub = tennis_up(court, team, a[:, 1]), tennis_up(court, team, b[:, 1])
        ka, kb = tennis_back(court, team, a[:, 1]), tennis_back(court, team, b[:, 1])
        both_up, both_back = ua & ub, ka & kb
        one_one = (ua & kb) | (ub & ka)
        who = f"{team.capitalize()} team (players {pa} and {pb})"
        if both_up.mean() >= 0.3:
            out.append(Insight(who, "strength", "Takes the net together",
                               f"Both partners were at the net {both_up.mean():.0%} of the time, the strongest "
                               "position in doubles."))
        if one_one.mean() >= 0.5:
            same_half = (np.sign(a[:, 0] - court.center_x) == np.sign(b[:, 0] - court.center_x)) & one_one
            out.append(Insight(who, "info", "Mostly one up, one back",
                               f"Played one up and one back {one_one.mean():.0%} of the time. The opening in this "
                               "formation is the diagonal between partners; when the back player gets a short "
                               "ball, both move up."))
            if same_half.sum() > 0.4 * one_one.sum():
                out.append(Insight(who, "improve", "Partners on the same half",
                                   f"In {same_half.sum() / one_one.sum():.0%} of one-up-one-back moments both partners "
                                   "were on the same half of the court, leaving the other half open. The net player "
                                   "should cover the half in front of them while the partner takes the other."))
        if both_back.sum() >= 5:
            gap = float(np.median(np.abs(a[both_back, 0] - b[both_back, 0])))
            if gap > 6.5:
                out.append(Insight(who, "improve", "Gap down the middle",
                                   f"When both were back, partners were {gap:.1f} m apart across the court. On an "
                                   "11 m wide court that leaves the middle open; shift toward the ball together."))
            elif gap < 2.5:
                out.append(Insight(who, "improve", "Bunched together",
                                   f"When both were back, partners were only {gap:.1f} m apart, which opens the "
                                   "alleys. Each partner should cover their half and shade toward the ball."))
    return out


def to_markdown(insights: list[Insight]) -> str:
    icon = {"strength": "✅", "improve": "🎯", "info": "ℹ️"}
    lines = []
    for kind, head in (("improve", "Work on"), ("strength", "Doing well"), ("info", "Notes")):
        items = [i for i in insights if i.kind == kind]
        if items:
            lines.append(f"### {head}")
            lines += [f"- {icon[kind]} **{i.who}: {i.title}.** {i.detail}" for i in items]
    return "\n".join(lines) or "Nothing to report yet."


def _context(summary: dict, insights: list[Insight]) -> str:
    cal = summary.get("calibration", {})
    slim = {"sport": summary.get("sport", "pickleball"), "format": summary.get("format"),
            "camera": {"court_fit": cal.get("score"), "view": cal.get("camera", {}).get("view"),
                       "height_m": cal.get("camera", {}).get("height_m")},
            "players": summary.get("players"), "teams": summary.get("teams"),
            "insights": [asdict(i) for i in insights]}
    return json.dumps(slim, indent=1, default=str)


SYSTEMS = {"pickleball": (
    "You are an experienced pickleball doubles coach. You are given movement data measured from a match "
    "video: court positions in meters (court is 6.1 m wide and 13.4 m long, kitchen line 2.13 m from the "
    "net), time at the kitchen line, partner spacing, and rule-based findings. You do not see the video, "
    "the ball or the shots, so do not invent anything about strokes, shot choice or the score. Be specific, "
    "practical and encouraging, refer to players by their numbers, and suggest one or two drills when useful."
), "tennis": (
    "You are an experienced tennis coach. You are given movement data measured from a match video: court "
    "positions in meters (doubles court 10.97 m wide and 23.77 m long, singles court 8.23 m wide, service line "
    "6.40 m from the net), how far behind the baseline players stand, time in no-man's land, net approaches, "
    "recovery to the middle after being pulled wide, doubles formations and rule-based findings. You do not see "
    "the video, the ball or the strokes, so do not invent anything about technique, shot choice or the score. "
    "Be specific, practical and encouraging, refer to players by their numbers, and suggest one or two drills "
    "when useful."
)}
SYSTEM = SYSTEMS["pickleball"]


def llm_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def ask_coach(summary: dict, insights: list[Insight], question: str | None = None,
              history: list[dict] | None = None) -> str:
    """Coach's-voice summary (question=None) or an answer to a follow-up question."""
    import anthropic

    msgs = [{"role": "user", "content": "Match data:\n" + _context(summary, insights)},
            {"role": "assistant", "content": "Got it. I have the movement data for this clip."}]
    msgs += history or []
    msgs.append({"role": "user", "content": question or
                 "Give the players a short coaching debrief: the two most important things to work on, "
                 "what they are doing well, and one drill for next session. Under 200 words."})
    system = SYSTEMS.get(summary.get("sport", "pickleball"), SYSTEM)
    resp = anthropic.Anthropic().messages.create(
        model=os.environ.get("PBCOACH_MODEL", DEFAULT_MODEL), max_tokens=800, system=system, messages=msgs)
    return "".join(b.text for b in resp.content if b.type == "text")


def offline_answer(question: str, insights: list[Insight]) -> str:
    """Keyword match over the insights, for when no API key is configured."""
    q = question.lower()
    hits = [i for i in insights if any(w in (i.who + i.title + i.detail).lower()
                                       for w in q.replace("?", " ").split() if len(w) > 3)]
    body = to_markdown(hits or insights)
    return (body + "\n\n_Set `ANTHROPIC_API_KEY` before starting the demo to get free-form answers from "
            "Claude. Without it, answers are matched from the findings above._")
