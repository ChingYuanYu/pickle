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

from . import court

KITCHEN_BAND_M = 1.0      # "at the kitchen line" = within 1 m of it
BACK_M = 3.0              # "back" = more than 3 m behind the kitchen line
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
    return court.NVZ_NEAR_Y - xy[:, 1] if team == "near" else xy[:, 1] - court.NVZ_FAR_Y


def analyze(rows: list[dict], summary: dict, fps: float) -> list[Insight]:
    tracks = _tracks(rows)
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

    for pid, (team, frames, xy) in sorted(tracks.items()):
        who = f"Player {pid} ({team} side)"
        behind = _behind_kitchen(team, xy)
        at_line = np.abs(behind) <= KITCHEN_BAND_M
        t = (frames - frames[0]) / fps
        # Moving up through the transition zone is fine; standing in it is not.
        k = max(1, int(round(0.5 * fps)))
        forward = np.gradient(-np.convolve(behind, np.ones(k) / k, mode="same"), t) if len(t) > 2 else np.zeros(len(t))
        transition = (behind > KITCHEN_BAND_M) & (behind < court.NVZ_NEAR_Y - 1.5) & (forward < 0.5)
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
    slim = {"camera": {"court_fit": cal.get("score"), "view": cal.get("camera", {}).get("view"),
                       "height_m": cal.get("camera", {}).get("height_m")},
            "players": summary.get("players"), "teams": summary.get("teams"),
            "insights": [asdict(i) for i in insights]}
    return json.dumps(slim, indent=1, default=str)


SYSTEM = (
    "You are an experienced pickleball doubles coach. You are given movement data measured from a match "
    "video: court positions in meters (court is 6.1 m wide and 13.4 m long, kitchen line 2.13 m from the "
    "net), time at the kitchen line, partner spacing, and rule-based findings. You do not see the video, "
    "the ball or the shots, so do not invent anything about strokes, shot choice or the score. Be specific, "
    "practical and encouraging, refer to players by their numbers, and suggest one or two drills when useful."
)


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
                 "Give this team a short coaching debrief: the two most important things to work on, "
                 "what they are doing well, and one drill for next session. Under 200 words."})
    resp = anthropic.Anthropic().messages.create(
        model=os.environ.get("PBCOACH_MODEL", DEFAULT_MODEL), max_tokens=800, system=SYSTEM, messages=msgs)
    return "".join(b.text for b in resp.content if b.type == "text")


def offline_answer(question: str, insights: list[Insight]) -> str:
    """Keyword match over the insights, for when no API key is configured."""
    q = question.lower()
    hits = [i for i in insights if any(w in (i.who + i.title + i.detail).lower()
                                       for w in q.replace("?", " ").split() if len(w) > 3)]
    body = to_markdown(hits or insights)
    return (body + "\n\n_Set `ANTHROPIC_API_KEY` before starting the demo to get free-form answers from "
            "Claude. Without it, answers are matched from the findings above._")
