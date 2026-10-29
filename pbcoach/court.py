"""Court models (pickleball and tennis) in court meters.

Coordinate frame (used everywhere downstream, independent of camera angle and sport):
  origin = left corner of the near baseline, x across the court (0..width),
  y along the court (0..length), z up. The net sits at y = net_y.

Every sport-specific number lives on a Court; the rest of the pipeline reads it from
`Calibration.court`. The module-level constants are the pickleball court, kept for
existing callers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FT = 0.3048
IN = 0.0254


@dataclass(frozen=True)
class Court:
    sport: str
    width: float
    length: float
    line_width: float
    net_height_post: float
    net_height_center: float
    net_overhang: float          # how far outside the sidelines the net posts stand
    across_y: tuple              # painted lines of constant y (calibration hypotheses)
    along_x: tuple               # painted lines of constant x
    segments: tuple              # painted segments ((x1, y1), (x2, y2))
    keypoints: dict              # named line intersections that physically exist on the court
    keypoint_help: dict          # plain-language name of each keypoint, for click-to-fix
    front_y: float               # each side's coaching reference line, as distance from its baseline
    front_name: str              # "kitchen line" / "baseline"
    shade: tuple = ()            # (x0, y0, x1, y1) areas drawn in a lighter color on maps
    runoff_side: float = 2.0     # where players can be, beyond the sidelines / baselines
    runoff_back: float = 2.0
    default_players: int = 4

    @property
    def net_y(self) -> float:
        return self.length / 2

    @property
    def center_x(self) -> float:
        return self.width / 2

    @property
    def corners(self) -> np.ndarray:
        return np.array([(0.0, 0.0), (self.width, 0.0), (self.width, self.length), (0.0, self.length)])

    def front_line(self, team: str) -> float:
        """y of the team's own reference line (kitchen line in pickleball, baseline in tennis)."""
        return self.front_y if team == "near" else self.length - self.front_y

    def behind(self, team: str, y):
        """Meters behind the team's reference line, toward its own back fence (negative = in front)."""
        y = np.asarray(y, dtype=np.float64)
        return self.front_line(team) - y if team == "near" else y - self.front_line(team)

    def from_baseline(self, team: str, y):
        """Meters in front of the team's own baseline (negative = behind it)."""
        y = np.asarray(y, dtype=np.float64)
        return y if team == "near" else self.length - y

    def side(self, y: float) -> str:
        return "near" if y < self.net_y else "far"

    def in_bounds(self, x: float, y: float, margin: float | None = None) -> bool:
        """Inside the court plus the area players can run to."""
        mx = self.runoff_side if margin is None else margin
        my = self.runoff_back if margin is None else margin
        return -mx <= x <= self.width + mx and -my <= y <= self.length + my

    def net_tape_height(self, x):
        """Height of the net's top at x: lowest in the middle, rising linearly to the posts."""
        x = np.asarray(x, dtype=np.float64)
        return self.net_height_center + (self.net_height_post - self.net_height_center) * np.minimum(
            1.0, np.abs(x - self.center_x) / (self.center_x + self.net_overhang))

    def sample_segments(self, step: float = 0.2) -> np.ndarray:
        """Points along every painted segment, spaced about `step` meters apart."""
        pts = []
        for (x1, y1), (x2, y2) in self.segments:
            n = max(2, int(np.hypot(x2 - x1, y2 - y1) / step) + 1)
            t = np.linspace(0.0, 1.0, n)
            pts.append(np.stack([x1 + (x2 - x1) * t, y1 + (y2 - y1) * t], axis=1))
        return np.concatenate(pts)

    def line_pieces(self, kind: str, v: float) -> list:
        """Painted segments lying on the model line y = v (kind "y") or x = v (kind "x")."""
        i = 1 if kind == "y" else 0
        return [s for s in self.segments if abs(s[0][i] - v) < 1e-9 and abs(s[1][i] - v) < 1e-9]

    def zone(self, x: float, y: float, margin: float = 0.0) -> str:
        """Coarse court zone of a ground position, for coaching metrics."""
        if not (-margin - self.runoff_side <= x <= self.width + margin + self.runoff_side
                and -margin - self.runoff_back <= y <= self.length + margin + self.runoff_back):
            return "out"
        side = self.side(y)
        b = float(self.from_baseline(side, y))
        if self.sport == "pickleball":
            if not (-margin <= x <= self.width + margin and -margin <= y <= self.length + margin):
                return "out"
            k = float(self.behind(side, y))
            if k < 0:
                return f"{side}_kitchen"
            if k <= 1.0:
                return f"{side}_kitchen_line"
            if b < 1.5:
                return f"{side}_baseline"
            return f"{side}_transition"
        # Tennis: depth bands measured from the player's own baseline.
        service = self.net_y - TENNIS_SERVICE_FROM_NET
        if b < -2.0:
            return f"{side}_deep"
        if b <= 1.0:
            return f"{side}_baseline"
        if b < service:
            return f"{side}_midcourt"
        return f"{side}_net"


# ---------------------------------------------------------------- pickleball

_PW, _PL = 20 * FT, 44 * FT
_NVZ = 7 * FT
_KN, _KF = _PL / 2 - _NVZ, _PL / 2 + _NVZ
_PC = _PW / 2

PICKLEBALL = Court(
    sport="pickleball",
    width=_PW,                     # 6.096 m
    length=_PL,                    # 13.4112 m
    line_width=2 * IN,
    net_height_post=36 * IN,
    net_height_center=34 * IN,
    net_overhang=0.3,
    across_y=(0.0, _KN, _KF, _PL),
    along_x=(0.0, _PC, _PW),
    segments=(
        ((0.0, 0.0), (_PW, 0.0)),        # near baseline
        ((0.0, _PL), (_PW, _PL)),        # far baseline
        ((0.0, 0.0), (0.0, _PL)),        # left sideline
        ((_PW, 0.0), (_PW, _PL)),        # right sideline
        ((0.0, _KN), (_PW, _KN)),        # near kitchen line
        ((0.0, _KF), (_PW, _KF)),        # far kitchen line
        ((_PC, 0.0), (_PC, _KN)),        # near centerline
        ((_PC, _KF), (_PC, _PL)),        # far centerline
    ),
    keypoints={
        "near_left": (0.0, 0.0), "near_center": (_PC, 0.0), "near_right": (_PW, 0.0),
        "nvz_near_left": (0.0, _KN), "nvz_near_center": (_PC, _KN), "nvz_near_right": (_PW, _KN),
        "nvz_far_left": (0.0, _KF), "nvz_far_center": (_PC, _KF), "nvz_far_right": (_PW, _KF),
        "far_left": (0.0, _PL), "far_center": (_PC, _PL), "far_right": (_PW, _PL),
    },
    keypoint_help={
        "near_left": "near baseline, left corner", "near_center": "near baseline, center line",
        "near_right": "near baseline, right corner", "nvz_near_left": "near kitchen line, left end",
        "nvz_near_center": "near kitchen line, center", "nvz_near_right": "near kitchen line, right end",
        "nvz_far_left": "far kitchen line, left end", "nvz_far_center": "far kitchen line, center",
        "nvz_far_right": "far kitchen line, right end", "far_left": "far baseline, left corner",
        "far_center": "far baseline, center line", "far_right": "far baseline, right corner",
    },
    front_y=_KN,
    front_name="kitchen line",
    shade=((0.0, _KN, _PW, _KF),),
    runoff_side=2.0,
    runoff_back=2.0,
    default_players=4,
)

# ---------------------------------------------------------------- tennis (ITF dimensions)

TENNIS_SERVICE_FROM_NET = 21 * FT       # 6.40 m
_TW, _TL = 36 * FT, 78 * FT             # 10.97 x 23.77 m doubles court
_AL = 4.5 * FT                          # 1.37 m doubles alley
_SL, _SR = _AL, _TW - _AL               # singles sidelines
_SN, _SF = _TL / 2 - TENNIS_SERVICE_FROM_NET, _TL / 2 + TENNIS_SERVICE_FROM_NET
_TC = _TW / 2
_MARK = 0.10                            # center marks on the baselines

TENNIS = Court(
    sport="tennis",
    width=_TW,
    length=_TL,
    line_width=2 * IN,
    net_height_post=42 * IN,            # 1.07 m
    net_height_center=36 * IN,          # 0.914 m
    net_overhang=3 * FT,                # posts 0.914 m outside the doubles sidelines
    across_y=(0.0, _SN, _SF, _TL),
    along_x=(0.0, _SL, _TC, _SR, _TW),
    segments=(
        ((0.0, 0.0), (_TW, 0.0)),        # near baseline
        ((0.0, _TL), (_TW, _TL)),        # far baseline
        ((0.0, 0.0), (0.0, _TL)),        # left doubles sideline
        ((_TW, 0.0), (_TW, _TL)),        # right doubles sideline
        ((_SL, 0.0), (_SL, _TL)),        # left singles sideline
        ((_SR, 0.0), (_SR, _TL)),        # right singles sideline
        ((_SL, _SN), (_SR, _SN)),        # near service line
        ((_SL, _SF), (_SR, _SF)),        # far service line
        ((_TC, _SN), (_TC, _SF)),        # center service line
        ((_TC, 0.0), (_TC, _MARK)),      # near center mark
        ((_TC, _TL - _MARK), (_TC, _TL)),  # far center mark
    ),
    keypoints={
        "near_left": (0.0, 0.0), "near_singles_left": (_SL, 0.0), "near_center": (_TC, 0.0),
        "near_singles_right": (_SR, 0.0), "near_right": (_TW, 0.0),
        "service_near_left": (_SL, _SN), "service_near_center": (_TC, _SN), "service_near_right": (_SR, _SN),
        "service_far_left": (_SL, _SF), "service_far_center": (_TC, _SF), "service_far_right": (_SR, _SF),
        "far_left": (0.0, _TL), "far_singles_left": (_SL, _TL), "far_center": (_TC, _TL),
        "far_singles_right": (_SR, _TL), "far_right": (_TW, _TL),
    },
    keypoint_help={
        "near_left": "near baseline, left doubles corner", "near_singles_left": "near baseline, left singles line",
        "near_center": "near baseline, center mark", "near_singles_right": "near baseline, right singles line",
        "near_right": "near baseline, right doubles corner",
        "service_near_left": "near service line, left end", "service_near_center": "near service line, center (T)",
        "service_near_right": "near service line, right end",
        "service_far_left": "far service line, left end", "service_far_center": "far service line, center (T)",
        "service_far_right": "far service line, right end",
        "far_left": "far baseline, left doubles corner", "far_singles_left": "far baseline, left singles line",
        "far_center": "far baseline, center mark", "far_singles_right": "far baseline, right singles line",
        "far_right": "far baseline, right doubles corner",
    },
    front_y=0.0,
    front_name="baseline",
    shade=((_SL, _SN, _SR, _SF),),
    runoff_side=3.66,                   # ITF recommended minimum run-off: 12 ft beside, 21 ft behind
    runoff_back=6.4,
    default_players=2,
)

COURTS = {c.sport: c for c in (PICKLEBALL, TENNIS)}


def get(sport: str | Court | None) -> Court:
    if isinstance(sport, Court):
        return sport
    return COURTS[(sport or "pickleball").lower()]


# Pickleball constants, for existing callers.
WIDTH = PICKLEBALL.width
LENGTH = PICKLEBALL.length
NET_Y = PICKLEBALL.net_y
NVZ_DEPTH = _NVZ
NVZ_NEAR_Y = _KN
NVZ_FAR_Y = _KF
CENTER_X = PICKLEBALL.center_x
LINE_WIDTH = PICKLEBALL.line_width
NET_HEIGHT_POST = PICKLEBALL.net_height_post
NET_HEIGHT_CENTER = PICKLEBALL.net_height_center
ACROSS_Y = PICKLEBALL.across_y
ALONG_X = PICKLEBALL.along_x
SEGMENTS = list(PICKLEBALL.segments)
KEYPOINTS = PICKLEBALL.keypoints
CORNERS = PICKLEBALL.corners


def sample_segments(step: float = 0.2) -> np.ndarray:
    return PICKLEBALL.sample_segments(step)


def zone(x: float, y: float, margin: float = 0.0) -> str:
    return PICKLEBALL.zone(x, y, margin)
