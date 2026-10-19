"""Player detection, angle-aware foot points, and tracking in court meters.

Two ideas keep this independent of the camera angle:
- The foot point is found along the image direction of world "up" at that spot (from the
  recovered camera), not by assuming people are upright in the image. That matters for
  overhead, tilted and wide-angle views.
- Tracking happens on the court plane in meters, so gating distances, speeds and team
  sides mean the same thing from every viewpoint.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from . import court
from .calibrate import Calibration


@dataclass
class Detection:
    box: tuple[float, float, float, float]   # x1, y1, x2, y2 in pixels
    conf: float
    foot_px: np.ndarray | None = None        # image foot point
    court_xy: np.ndarray | None = None       # meters
    mask: np.ndarray | None = None           # optional blob pixels (N, 2) as x, y


# ---------------------------------------------------------------- geometry

def up_direction(cal: Calibration, px: np.ndarray) -> np.ndarray:
    """Unit image direction of world-up at pixel px (assumed to be on the ground)."""
    g = cal.to_court([px])[0]
    a = cal.pose.project3d([(g[0], g[1], 0.0)])[0]
    b = cal.pose.project3d([(g[0], g[1], 1.0)])[0]
    d = b - a
    n = np.linalg.norm(d)
    return d / n if n > 1e-9 else np.array([0.0, -1.0])


def foot_from_box(cal: Calibration, box) -> np.ndarray:
    """Where the person's vertical axis leaves the box on the ground side."""
    x1, y1, x2, y2 = box
    c = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    u = up_direction(cal, np.array([(x1 + x2) / 2, y2]))
    d = -u
    ts = []
    for k, lo, hi in ((0, x1, x2), (1, y1, y2)):
        if abs(d[k]) > 1e-9:
            ts += [(lo - c[k]) / d[k], (hi - c[k]) / d[k]]
    t = min(t for t in ts if t > 0) if any(t > 0 for t in ts) else (y2 - c[1])
    return c + d * t


def foot_from_blob(cal: Calibration, pts: np.ndarray) -> np.ndarray:
    """Lowest part of a blob along the local world-down direction."""
    c = pts.mean(0)
    for _ in range(2):
        u = up_direction(cal, c)
        s = pts @ (-u)
        lo = np.percentile(s, 97)
        sel = pts[s >= lo]
        c = sel.mean(0)
    return c


def expected_height_px(cal: Calibration, court_xy, height_m: float = 1.75) -> float:
    a = cal.pose.project3d([(court_xy[0], court_xy[1], 0.0)])[0]
    b = cal.pose.project3d([(court_xy[0], court_xy[1], height_m)])[0]
    return float(np.linalg.norm(b - a))


# ---------------------------------------------------------------- detectors

class YoloDetector:
    """Ultralytics YOLO person detector (downloads weights on first use)."""

    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.25, imgsz: int = 1280):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        r = self.model.predict(frame, classes=[0], conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        out = []
        for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            out.append(Detection((float(x1), float(y1), float(x2), float(y2)), float(c)))
        return out


class ForegroundDetector:
    """Moving-blob detector against a static background. Needs no model and works from any
    angle, but requires a fixed camera and merges players who overlap in the image."""

    def __init__(self, background: np.ndarray, thresh: float = 28.0):
        self.bg = cv2.cvtColor(cv2.GaussianBlur(background, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.int16)
        self.thresh = thresh
        h, w = background.shape[:2]
        self.min_area = max(30, int(1e-4 * h * w))

    def foreground(self, frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.int16)
        diff = np.abs(lab - self.bg)
        score = diff[..., 0] * 0.5 + diff[..., 1] + diff[..., 2]
        fg = (score > self.thresh).astype(np.uint8) * 255
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        fg = self.foreground(frame)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(fg)
        out = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < self.min_area:
                continue
            ys, xs = np.nonzero(labels[y:y + h, x:x + w] == i)
            pts = np.stack([xs + x, ys + y], 1).astype(np.float64)
            out.append(Detection((x, y, x + w, y + h), min(1.0, area / (20 * self.min_area)), mask=pts))
        return out


class OccupancyDetector:
    """Finds people directly on the court plane from a foreground mask (a simplified
    probabilistic occupancy map). For every ground cell it checks whether the image of a
    person-sized vertical column standing there is covered by foreground. Because the test
    uses the recovered camera, it works the same from behind the baseline, the side, a corner
    or overhead, and it separates players whose image blobs merge."""

    def __init__(self, cal: Calibration, background: np.ndarray, cell_m: float = 0.1, margin_m: float = 2.0,
                 heights=(0.1, 0.4, 0.7, 1.0, 1.3, 1.6), thresh: float = 0.8):
        self.fg = ForegroundDetector(background)
        h, w = background.shape[:2]
        self.size = (w, h)
        xs = np.arange(-margin_m, court.WIDTH + margin_m + 1e-6, cell_m)
        ys = np.arange(-margin_m, court.LENGTH + margin_m + 1e-6, cell_m)
        self.gx, self.gy = np.meshgrid(xs, ys)
        self.cell = cell_m
        self.thresh = thresh
        pts = []
        for z in heights:
            X = np.stack([self.gx.ravel(), self.gy.ravel(), np.full(self.gx.size, z)], 1)
            pts.append(cal.pose.project3d(X))
        self.px = np.stack(pts, 1)                              # (cells, heights, 2)
        self.inside = ((self.px[..., 0] >= 0) & (self.px[..., 0] < w) & (self.px[..., 1] >= 0) & (self.px[..., 1] < h))
        self.ix = np.clip(self.px[..., 0], 0, w - 1).astype(np.int32)
        self.iy = np.clip(self.px[..., 1], 0, h - 1).astype(np.int32)
        # A column needs its feet and at least half its height inside the frame to be judged.
        self.valid = self.inside[:, 0] & (self.inside.sum(1) >= len(heights) // 2 + 1)
        # Ground just in front of each cell (toward the camera) must be empty: a real player's
        # feet sit on visible ground, while "ghost" cells behind a player (common with low
        # cameras, where viewing rays are nearly horizontal) have that player's body there.
        cam = cal.pose.position
        g = np.stack([self.gx.ravel(), self.gy.ravel()], 1)
        to_cam = cam[:2] - g
        to_cam /= np.maximum(np.linalg.norm(to_cam, axis=1, keepdims=True), 1e-6)
        front = np.stack([cal.pose.project3d(np.hstack([g + d * to_cam, np.zeros((len(g), 1))]))
                          for d in (0.45, 0.9, 1.35)], 1)               # (cells, 3, 2)
        self.front_in = (front[..., 0] >= 0) & (front[..., 0] < w) & (front[..., 1] >= 0) & (front[..., 1] < h)
        self.fx = np.clip(front[..., 0], 0, w - 1).astype(np.int32)
        self.fy = np.clip(front[..., 1], 0, h - 1).astype(np.int32)
        # The net hides far players' feet in many views: samples behind it count as unknown.
        from .calibrate import _net_geometry
        net = np.zeros((h, w), bool)
        poly, _ = _net_geometry(cal.pose)
        if poly is not None and np.isfinite(poly).all() and np.abs(poly).max() < 1e6:
            m = np.zeros((h, w), np.uint8)
            cv2.fillPoly(m, [poly.astype(np.int32)], 255)
            net = m > 0
        self.unknown = net[self.iy, self.ix] | ~self.inside
        self.front_in &= ~net[self.fy, self.fx]
        self.cal = cal

    def occupancy(self, fg: np.ndarray, hidden: np.ndarray | None = None):
        """Per ground cell: share of its person-column samples on foreground, and whether its
        feet are actually seen. Samples hidden by the net or by already-accepted players are
        unknown rather than empty."""
        # Feet are judged on a slightly dilated mask: legs are thin and may straddle a sample.
        feet_fg = cv2.dilate(fg, np.ones((7, 7), np.uint8)) > 0
        fg = fg > 0
        unknown = self.unknown.copy()
        if hidden is not None:
            unknown |= hidden[self.iy, self.ix]
        known = ~unknown
        hit = fg[self.iy, self.ix] & known
        occ = hit.sum(1) / np.maximum(known.sum(1), 1)
        feet_seen = feet_fg[self.iy[:, 0], self.ix[:, 0]] & known[:, 0]
        hit[:, 0] |= feet_seen
        feet_ok = feet_seen | (unknown[:, 0] & self.inside[:, 0])
        front_in = self.front_in if hidden is None else self.front_in & ~hidden[self.fy, self.fx]
        occ[~self.valid | ~feet_ok | (known.sum(1) < 3)] = 0
        occ[(front_in & fg[self.fy, self.fx]).any(1)] *= 0.3
        return occ.reshape(self.gx.shape), feet_seen.reshape(self.gx.shape)

    def __call__(self, frame: np.ndarray, max_people: int = 8) -> list[Detection]:
        """Greedy explaining-away. Cells whose feet are visible are accepted nearest-to-camera
        first (a ghost further along a viewing ray cannot beat the real player in front of it);
        cells whose feet are hidden (behind the net or another player) need stronger evidence.
        Each accepted player's silhouette becomes "unknown" for the cells still to be judged."""
        fg = self.fg.foreground(frame)
        h, w = fg.shape
        hidden = np.zeros((h, w), bool)
        cam = self.cal.pose.position[:2]
        dist = np.hypot(self.gx - cam[0], self.gy - cam[1])
        r = max(1, int(round(0.45 / self.cell)))
        out = []
        for _ in range(4 * max_people):
            if len(out) >= max_people:
                break
            occ, feet = self.occupancy(fg, hidden)
            sm = cv2.blur(occ.astype(np.float32), (3, 3))
            seen = (sm >= self.thresh) & (occ >= self.thresh) & feet
            solid = seen & (cv2.blur(seen.astype(np.float32), (3, 3)) >= 2 / 9)  # isolated cells are noise
            seen = solid if solid.any() else seen
            if seen.any():
                iy, ix = np.unravel_index(np.argmin(np.where(seen, dist, np.inf)), seen.shape)
                y0, y1, x0, x1 = max(0, iy - r), iy + r + 1, max(0, ix - r), ix + r + 1
                wts = np.clip(occ[y0:y1, x0:x1] - 0.5, 0, None)
                cx = float((self.gx[y0:y1, x0:x1] * wts).sum() / wts.sum())
                cy = float((self.gy[y0:y1, x0:x1] * wts).sum() / wts.sum())
            else:
                # Feet hidden (net or another player): the image only pins the player to a
                # stretch along the viewing ray, so take the centre of that stretch.
                strong = (sm >= max(self.thresh, 0.9)).astype(np.uint8)
                if not strong.any():
                    break
                n, lab = cv2.connectedComponents(strong)
                iy, ix = np.unravel_index(np.argmin(np.where(strong > 0, dist, np.inf)), sm.shape)
                comp = lab == lab[iy, ix]
                wts = np.where(comp, np.clip(occ - 0.5, 0, None), 0)
                if wts.sum() <= 0:
                    wts = comp.astype(float)
                cx = float((self.gx * wts).sum() / wts.sum())
                cy = float((self.gy * wts).sum() / wts.sum())
            body = self.cal.pose.project3d(_cylinder_pts((cx, cy), 0.3, 0.0, 1.85))
            m = np.zeros((h, w), np.uint8)
            cv2.fillPoly(m, [cv2.convexHull(body.astype(np.int32))], 255)
            hidden |= m > 0
            if any(np.hypot(cx - d.court_xy[0], cy - d.court_xy[1]) < 0.6 for d in out):
                continue  # leftover pixels of someone already found (e.g. a shadow)
            x_lo, y_lo = body.min(0)
            x_hi, y_hi = body.max(0)
            out.append(Detection((x_lo, y_lo, x_hi, y_hi), float(sm[iy, ix]),
                                 foot_px=self.cal.to_image([(cx, cy)])[0], court_xy=np.array([cx, cy])))
        return out


def _cylinder_pts(c, r, z0, z1, n=16):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a)], 1)
    return np.vstack([np.hstack([ring, np.full((n, 1), z0)]), np.hstack([ring, np.full((n, 1), z1)])])


def locate(cal: Calibration, dets: list[Detection], margin_m: float = 2.0,
           size_range=(0.35, 1.8)) -> list[Detection]:
    """Attach foot points and court positions; drop people off court or of implausible size."""
    kept = []
    for d in dets:
        if d.court_xy is not None:  # already located on the ground (occupancy detector)
            kept.append(d)
            continue
        d.foot_px = foot_from_blob(cal, d.mask) if d.mask is not None else foot_from_box(cal, d.box)
        d.court_xy = cal.to_court([d.foot_px])[0]
        x, y = d.court_xy
        if not (-margin_m <= x <= court.WIDTH + margin_m and -margin_m <= y <= court.LENGTH + margin_m):
            continue
        exp_h = expected_height_px(cal, d.court_xy)
        u = up_direction(cal, d.foot_px)
        if d.mask is not None:
            extent = float(np.ptp(d.mask @ u))
        else:
            x1, y1, x2, y2 = d.box
            corners = np.array([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
            extent = float(np.ptp(corners @ u))
        if exp_h > 0 and not (size_range[0] * exp_h <= extent <= size_range[1] * exp_h * 1.6):
            continue
        kept.append(d)
    return kept


# ---------------------------------------------------------------- tracking

@dataclass
class Track:
    id: int
    x: np.ndarray                      # state: x, y, vx, vy
    P: np.ndarray
    hits: int = 1
    misses: int = 0
    history: list = field(default_factory=list)

    @property
    def pos(self):
        return self.x[:2]

    @property
    def team(self) -> str:
        return "near" if self.x[1] < court.NET_Y else "far"


class CourtTracker:
    """Constant-velocity Kalman tracks on the court plane with Hungarian assignment."""

    def __init__(self, fps: float, max_players: int = 4, gate_m: float = 1.5, max_misses: int | None = None):
        self.dt = 1.0 / fps
        self.max_players = max_players
        self.gate = gate_m
        self.max_misses = max_misses or int(2 * fps)
        self.coast = max(1, int(0.5 * fps))
        self.tracks: list[Track] = []
        self.next_id = 1
        dt = self.dt
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1.0]])
        self.Q = np.diag([0.01, 0.01, 1.0, 1.0]) * dt
        self.Hm = np.array([[1, 0, 0, 0], [0, 1, 0, 0.0]])
        self.R = np.eye(2) * 0.05 ** 2

    def update(self, dets: list[Detection], frame_idx: int) -> list[Track]:
        for t in self.tracks:
            t.x = self.F @ t.x
            t.P = self.F @ t.P @ self.F.T + self.Q
        Z = np.array([d.court_xy for d in dets]) if dets else np.zeros((0, 2))
        assigned_t, assigned_d = set(), set()
        if len(self.tracks) and len(Z):
            C = np.linalg.norm(np.array([t.pos for t in self.tracks])[:, None] - Z[None], axis=2)
            ri, ci = linear_sum_assignment(C)
            for r, c in zip(ri, ci):
                # The gate widens while a track is unseen, since the player kept moving.
                if C[r, c] <= self.gate + 0.3 * self.tracks[r].misses:
                    self._correct(self.tracks[r], Z[c])
                    assigned_t.add(r)
                    assigned_d.add(c)
        for i, t in enumerate(self.tracks):
            if i not in assigned_t:
                t.misses += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        # New tracks, strongest detections first, while there is room on court.
        free = sorted((c for c in range(len(Z)) if c not in assigned_d), key=lambda c: -dets[c].conf)
        for c in free:
            if len(self.tracks) >= self.max_players:
                stale = max(self.tracks, key=lambda t: t.misses)
                if stale.misses < 3:
                    break
                self.tracks.remove(stale)  # a long-lost track gives way to a fresh detection
            if any(np.linalg.norm(t.pos - Z[c]) < 0.6 for t in self.tracks):
                continue
            self.tracks.append(Track(self.next_id, np.array([*Z[c], 0, 0.0]), np.diag([0.1, 0.1, 1, 1.0])))
            self.next_id += 1
        for t in self.tracks:
            t.x[2:] *= 0.0 if t.misses else 1.0   # do not coast on stale velocity
            if t.misses <= self.coast:
                t.history.append((frame_idx, float(t.x[0]), float(t.x[1])))
        # Briefly occluded players are reported at their last known position.
        return [t for t in self.tracks if t.misses <= self.coast]

    def _correct(self, t: Track, z):
        S = self.Hm @ t.P @ self.Hm.T + self.R
        K = t.P @ self.Hm.T @ np.linalg.inv(S)
        t.x = t.x + K @ (z - self.Hm @ t.x)
        t.P = (np.eye(4) - K @ self.Hm) @ t.P
        t.hits += 1
        t.misses = 0
