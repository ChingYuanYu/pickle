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
    cov: np.ndarray | None = None            # 2x2 court-position covariance (m^2)
    keypoints: np.ndarray | None = None      # optional (17, 3) COCO pose: x, y, confidence
    feat: np.ndarray | None = None           # optional appearance (clothing color histogram)


def appearance(frame: np.ndarray, d: Detection) -> np.ndarray | None:
    """Hue-saturation histogram of the torso and shorts: what tells two players apart."""
    x1, y1, x2, y2 = d.box
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = int(x1 + 0.2 * bw), int(x2 - 0.2 * bw)
    y1, y2 = int(y1 + 0.2 * bh), int(y1 + 0.65 * bh)
    patch = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if patch.size < 12:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    # Dark and washed-out pixels have unreliable hue: count them by brightness instead.
    h = cv2.calcHist([hsv], [0, 1], (hsv[..., 2] > 50).astype(np.uint8), [12, 4], [0, 180, 40, 256]).ravel()
    v = cv2.calcHist([hsv], [2], None, [4], [0, 256]).ravel()
    f = np.concatenate([h, v * 0.5]).astype(np.float32)
    return f / max(float(f.sum()), 1e-6)


def appearance_distance(a, b) -> float:
    """Bhattacharyya distance in [0, 1]; 0 when either side is unknown."""
    if a is None or b is None:
        return 0.0
    return float(np.sqrt(max(0.0, 1.0 - np.sum(np.sqrt(a * b)))))


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


def foot_from_keypoints(cal: Calibration, kp: np.ndarray | None, box, min_conf: float = 0.4):
    """Ground point below the visible ankles, or None when no ankle is trusted.

    Ankles are tighter than a box bottom, which grows with shadows, a stretched leg or the
    box detector's padding, and that matters most for small far players.
    """
    if kp is None:
        return None
    ank = kp[[15, 16]]
    ank = ank[ank[:, 2] >= min_conf]
    if not len(ank):
        return None
    p = ank[:, :2].mean(0)
    if p[1] > box[3] + 2 or p[1] < box[1]:
        return None
    # The ankle joint sits ~8 cm above the floor: about 5% of the person's height, straight down.
    u = up_direction(cal, p)
    return p - u * 0.05 * (box[3] - box[1])


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


def ground_covariance(cal: Calibration, foot_px, sigma_px: float) -> np.ndarray:
    """Court-position covariance of a foot point known to +-sigma_px pixels.

    One pixel is a few centimeters near a high camera but can be most of a meter on the far
    side of a low one, so the tracker must trust far measurements much less.
    """
    f = np.asarray(foot_px, dtype=np.float64)
    pts = cal.to_court([f, f + (1.0, 0.0), f + (0.0, 1.0)])
    J = np.stack([pts[1] - pts[0], pts[2] - pts[0]], axis=1)    # meters per pixel
    if not np.isfinite(J).all():
        return np.eye(2) * 25.0
    return sigma_px ** 2 * J @ J.T + np.eye(2) * 0.05 ** 2


def lift_keypoints(cal: Calibration, keypoints: np.ndarray, ground_xy, height_m: float = 1.75) -> np.ndarray:
    """3D joints (court meters, z up) from 2D pose keypoints of a person standing at ground_xy.

    One camera cannot see depth across the body, so the 2D pose is stood upright in the plane
    through the player that faces the camera, scaled by how tall a person looks at that spot.
    Good enough for a replay figure; limbs pointing at the camera come out flattened.
    """
    g = np.asarray(ground_xy, dtype=np.float64)
    foot = cal.to_image([g])[0]
    up = up_direction(cal, foot)
    side = np.array([-up[1], up[0]])
    m_per_px = height_m / max(expected_height_px(cal, g, height_m), 1e-6)
    C = np.asarray(cal.pose.position, dtype=np.float64)
    toward = C[:2] - g
    toward /= max(np.linalg.norm(toward), 1e-9)
    lateral = np.array([toward[1], -toward[0]])     # image-right direction on the ground
    if (cal.to_image([g + lateral])[0] - foot) @ side < 0:
        lateral = -lateral
    d = keypoints[:, :2] - foot
    z = np.clip((d @ up) * m_per_px, 0.0, 2.4)
    xy = g + np.outer((d @ side) * m_per_px, lateral)
    return np.c_[xy, z]


def ground_from_head(cal: Calibration, head_px, head_m: float = 1.65):
    """Ground position of a person whose feet are out of frame, from where their head is."""
    pose = cal.pose
    Hh = pose.K @ np.stack([pose.R[:, 0], pose.R[:, 1], head_m * pose.R[:, 2] + pose.t], axis=1)
    p = np.linalg.solve(Hh, np.array([head_px[0], head_px[1], 1.0]))
    return p[:2] / p[2] if abs(p[2]) > 1e-12 else None


def expected_height_px(cal: Calibration, court_xy, height_m: float = 1.75) -> float:
    a = cal.pose.project3d([(court_xy[0], court_xy[1], 0.0)])[0]
    b = cal.pose.project3d([(court_xy[0], court_xy[1], height_m)])[0]
    return float(np.linalg.norm(b - a))


# ---------------------------------------------------------------- detectors

def _nms(dets: list[Detection], iou: float = 0.5) -> list[Detection]:
    keep: list[Detection] = []
    for d in sorted(dets, key=lambda d: -d.conf):
        x1, y1, x2, y2 = d.box
        ok = True
        for k in keep:
            a1, b1, a2, b2 = k.box
            iw, ih = max(0.0, min(x2, a2) - max(x1, a1)), max(0.0, min(y2, b2) - max(y1, b1))
            inter = iw * ih
            small = min((x2 - x1) * (y2 - y1), (a2 - a1) * (b2 - b1))
            union = (x2 - x1) * (y2 - y1) + (a2 - a1) * (b2 - b1) - inter
            if (union > 0 and inter / union > iou) or (small > 0 and inter / small > 0.8):
                ok = False
                break
        if ok:
            keep.append(d)
    return keep


class YoloDetector:
    """Ultralytics YOLO people detector with a zoomed second look at the far court, plus pose.

    Far players can be a few dozen pixels tall. When a calibration is given, the region where
    far-side players can appear is cropped, enlarged and searched again, and the results are
    merged. Pose keypoints come from running a pose model on each person's enlarged crop,
    which finds far players' joints far more reliably than whole-frame pose detection.
    Weights download on first use.
    """

    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.25, cal: Calibration | None = None,
                 pose_weights: str | None = "yolov8n-pose.pt"):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.pose = YOLO(pose_weights) if pose_weights else None
        self.conf = conf
        self.cal = cal
        self._roi = None

    def _far_roi(self, w: int, h: int):
        if self._roi is None and self.cal is not None:
            xs, ys = np.meshgrid([-1.5, court.WIDTH / 2, court.WIDTH + 1.5], [court.NET_Y, court.LENGTH + 2.5])
            g = np.stack([xs.ravel(), ys.ravel()], 1)
            pts = np.concatenate([self.cal.pose.project3d(np.c_[g, np.full(len(g), z)]) for z in (0.0, 2.2)])
            if np.isfinite(pts).all():
                x1, y1 = np.clip(pts.min(0) - 10, 0, (w, h)).astype(int)
                x2, y2 = np.clip(pts.max(0) + 10, 0, (w, h)).astype(int)
                if x2 - x1 > 32 and y2 - y1 > 16 and (x2 - x1) * (y2 - y1) < 0.35 * w * h:
                    self._roi = (x1, y1, x2, y2)
            self._roi = self._roi or ()
        return self._roi

    def _detect(self, img, imgsz, offset=(0, 0), scale=1.0) -> list[Detection]:
        r = self.model.predict(img, classes=[0], conf=self.conf, imgsz=imgsz, verbose=False)[0]
        ox, oy = offset
        return [Detection((ox + x1 / scale, oy + y1 / scale, ox + x2 / scale, oy + y2 / scale), float(c))
                for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        imgsz = int(min(1280, -(-max(h, w) // 32) * 32))   # never upsample the whole frame
        dets = self._detect(frame, imgsz)
        roi = self._far_roi(w, h)
        if roi:
            x1, y1, x2, y2 = roi
            scale = min(3.0, 640.0 / (x2 - x1))
            if scale > 1.2:
                crop = cv2.resize(frame[y1:y2, x1:x2], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                # People cut by the crop's edge are half-boxes of someone the full frame already has.
                inner = [d for d in self._detect(crop, 640, (x1, y1), scale)
                         if (d.box[0] > x1 + 2 or x1 == 0) and (d.box[2] < x2 - 2 or x2 == w)
                         and (d.box[3] < y2 - 2 or y2 == h) and (d.box[1] > y1 + 2 or y1 == 0)]
                dets = _nms(dets + inner)
        if self.pose is not None and dets:
            self._add_pose(frame, dets)
        for d in dets:
            d.feat = appearance(frame, d)
        return dets

    def _add_pose(self, frame: np.ndarray, dets: list[Detection], target_h: int = 256):
        h, w = frame.shape[:2]
        crops, meta = [], []
        for d in dets:
            x1, y1, x2, y2 = d.box
            bw, bh = x2 - x1, y2 - y1
            cx1, cy1 = int(max(0, x1 - 0.25 * bw)), int(max(0, y1 - 0.1 * bh))
            cx2, cy2 = int(min(w, x2 + 0.25 * bw)), int(min(h, y2 + 0.1 * bh))
            if cx2 - cx1 < 4 or cy2 - cy1 < 8:
                continue
            s = max(1.0, target_h / (cy2 - cy1))
            crops.append(cv2.resize(frame[cy1:cy2, cx1:cx2], None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC))
            meta.append((d, cx1, cy1, s))
        if not crops:
            return
        for (d, cx1, cy1, s), r in zip(meta, self.pose.predict(crops, conf=0.2, imgsz=320, verbose=False)):
            if r.keypoints is None or len(r.boxes) == 0:
                continue
            # The person this box is about is the one whose box best matches the crop's center.
            b = r.boxes.xyxy.cpu().numpy() / s + (cx1, cy1, cx1, cy1)
            c = np.array([(d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2])
            i = int(np.argmin(np.linalg.norm((b[:, :2] + b[:, 2:]) / 2 - c, axis=1)))
            kp = r.keypoints.data[i].cpu().numpy().astype(np.float64)
            kp[:, :2] = kp[:, :2] / s + (cx1, cy1)
            d.keypoints = kp


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
    """Attach foot points, court positions and their uncertainty; drop people off court or of implausible size."""
    kept = []
    for d in dets:
        if d.court_xy is not None:  # already located on the ground (occupancy detector)
            if d.cov is None:
                d.cov = np.eye(2) * 0.1 ** 2
            kept.append(d)
            continue
        if d.mask is not None:
            d.foot_px = foot_from_blob(cal, d.mask)
        else:
            d.foot_px = foot_from_keypoints(cal, d.keypoints, d.box)
            if d.foot_px is None:
                d.foot_px = foot_from_box(cal, d.box)
        d.court_xy = cal.to_court([d.foot_px])[0]
        cut = d.mask is None and d.box[3] >= cal.image_size[1] - 3 and (
            d.keypoints is None or (d.keypoints[[15, 16], 2] < 0.4).all())
        if cut:  # feet below the frame edge: the box bottom is not the feet, the head still is the head
            kp = d.keypoints
            head = kp[0, :2] if kp is not None and kp[0, 2] >= 0.4 else ((d.box[0] + d.box[2]) / 2, d.box[1])
            g = ground_from_head(cal, head, 1.6 if kp is not None and kp[0, 2] >= 0.4 else 1.72)
            if g is not None and np.isfinite(g).all():
                d.court_xy = g
                d.foot_px = cal.to_image([g])[0]
        if not np.isfinite(d.court_xy).all():
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
        # Feet are found to within a few percent of the person's image height.
        d.cov = ground_covariance(cal, d.foot_px, max(1.5, 0.03 * extent))
        if not cut and d.mask is None and d.box[1] > 2 and exp_h > 0:
            _fuse_height(cal, d, extent, exp_h)
        x, y = d.court_xy
        if not (-margin_m <= x <= court.WIDTH + margin_m and -margin_m <= y <= court.LENGTH + margin_m):
            continue
        kept.append(d)
    return kept


def _fuse_height(cal: Calibration, d: Detection, height_px: float, expected_px: float, rel_sd: float = 0.08):
    """Blend the foot-point distance with the distance implied by how tall the person looks.

    From a low camera the far players' feet sit where one pixel spans most of a meter, while
    their apparent height still says roughly how far away they are (people are 1.75 m give
    or take ~8%). Both estimates are combined along the camera's line of sight, weighted by
    their uncertainties, so near players keep their precise foot point.
    """
    C = np.asarray(cal.pose.position[:2], dtype=np.float64)
    g = d.court_xy
    r = float(np.linalg.norm(g - C))
    if r < 1e-6:
        return
    u = (g - C) / r
    var_foot = float(u @ d.cov @ u)
    r_h = r * expected_px * (1.7 / 1.75) / height_px   # apparent size falls off with distance
    var_h = (rel_sd * r_h) ** 2
    w = var_h / (var_h + var_foot)
    r_new = w * r + (1 - w) * r_h
    d.court_xy = C + u * r_new
    var_new = var_foot * var_h / (var_foot + var_h)
    n = np.array([-u[1], u[0]])
    d.cov = var_new * np.outer(u, u) + float(n @ d.cov @ n) * np.outer(n, n)


# ---------------------------------------------------------------- tracking

@dataclass
class Track:
    id: int
    x: np.ndarray                      # state: x, y, vx, vy
    P: np.ndarray
    side: str = "near"                 # side of the net the track started on
    det: Detection | None = None       # detection matched in the latest frame, if any
    feat: np.ndarray | None = None     # running appearance
    hits: int = 1
    misses: int = 0
    history: list = field(default_factory=list)

    @property
    def pos(self):
        return self.x[:2]

    @property
    def team(self) -> str:
        return self.side


def _side(xy) -> str:
    return "near" if xy[1] < court.NET_Y else "far"


class CourtTracker:
    """Constant-velocity Kalman tracks on the court plane with Hungarian assignment.

    Each detection carries its own position covariance, so a far player seen by a low camera
    (where one pixel is most of a meter) is smoothed hard and gated loosely, while a near
    player is followed closely. Doubles has two players per side, so a side never holds more
    than two tracks: a detection that cannot be matched re-acquires that side's lost player
    instead of starting a new identity.
    """

    def __init__(self, fps: float, max_players: int = 4, gate_m: float = 1.5, max_misses: int | None = None,
                 per_side: int | None = None):
        self.dt = 1.0 / fps
        self.max_players = max_players
        self.per_side = per_side or max(1, max_players // 2)
        self.gate = gate_m
        self.max_misses = max_misses or int(2 * fps)
        self.coast = max(1, int(0.5 * fps))
        self.min_hits = max(3, int(fps))
        self.tracks: list[Track] = []
        self.next_id = 1
        dt = self.dt
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1.0]])
        self.Q = np.diag([0.01, 0.01, 4.0, 4.0]) * dt   # players accelerate hard
        self.Hm = np.array([[1, 0, 0, 0], [0, 1, 0, 0.0]])
        self.R = np.eye(2) * 0.05 ** 2

    def _R(self, d: Detection) -> np.ndarray:
        return d.cov if d.cov is not None else self.R

    def _cost(self, t: Track, d: Detection) -> float:
        """Mahalanobis distance, floored so that tiny covariances still allow normal motion."""
        if _side(d.court_xy) != t.side:
            return np.inf   # players stay on their side of the net during a rally
        S = self.Hm @ t.P @ self.Hm.T + self._R(d) + np.eye(2) * (self.gate / 3) ** 2
        r = d.court_xy - t.pos
        # Looking different costs as much as being three standard deviations away.
        return float(np.sqrt(r @ np.linalg.solve(S, r))) + 6.0 * appearance_distance(t.feat, d.feat)

    def update(self, dets: list[Detection], frame_idx: int) -> list[Track]:
        for t in self.tracks:
            t.x = self.F @ t.x
            t.P = self.F @ t.P @ self.F.T + self.Q
            if t.misses > self.coast:   # long unseen: stop growing, never more uncertain than ~3 m
                t.P = np.diag([min(t.P[0, 0], 9.0), min(t.P[1, 1], 9.0), 1.0, 1.0])
            t.det = None
        assigned_t, assigned_d = set(), set()
        if self.tracks and dets:
            C = np.array([[self._cost(t, d) for d in dets] for t in self.tracks])
            ri, ci = linear_sum_assignment(np.minimum(C, 1e6))
            for r, c in zip(ri, ci):
                if C[r, c] <= 3.5 + 0.3 * self.tracks[r].misses:   # ~99.8% gate, widening while unseen
                    self._correct(self.tracks[r], dets[c])
                    assigned_t.add(r)
                    assigned_d.add(c)
        for i, t in enumerate(self.tracks):
            if i not in assigned_t:
                t.misses += 1
        # Established players are kept while unseen (out of frame, behind someone) so they can be
        # re-acquired with their identity; only short-lived tracks (ghosts) are dropped.
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses or t.hits >= self.min_hits]
        free = sorted((c for c in range(len(dets)) if c not in assigned_d), key=lambda c: -dets[c].conf)
        for c in free:
            d = dets[c]
            side = _side(d.court_xy)
            if any(np.linalg.norm(t.pos - d.court_xy) < 0.6 for t in self.tracks):
                continue
            mine = [t for t in self.tracks if t.side == side]
            if len(mine) >= self.per_side or len(self.tracks) >= self.max_players:
                lost = [t for t in mine if t.misses >= 2]
                limit = 0.5 if len(lost) > 1 else 0.7   # one candidate: only rule out a clear mismatch
                lost = [t for t in lost if appearance_distance(t.feat, d.feat) < limit]
                if not lost:
                    continue  # a third person on this side (spectator, coach, ghost), or not this player
                # Re-acquire the lost player who looks most like this person, keeping the identity.
                t = min(lost, key=lambda t: appearance_distance(t.feat, d.feat))
                t.feat = d.feat if t.feat is None or d.feat is None else 0.8 * t.feat + 0.2 * d.feat
                t.x = np.array([*d.court_xy, 0, 0.0])
                t.P = np.diag([*np.diag(self._R(d)), 1, 1.0])
                t.misses = 0
                t.det = d
                continue
            self.tracks.append(Track(self.next_id, np.array([*d.court_xy, 0, 0.0]),
                                     np.diag([*np.diag(self._R(d)), 1, 1.0]), side=side, det=d, feat=d.feat))
            self.next_id += 1
        for t in self.tracks:
            if t.misses:
                t.x[2:] = 0.0   # do not coast on stale velocity
            if t.misses <= self.coast:
                t.history.append((frame_idx, float(t.x[0]), float(t.x[1])))
        # Briefly occluded players are reported at their last known position.
        return [t for t in self.tracks if t.misses <= self.coast]

    def _correct(self, t: Track, d: Detection):
        S = self.Hm @ t.P @ self.Hm.T + self._R(d)
        K = t.P @ self.Hm.T @ np.linalg.inv(S)
        t.x = t.x + K @ (d.court_xy - self.Hm @ t.x)
        t.P = (np.eye(4) - K @ self.Hm) @ t.P
        t.hits += 1
        t.misses = 0
        t.det = d
        if d.feat is not None:
            t.feat = d.feat if t.feat is None else 0.9 * t.feat + 0.1 * d.feat
