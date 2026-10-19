"""Automatic court calibration from any camera angle.

1. Extract thin bright line pixels and fit infinite lines (lines.py).
2. Group lines by shared vanishing point, so "across" and "along" families are found
   from geometry, not from assuming they look horizontal or vertical.
3. Enumerate hypotheses: two image lines from one family matched to two model lines of
   constant y, two from another family matched to two model lines of constant x. Each
   hypothesis gives a homography; it is scored by how much painted court it explains.
4. Refine the winner by re-fitting every model line to nearby line pixels.

A manual fallback (calibrate_from_points) covers footage where the lines are not visible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import cv2
import numpy as np

from . import court
from .camera import CameraPose, best_focal, consistency_weight, focal_residual_curve, pose_from_homography, project, project_w
from .lines import Line, detect_lines, group_by_vanishing_point, line_mask


@dataclass
class Calibration:
    H: np.ndarray                      # court meters -> image pixels
    image_size: tuple[int, int]
    score: float                       # fraction of visible painted lines confirmed by pixels
    pose: CameraPose
    method: str = "auto"
    notes: list[str] = field(default_factory=list)

    @property
    def H_inv(self) -> np.ndarray:
        return np.linalg.inv(self.H)

    def to_court(self, pts) -> np.ndarray:
        return project(self.H_inv, pts)

    def to_image(self, pts) -> np.ndarray:
        return project(self.H, pts)

    @property
    def ok(self) -> bool:
        return self.score >= 0.45

    def summary(self) -> dict:
        return {
            "method": self.method,
            "score": round(self.score, 3),
            "ok": self.ok,
            "H": np.round(self.H, 6).tolist(),
            "camera": self.pose.summary(),
            "notes": self.notes,
        }


# ---------------------------------------------------------------- hypothesis search

def _square_to_quad(q: np.ndarray) -> np.ndarray:
    """Batched Heckbert square->quad. q: (N, 4, 2) for unit-square corners (0,0),(1,0),(1,1),(0,1)."""
    x0, y0 = q[:, 0, 0], q[:, 0, 1]
    x1, y1 = q[:, 1, 0], q[:, 1, 1]
    x2, y2 = q[:, 2, 0], q[:, 2, 1]
    x3, y3 = q[:, 3, 0], q[:, 3, 1]
    sx, sy = x0 - x1 + x2 - x3, y0 - y1 + y2 - y3
    dx1, dx2, dy1, dy2 = x1 - x2, x3 - x2, y1 - y2, y3 - y2
    den = dx1 * dy2 - dx2 * dy1
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (sx * dy2 - dx2 * sy) / den
        h = (dx1 * sy - sx * dy1) / den
    H = np.zeros((len(q), 3, 3))
    H[:, 0] = np.stack([x1 - x0 + g * x1, x3 - x0 + h * x3, x0], 1)
    H[:, 1] = np.stack([y1 - y0 + g * y1, y3 - y0 + h * y3, y0], 1)
    H[:, 2] = np.stack([g, h, np.ones_like(g)], 1)
    return H


def _intersect(l1: np.ndarray, l2: np.ndarray) -> np.ndarray:
    p = np.cross(l1, l2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return p[..., :2] / p[..., 2:3]


class _Scorer:
    """Scores homographies by painted-line pixels they explain, weighted by image length."""

    def __init__(self, mask: np.ndarray, step: float):
        self.h, self.w = mask.shape[:2]
        diag = np.hypot(self.h, self.w)
        self.tol = max(2.5, 0.004 * diag)
        self.cap = 0.02 * diag
        self.dist = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
        pts, same_next = [], []
        for (x1, y1), (x2, y2) in court.SEGMENTS:
            n = max(2, int(np.hypot(x2 - x1, y2 - y1) / step) + 1)
            t = np.linspace(0, 1, n)
            pts.append(np.stack([x1 + (x2 - x1) * t, y1 + (y2 - y1) * t], 1))
            sn = np.ones(n, bool)
            sn[-1] = False
            same_next.append(sn)
        self.pts = np.concatenate(pts)
        self.same_next = np.concatenate(same_next)
        self.hom = np.hstack([self.pts, np.ones((len(self.pts), 1))])

    def __call__(self, Hs: np.ndarray, return_parts: bool = False):
        P = np.einsum("nij,kj->nki", Hs, self.hom)        # (N, K, 3)
        w = P[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            xy = P[..., :2] / w[..., None]
        nxt = np.roll(xy, -1, axis=1)
        spacing = np.linalg.norm(nxt - xy, axis=2)
        prev = np.roll(spacing, 1, axis=1)
        spacing = np.where(self.same_next[None], spacing, prev)
        spacing = np.nan_to_num(np.minimum(spacing, self.cap), nan=0.0)
        x, y = xy[..., 0], xy[..., 1]
        vis = (w > 0) & (x >= 0) & (x < self.w - 1) & (y >= 0) & (y < self.h - 1)
        xi = np.clip(np.nan_to_num(x), 0, self.w - 1).astype(np.int32)
        yi = np.clip(np.nan_to_num(y), 0, self.h - 1).astype(np.int32)
        d = self.dist[yi, xi]
        hit = vis & (d < self.tol)
        # Each image location may be explained once: stops hypotheses that fold several
        # model lines onto one strong image line (e.g. the whole court squeezed onto the net tape).
        q = max(3, int(self.tol))
        key = (yi // q).astype(np.int64) * (self.w // q + 1) + xi // q
        key = np.where(hit, key, -1 - np.arange(key.shape[1])[None])
        order = np.argsort(key, axis=1, kind="stable")
        ks = np.take_along_axis(key, order, 1)
        dup_sorted = np.zeros_like(ks, dtype=bool)
        dup_sorted[:, 1:] = ks[:, 1:] == ks[:, :-1]
        dup = np.zeros_like(dup_sorted)
        np.put_along_axis(dup, order, dup_sorted, 1)
        explained = (spacing * (hit & ~dup)).sum(1)
        missing = (spacing * (vis & ~hit)).sum(1)
        score = explained - 0.5 * missing
        if return_parts:
            return score, explained, missing
        return score


def _valid(Hs: np.ndarray, size) -> np.ndarray:
    """Reject degenerate, mirrored and behind-camera hypotheses."""
    w_img, h_img = size
    ok = np.isfinite(Hs).all(axis=(1, 2))
    Hs = np.where(ok[:, None, None], Hs, np.eye(3))
    c = np.hstack([court.CORNERS, np.ones((4, 1))])
    P = np.einsum("nij,kj->nki", Hs, c)
    w = P[..., 2]
    same_side = (np.sign(w) == np.sign(w[:, :1])).all(1)
    Hs_pos = Hs * np.sign(w[:, :1])[:, :, None]
    P = np.einsum("nij,kj->nki", Hs_pos, c)
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = P[..., :2] / P[..., 2:3]
    # Shoelace area; a real camera above the court never mirrors it.
    x, y = xy[..., 0], xy[..., 1]
    area = 0.5 * (x * np.roll(y, -1, 1) - np.roll(x, -1, 1) * y).sum(1)
    diag = np.hypot(w_img, h_img)
    not_mirrored = area < 0  # court x-right/y-far seen from above flips sign in image coords (y down)
    big_enough = np.abs(area) > 0.005 * w_img * h_img
    bounded = (np.abs(np.nan_to_num(xy, nan=1e12)) < 30 * diag).all(axis=(1, 2))
    # Convex quad: all cross products of consecutive edges share a sign.
    e = np.roll(xy, -1, 1) - xy
    cross = e[..., 0] * np.roll(e[..., 1], -1, 1) - e[..., 1] * np.roll(e[..., 0], -1, 1)
    convex = (np.sign(cross) == np.sign(cross[:, :1])).all(1)
    return ok & same_side & not_mirrored & big_enough & bounded & convex, Hs_pos


def _hypotheses(lines: list[Line], fam_a: list[int], fam_b: list[int]) -> np.ndarray:
    """All homographies with fam_a lines as constant-y court lines and fam_b as constant-x."""
    L = np.array([l.abc for l in lines])
    across = [(i, j, ya, yb) for i, j in combinations(fam_a, 2)
              for ya, yb in combinations(court.ACROSS_Y, 2) for (i, j) in ((i, j), (j, i))]
    along = [(k, m, xa, xb) for k, m in combinations(fam_b, 2)
             for xa, xb in combinations(court.ALONG_X, 2) for (k, m) in ((k, m), (m, k))]
    if not across or not along:
        return np.zeros((0, 3, 3))
    A = np.array(across)
    B = np.array(along)
    ia, ib = np.meshgrid(np.arange(len(A)), np.arange(len(B)), indexing="ij")
    A, B = A[ia.ravel()], B[ib.ravel()]
    la1, la2 = L[A[:, 0].astype(int)], L[A[:, 1].astype(int)]
    lb1, lb2 = L[B[:, 0].astype(int)], L[B[:, 1].astype(int)]
    q = np.stack([_intersect(lb1, la1), _intersect(lb2, la1), _intersect(lb2, la2), _intersect(lb1, la2)], 1)
    Hq = _square_to_quad(q)
    ya, yb, xa, xb = A[:, 2], A[:, 3], B[:, 2], B[:, 3]
    Sinv = np.zeros((len(A), 3, 3))
    Sinv[:, 0, 0] = 1 / (xb - xa)
    Sinv[:, 0, 2] = -xa / (xb - xa)
    Sinv[:, 1, 1] = 1 / (yb - ya)
    Sinv[:, 1, 2] = -ya / (yb - ya)
    Sinv[:, 2, 2] = 1
    return Hq @ Sinv


def _search(lines: list[Line], families, mask: np.ndarray, top_k: int = 40, keep: int = 12):
    """Cheap batched scoring of every hypothesis; returns the `keep` most promising homographies."""
    size = (mask.shape[1], mask.shape[0])
    coarse = _Scorer(mask, step=0.5)
    pool = []
    fams = [f for _, f in families if len(f) >= 2]
    for fa in range(len(fams)):
        for fb in range(len(fams)):
            if fa == fb:
                continue
            Hs = _hypotheses(lines, fams[fa], fams[fb])
            for chunk in range(0, len(Hs), 20000):
                Hc = Hs[chunk:chunk + 20000]
                ok, Hc = _valid(Hc, size)
                Hc = Hc[ok]
                if len(Hc) == 0:
                    continue
                s = coarse(Hc)
                s = np.where(s > 0, s * consistency_weight(Hc, size), s)
                k = min(top_k, len(s))
                best = np.argpartition(-s, k - 1)[:k]
                pool.extend((float(s[i]), Hc[i]) for i in best)
    if not pool:
        return []
    pool.sort(key=lambda t: -t[0])
    fine = _Scorer(mask, step=0.1)
    cands = np.array([H for _, H in pool[:top_k * 3]])
    s = fine(cands)
    s = np.where(s > 0, s * consistency_weight(cands, size), s)
    order = np.argsort(-s)
    out = []
    for i in order:  # drop near-duplicates so the shortlist covers distinct interpretations
        c = project(cands[i], court.CORNERS)
        if all(np.abs(c - project(o, court.CORNERS)).max() > 4 for o in out):
            out.append(cands[i])
        if len(out) == keep:
            break
    return out


def _net_geometry(pose: CameraPose):
    """Image polygon of the net and the 2D line of its top tape, from the recovered camera."""
    x0, x1, y = -0.3, court.WIDTH + 0.3, court.NET_Y
    quad = np.array([(x0, y, 0), (x1, y, 0), (x1, y, court.NET_HEIGHT_POST), (x0, y, court.NET_HEIGHT_POST)])
    X = np.hstack([quad, np.ones((4, 1))]) @ pose.P().T
    if (X[:, 2] <= 0).any():
        return None, None
    poly = X[:, :2] / X[:, 2:3]
    tape = np.array([(x, y, court.NET_HEIGHT_CENTER + (court.NET_HEIGHT_POST - court.NET_HEIGHT_CENTER)
                      * abs(x - court.CENTER_X) / court.CENTER_X) for x in np.linspace(x0, x1, 25)])
    return poly, pose.project3d(tape)


def _evaluate(H: np.ndarray, mask: np.ndarray) -> float:
    """Full score of one hypothesis: explained line length, minus painted lines that should be
    visible but are not (ignoring what the net hides), minus unexplained line pixels inside the court."""
    h, w = mask.shape[:2]
    if not np.isfinite(H).all():
        return -np.inf
    scorer = _Scorer(mask, step=0.05)
    pose = pose_from_homography(H, (w, h))
    if pose.position[2] <= 0.2:
        return -np.inf
    poly, tape = _net_geometry(pose)
    occluded = np.zeros((h, w), np.uint8)
    if poly is not None and np.isfinite(poly).all() and np.abs(poly).max() < 1e6:
        cv2.fillPoly(occluded, [poly.astype(np.int32)], 255)
    P = np.hstack([scorer.pts, np.ones((len(scorer.pts), 1))]) @ H.T
    xy = P[:, :2] / P[:, 2:3]
    xi = np.clip(xy[:, 0], 0, w - 1).astype(int)
    yi = np.clip(xy[:, 1], 0, h - 1).astype(int)
    behind_net = occluded[yi, xi] > 0
    _, explained, _ = scorer(H[None], return_parts=True)
    vis = (P[:, 2] > 0) & (xy[:, 0] >= 0) & (xy[:, 0] < w - 1) & (xy[:, 1] >= 0) & (xy[:, 1] < h - 1)
    spacing = np.linalg.norm(np.diff(xy, axis=0, append=xy[-1:]), axis=1)
    spacing = np.where(scorer.same_next, spacing, np.roll(spacing, 1))
    spacing = np.minimum(np.nan_to_num(spacing), scorer.cap)
    miss = vis & ~behind_net & (scorer.dist[yi, xi] >= scorer.tol)
    missing = float((spacing * miss).sum())
    # Unexplained painted pixels inside the hypothesized court (e.g. a neighbouring court's lines).
    model = np.zeros((h, w), np.uint8)
    thick = int(max(3, 2 * scorer.tol))
    for (x1, y1), (x2, y2) in court.SEGMENTS:
        seg, wv = project_w(H, np.linspace((x1, y1), (x2, y2), 30))
        if (wv > 0).all():
            cv2.polylines(model, [seg.astype(np.int32)], False, 255, thick)
    if tape is not None and np.isfinite(tape).all() and np.abs(tape).max() < 1e6:
        cv2.polylines(model, [tape.astype(np.int32)], False, 255, thick)
    inside = np.zeros((h, w), np.uint8)
    corners, wv = project_w(H, court.CORNERS)
    if (wv > 0).all():
        cv2.fillPoly(inside, [corners.astype(np.int32)], 255)
    stray = (mask > 0) & (inside > 0) & (model == 0) & (occluded == 0)
    line_px = max(1.0, float(np.median(2 * cv2.distanceTransform(mask, cv2.DIST_L2, 3)[mask > 0])))
    unexplained = stray.sum() / line_px
    tape_hit, tape_miss = _net_tape_evidence(H, scorer.dist, scorer.tol)
    raw = float(explained[0]) + tape_hit - 0.5 * (missing + tape_miss) - 3.0 * unexplained
    # A real camera must explain the match: penalize hypotheses whose two focal estimates disagree.
    return raw * float(consistency_weight(H, (w, h))) if raw > 0 else raw


# ---------------------------------------------------------------- refinement

_MODEL_LINES = [("y", v) for v in court.ACROSS_Y] + [("x", v) for v in court.ALONG_X]


def _model_line_extent(kind, v):
    if kind == "y":
        return (0.0, v), (court.WIDTH, v)
    if v == court.CENTER_X:  # centerline is painted only outside the kitchen; fit both pieces
        return (v, 0.0), (v, court.LENGTH)
    return (v, 0.0), (v, court.LENGTH)


def _refine(H: np.ndarray, mask: np.ndarray, iters: int = 4) -> np.ndarray:
    """Re-fit every visible model line to the line pixels near its projection, then re-solve H
    from all line intersections. Pixels on the net tape (located with the recovered camera)
    are ignored, and each line's search band never reaches halfway to its neighbours."""
    h, w = mask.shape[:2]
    diag = np.hypot(h, w)
    ys, xs = np.nonzero(mask)
    mpts = np.stack([xs, ys], 1).astype(np.float64)
    neighbours = {("y", v): [u for u in court.ACROSS_Y + (court.NET_Y,) if u != v] for v in court.ACROSS_Y}
    neighbours.update({("x", v): [u for u in court.ALONG_X if u != v] for v in court.ALONG_X})
    for it in range(iters):
        base_band = max(2.0, 0.012 * diag * (0.6 ** it))
        keep = np.ones(len(mpts), bool)
        pose = pose_from_homography(H, (w, h))
        _, tape = _net_geometry(pose)
        if tape is not None and np.isfinite(tape).all() and np.abs(tape).max() < 1e6:
            tape_mask = np.zeros((h, w), np.uint8)
            cv2.polylines(tape_mask, [tape.astype(np.int32)], False, 255, int(max(4, base_band)))
            keep = tape_mask[ys, xs] == 0
        fitted = {}
        for kind, v in _MODEL_LINES:
            p1, p2 = _model_line_extent(kind, v)
            seg = np.linspace(p1, p2, 60)
            if kind == "x" and v == court.CENTER_X:
                seg = seg[(seg[:, 1] <= court.NVZ_NEAR_Y) | (seg[:, 1] >= court.NVZ_FAR_Y)]
            img, wv = project_w(H, seg)
            vis = (wv > 0) & (img[:, 0] > 0) & (img[:, 0] < w) & (img[:, 1] > 0) & (img[:, 1] < h)
            if vis.sum() < 5:
                continue
            a, b = img[vis][0], img[vis][-1]
            if np.hypot(*(b - a)) < 0.03 * diag:
                continue
            abc = np.cross([*a, 1.0], [*b, 1.0])
            abc /= np.hypot(abc[0], abc[1])
            # Band: at most ~40% of the image gap to the nearest neighbouring model line.
            mid_m = seg[vis][len(seg[vis]) // 2]
            gaps = []
            for u in neighbours[(kind, v)]:
                q = (mid_m[0], u) if kind == "y" else (u, mid_m[1])
                qi, qw = project_w(H, [q])
                if qw[0] > 0:
                    gaps.append(abs(abc @ (*qi[0], 1.0)))
            band = min([base_band] + [0.4 * g for g in gaps])
            band = max(band, 1.5)
            d = np.abs(mpts @ abc[:2] + abc[2])
            t = (mpts - a) @ (b - a) / np.dot(b - a, b - a)
            sel = keep & (d < band) & (t > -0.02) & (t < 1.02)
            if sel.sum() < max(15, 0.02 * diag):
                continue
            vx, vy, x0, y0 = cv2.fitLine(mpts[sel].astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            fitted[(kind, v)] = np.array([vy, -vx, -(vy * x0 - vx * y0)])
        across = [k for k in fitted if k[0] == "y"]
        along = [k for k in fitted if k[0] == "x"]
        if len(across) < 2 or len(along) < 2:
            break
        src, dst = [], []
        for ka in across:
            for kb in along:
                p = np.cross(fitted[ka], fitted[kb])
                if abs(p[2]) < 1e-12:
                    continue
                src.append((kb[1], ka[1]))
                dst.append(p[:2] / p[2])
        if len(src) < 4:
            break
        Hn, _ = cv2.findHomography(np.array(src), np.array(dst), 0)
        if Hn is None:
            break
        H = Hn
    return H / H[2, 2]


# ---------------------------------------------------------------- public API

def _canonical(H: np.ndarray, size) -> np.ndarray:
    """The court looks the same rotated 180 degrees; label the end closer to the camera as y = 0."""
    R180 = np.array([[-1, 0, court.WIDTH], [0, -1, court.LENGTH], [0, 0, 1.0]])
    pose_a = pose_from_homography(H, size)
    # Camera y-position below the court midpoint means the near baseline is already y = 0.
    return H if pose_a.position[1] <= court.NET_Y else H @ R180


def _net_tape_samples(H, size, f):
    pose = pose_from_homography(H, size, f)
    _, tape = _net_geometry(pose)
    if tape is None or not np.isfinite(tape).all() or np.abs(tape).max() > 1e6:
        return None
    return np.concatenate([np.linspace(a, b, 8, endpoint=False) for a, b in zip(tape[:-1], tape[1:])])


def _net_tape_evidence(H, dist, tol, n_focal: int = 40):
    """Image length of the net's white tape that this court hypothesis explains (best over
    focal lengths), and the length it predicts but is not there. The net stands on the
    court's midline, so a hypothesis that gets the court's orientation wrong puts it
    somewhere with no tape."""
    h, w = dist.shape
    best = (0.0, 0.0)
    f0 = best_focal(H, (w, h))  # the tape may refine the focal length, not override the court's
    for f in np.geomspace(0.75 * f0, 1.33 * f0, n_focal):
        pts = _net_tape_samples(H, (w, h), f)
        if pts is None:
            continue
        vis = (pts[:, 0] >= 0) & (pts[:, 0] < w - 1) & (pts[:, 1] >= 0) & (pts[:, 1] < h - 1)
        if vis.sum() < 2:
            continue
        p = pts[vis]
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        seg = np.append(seg, seg[-1] if len(seg) else 0.0)
        on = dist[p[:, 1].astype(int), p[:, 0].astype(int)] < tol
        hit, miss = float((seg * on).sum()), float((seg * ~on).sum())
        if hit - 0.5 * miss > best[0] - 0.5 * best[1]:
            best = (hit, miss)
    return best


def _focal_from_net(H: np.ndarray, mask: np.ndarray) -> float | None:
    """The court plane alone pins down the focal length poorly when the camera looks straight
    down the court. The net tape is a painted-white line at a known height (0.86 to 0.91 m),
    so try focal lengths and keep the one whose predicted tape lands on line pixels."""
    h, w = mask.shape[:2]
    fs, e = focal_residual_curve(H, (w, h))
    i0 = int(np.argmin(e))
    f0 = float(fs[i0])
    # If the court alone already pins the focal length (residual rises fast around its best
    # value, as in side and corner views), the tape may only fine-tune it.
    lo, hi = np.interp([0.75 * f0, 1.33 * f0], fs, e)
    if min(lo, hi) > e[i0] + 0.05:
        candidates = np.geomspace(0.85 * f0, 1.18 * f0, 40)
    else:
        candidates = np.geomspace(0.3 * w, 3.0 * w, 120)
    dist = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
    tol = max(2.5, 0.004 * np.hypot(h, w))
    best, best_hits = None, -1.0
    for f in candidates:
        pose = pose_from_homography(H, (w, h), f)
        _, tape = _net_geometry(pose)
        if tape is None or not np.isfinite(tape).all():
            continue
        pts = np.concatenate([np.linspace(a, b, 8, endpoint=False) for a, b in zip(tape[:-1], tape[1:])])
        vis = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
        if vis.sum() < 20:
            continue
        p = pts[vis].astype(int)
        hits = float((dist[p[:, 1], p[:, 0]] < tol).mean())
        # Ground-only estimate breaks ties: prefer focal lengths close to it.
        hits -= 0.02 * abs(np.log(f / f0))
        if hits > best_hits:
            best, best_hits = f, hits
    if best is None or best_hits < 0.6:
        return None
    return float(best)


def line_coverage(H: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of the visible painted court lines (by image length) that have line pixels under them."""
    s = _Scorer(mask, step=0.05)
    _, explained, missing = s(H[None], return_parts=True)
    tot = explained[0] + missing[0]
    return float(explained[0] / tot) if tot > 0 else 0.0


def calibrate_image(img: np.ndarray, focal_px: float | None = None) -> Calibration | None:
    """Find the court in one image (or a median background frame)."""
    h, w = img.shape[:2]
    mask = line_mask(img)
    lines = detect_lines(mask)
    fams = group_by_vanishing_point(lines)
    if len(fams) < 2:
        return None
    shortlist = _search(lines, fams, mask)
    if not shortlist:
        return None
    options = []
    for Hc in shortlist:  # keep the refined fit only where it actually explains the image better
        options += [Hc, _refine(Hc, mask)]
    scores = [_evaluate(Hc, mask) for Hc in options]
    H = options[int(np.argmax(scores))]
    H = _canonical(H, (w, h))
    if focal_px is None:
        focal_px = _focal_from_net(H, mask)
    pose = pose_from_homography(H, (w, h), focal_px)
    cal = Calibration(H / H[2, 2], (w, h), line_coverage(H, mask), pose)
    if pose.position[2] < 0.3:
        cal.notes.append("camera height under 0.3 m looks implausible; check calibration")
    if pose.elevation_deg < 8:
        cal.notes.append("very low camera angle: far-court positions will be imprecise")
    return cal


def calibrate_from_points(image_pts, keypoint_names, image_size, focal_px: float | None = None) -> Calibration:
    """Manual fallback: at least 4 clicked court keypoints (names from court.KEYPOINTS)."""
    src = np.array([court.KEYPOINTS[n] for n in keypoint_names], dtype=np.float64)
    dst = np.asarray(image_pts, dtype=np.float64)
    H, _ = cv2.findHomography(src, dst, 0)
    pose = pose_from_homography(H, image_size, focal_px)
    return Calibration(H / H[2, 2], tuple(image_size), 1.0, pose, method="manual")


def median_background(frames: list[np.ndarray]) -> np.ndarray:
    """Per-pixel median over frames: removes moving players so lines are unoccluded."""
    return np.median(np.stack(frames), axis=0).astype(np.uint8)
