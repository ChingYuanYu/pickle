"""Court-line pixel extraction and straight-line detection.

Nothing here assumes a camera angle: lines are found as thin bright structures
of any orientation, then grouped by the vanishing point they share.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Line:
    """Infinite image line a*x + b*y + c = 0 with (a, b) unit length."""
    abc: np.ndarray
    support: float          # total length (px) of the segments that formed it
    mid: np.ndarray         # a point on the observed part of the line

    @property
    def direction(self) -> np.ndarray:
        return np.array([-self.abc[1], self.abc[0]])


def line_mask(img: np.ndarray, max_line_px: int | None = None, max_saturation: float | None = 0.25) -> np.ndarray:
    """Binary mask of thin bright structures (painted lines), any orientation.

    max_saturation (chroma change per unit of brightness gain) keeps only near-white lines, which rejects the colored lines of
    other sports painted on shared courts. Pass None for courts with colored lines.
    """
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    # Painted lines are bright and low-saturation; boost that, damp saturated colors.
    sat = np.abs(lab[:, :, 1].astype(np.int16) - 128) + np.abs(lab[:, :, 2].astype(np.int16) - 128)
    L = np.clip(L.astype(np.int16) - (sat // 2), 0, 255).astype(np.uint8)
    max_line_px = max_line_px or max(9, int(0.025 * max(h, w)) | 1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max_line_px, max_line_px))
    tophat = cv2.morphologyEx(L, cv2.MORPH_TOPHAT, k)
    thr = max(18.0, float(np.percentile(tophat, 96)))
    mask = (tophat > thr).astype(np.uint8) * 255
    if max_saturation is not None:
        # A white line blended into the surface only shrinks the surface's chroma toward
        # neutral. Anything else (yellow tennis lines, colored signage) leaves a residual.
        ab = lab[:, :, 1:].astype(np.float32) - 128.0
        bg = np.dstack([cv2.medianBlur(lab[:, :, c], max_line_px | 1) for c in (1, 2)]).astype(np.float32) - 128.0
        bn = (bg ** 2).sum(-1, keepdims=True)
        s = np.clip((ab * bg).sum(-1, keepdims=True) / np.maximum(bn, 1.0), 0.0, 1.0)
        resid = np.linalg.norm(ab - s * bg, axis=-1)
        # Blending scales the residual and the brightness gain equally, so their ratio
        # separates white from colored lines even when a far line is only a pixel wide.
        mask[resid > max_saturation * np.maximum(tophat.astype(np.float32), 1.0)] = 0
    # Drop thick bright blobs (shirts, shoes, signage): they survive an opening that removes lines.
    blob_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max_line_px // 2 | 1, max_line_px // 2 | 1))
    blobs = cv2.morphologyEx(mask, cv2.MORPH_OPEN, blob_k)
    blobs = cv2.dilate(blobs, blob_k)
    mask[blobs > 0] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return mask


def _seg_to_abc(x1, y1, x2, y2) -> np.ndarray:
    a, b = y2 - y1, x1 - x2
    n = np.hypot(a, b)
    a, b = a / n, b / n
    return np.array([a, b, -(a * x1 + b * y1)])


def detect_lines(mask: np.ndarray, max_lines: int = 18) -> list[Line]:
    h, w = mask.shape[:2]
    diag = float(np.hypot(h, w))
    segs = cv2.HoughLinesP(mask, 1, np.pi / 720, threshold=max(25, int(0.02 * diag)),
                           minLineLength=0.03 * diag, maxLineGap=0.012 * diag)
    if segs is None:
        return []
    segs = np.asarray(segs).reshape(-1, 4).astype(np.float64)
    lengths = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    order = np.argsort(-lengths)
    base_tol = max(3.0, 0.004 * diag)
    # Local painted-line thickness, so both edges of a wide near-camera line merge into one.
    inside = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    clusters: list[dict] = []
    for idx in order:
        x1, y1, x2, y2 = segs[idx]
        abc = _seg_to_abc(x1, y1, x2, y2)
        sp = np.array(_sample(x1, y1, x2, y2, step=10.0)).astype(int)
        sp[:, 0] = np.clip(sp[:, 0], 0, w - 1)
        sp[:, 1] = np.clip(sp[:, 1], 0, h - 1)
        thick = 2.0 * float(np.percentile(inside[sp[:, 1], sp[:, 0]], 90))
        tol = max(base_tol, 1.2 * thick + 2.0)
        placed = False
        for c in clusters:
            cos = abs(c["abc"][0] * abc[0] + c["abc"][1] * abc[1])
            if cos < np.cos(np.radians(2.0)):
                continue
            d1 = abs(c["abc"] @ (x1, y1, 1.0))
            d2 = abs(c["abc"] @ (x2, y2, 1.0))
            if d1 < tol and d2 < tol:
                c["pts"].extend(_sample(x1, y1, x2, y2))
                c["support"] += lengths[idx]
                c["abc"] = _fit(c["pts"])
                placed = True
                break
        if not placed:
            clusters.append({"abc": abc, "pts": _sample(x1, y1, x2, y2), "support": lengths[idx]})
    clusters.sort(key=lambda c: -c["support"])
    lines = [Line(c["abc"], float(c["support"]), np.mean(np.array(c["pts"]), axis=0)) for c in clusters]
    return lines[:max_lines]


def _sample(x1, y1, x2, y2, step=6.0):
    n = max(2, int(np.hypot(x2 - x1, y2 - y1) / step) + 1)
    t = np.linspace(0, 1, n)
    return list(zip(x1 + (x2 - x1) * t, y1 + (y2 - y1) * t))


def _fit(pts) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float32)
    vx, vy, x0, y0 = cv2.fitLine(p, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    a, b = vy, -vx
    return np.array([a, b, -(a * x0 + b * y0)], dtype=np.float64)


def group_by_vanishing_point(lines: list[Line], max_families: int = 3, angle_tol_deg: float = 1.5):
    """Split lines into families that share a vanishing point (finite or at infinity).

    Returns a list of (vanishing_point_homogeneous, [line indices]) sorted by support.
    """
    if len(lines) < 2:
        return []
    L = np.array([l.abc for l in lines])
    mids = np.array([l.mid for l in lines])
    dirs = np.array([l.direction for l in lines])
    sup = np.array([l.support for l in lines])
    remaining = np.ones(len(lines), bool)
    families = []
    cos_tol = np.cos(np.radians(angle_tol_deg))
    for _ in range(max_families):
        idx = np.flatnonzero(remaining)
        if len(idx) < 2:
            break
        best = None
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                v = np.cross(L[idx[a]], L[idx[b]])
                inl = _vp_inliers(v, mids, dirs, cos_tol) & remaining
                s = sup[inl].sum()
                if inl.sum() >= 2 and (best is None or s > best[0]):
                    best = (s, v, inl)
        if best is None:
            break
        _, v, inl = best
        for _ in range(3):  # re-estimate the vanishing point from all inliers, then re-collect
            Wl = L[inl] * np.sqrt(sup[inl])[:, None]
            v = np.linalg.svd(Wl)[2][-1]
            new = _vp_inliers(v, mids, dirs, cos_tol) & remaining
            if new.sum() < 2:
                break
            inl = new
        families.append(v)
        remaining &= ~inl
    # Final membership may overlap: a line near two vanishing points is tried in both families.
    loose = np.cos(np.radians(2.5 * angle_tol_deg))
    return [(v, list(np.flatnonzero(_vp_inliers(v, mids, dirs, loose)))) for v in families]


def _vp_inliers(v, mids, dirs, cos_tol):
    d = np.stack([v[0] - v[2] * mids[:, 0], v[1] - v[2] * mids[:, 1]], axis=1)
    n = np.linalg.norm(d, axis=1)
    ok = n > 1e-9
    cos = np.zeros(len(mids))
    cos[ok] = np.abs((d[ok] * dirs[ok]).sum(1)) / n[ok]
    return cos >= cos_tol
