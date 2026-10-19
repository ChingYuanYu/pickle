"""Camera geometry: homography helpers and camera pose recovered from a court homography.

The court homography H maps court meters (x, y, 1) to image pixels. Writing it as
H ~ K [r1 r2 t] lets us recover the focal length, camera position and viewing angle,
which is what makes the rest of the pipeline independent of where the camera stands.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from . import court


def project(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to Nx2 points."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    p = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return p[:, :2] / p[:, 2:3]


def project_w(H: np.ndarray, pts: np.ndarray):
    """Like project() but also returns the homogeneous w (sign tells front/behind)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    p = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return p[:, :2] / p[:, 2:3], p[:, 2]


def image_to_court(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return project(np.linalg.inv(H), pts)


@dataclass
class CameraPose:
    focal_px: float
    cx: float
    cy: float
    R: np.ndarray            # world -> camera rotation
    t: np.ndarray            # world -> camera translation
    position: np.ndarray     # camera center in court meters (x, y, z)
    azimuth_deg: float       # 0 = straight down the court's long axis, 90 = side-on
    elevation_deg: float     # angle above the ground, seen from court center
    view: str                # baseline / corner / sideline / overhead

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.focal_px, 0, self.cx], [0, self.focal_px, self.cy], [0, 0, 1.0]])

    def P(self) -> np.ndarray:
        return self.K @ np.hstack([self.R, self.t.reshape(3, 1)])

    def project3d(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
        p = np.hstack([X, np.ones((len(X), 1))]) @ self.P().T
        return p[:, :2] / p[:, 2:3]

    def up_vanishing_point(self) -> np.ndarray:
        """Image point where vertical world lines meet (homogeneous)."""
        return self.K @ self.R[:, 2]

    def summary(self) -> dict:
        d = asdict(self)
        for k in ("R", "t", "position"):
            d[k] = np.round(np.asarray(d[k]), 4).tolist()
        d["focal_px"] = round(self.focal_px, 1)
        d["height_m"] = round(float(self.position[2]), 2)
        return d


def _focal_from_h(A: np.ndarray, width: float) -> float:
    a1, a2 = A[:, 0], A[:, 1]
    cands = []
    den = a1[2] * a2[2]
    if abs(den) > 1e-12:
        f2 = -(a1[0] * a2[0] + a1[1] * a2[1]) / den
        if f2 > 0:
            cands.append(np.sqrt(f2))
    den = a2[2] ** 2 - a1[2] ** 2
    if abs(den) > 1e-12:
        f2 = ((a1[0] ** 2 + a1[1] ** 2) - (a2[0] ** 2 + a2[1] ** 2)) / den
        if f2 > 0:
            cands.append(np.sqrt(f2))
    lo, hi = 0.3 * width, 6.0 * width
    for f in cands:
        if lo <= f <= hi:
            return float(f)
    return float(width)  # ~53 degree horizontal field of view fallback


_FOCAL_GRID = np.geomspace(0.25, 3.0, 80)   # focal length / image width: ~150 to ~19 degree fov


def focal_consistency(H: np.ndarray, image_size: tuple[int, int]):
    """How far a homography is from anything a real pinhole camera could produce.

    For the true court, K^-1 H has two columns that are perpendicular and equally long
    (the court's x and y axes) for some plausible focal length. A wrong match (say, two
    neighbouring courts read as one) generally fails that for every focal length.
    Returns the smallest residual over plausible focal lengths (0 = perfectly consistent).
    Accepts (3,3) or (N,3,3).
    """
    Hs = np.asarray(H, dtype=np.float64)
    single = Hs.ndim == 2
    Hs = Hs.reshape(-1, 3, 3)
    w, h = image_size
    Tinv = np.array([[1, 0, -w / 2.0], [0, 1, -h / 2.0], [0, 0, 1.0]])
    A = Tinv @ Hs
    u1, u2 = A[:, :2, 0] / w, A[:, :2, 1] / w                       # (N, 2)
    z1, z2 = A[:, 2, 0:1], A[:, 2, 1:2]                              # (N, 1)
    f = _FOCAL_GRID[None, :]                                         # (1, F)
    n1 = np.sqrt((u1 ** 2).sum(1, keepdims=True) / f ** 2 + z1 ** 2)
    n2 = np.sqrt((u2 ** 2).sum(1, keepdims=True) / f ** 2 + z2 ** 2)
    dot = (u1 * u2).sum(1, keepdims=True) / f ** 2 + z1 * z2
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.hypot(dot / (n1 * n2), np.log(n1 / n2))
    out = np.nan_to_num(e, nan=10.0).min(1)
    return float(out[0]) if single else out


def consistency_weight(H, image_size):
    """Multiplier in (0, 1] applied to hypothesis scores."""
    e = focal_consistency(H, image_size)
    return np.exp(-np.maximum(0.0, e - 0.02) / 0.08)


def pose_from_homography(H: np.ndarray, image_size: tuple[int, int], focal_px: float | None = None) -> CameraPose:
    """Recover camera intrinsics (focal only, principal point at center) and pose."""
    w, h = image_size
    cx, cy = w / 2.0, h / 2.0
    T = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1.0]])
    A = np.linalg.inv(T) @ H
    f = focal_px or _focal_from_h(A, w)
    M = np.diag([1 / f, 1 / f, 1.0]) @ A
    lam = 1.0 / np.sqrt(np.linalg.norm(M[:, 0]) * np.linalg.norm(M[:, 1]))
    r1, r2, t = lam * M[:, 0], lam * M[:, 1], lam * M[:, 2]
    if t[2] < 0:  # court must be in front of the camera
        r1, r2, t = -r1, -r2, -t
    R = np.stack([r1, r2, np.cross(r1, r2)], axis=1)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    C = -R.T @ t
    center = np.array([court.WIDTH / 2, court.LENGTH / 2])
    d = C[:2] - center
    horiz = float(np.hypot(*d))
    azimuth = float(np.degrees(np.arctan2(abs(d[0]), abs(d[1]))))
    elevation = float(np.degrees(np.arctan2(C[2], horiz)))
    if elevation > 60:
        view = "overhead"
    elif azimuth < 25:
        view = "baseline"
    elif azimuth > 65:
        view = "sideline"
    else:
        view = "corner"
    return CameraPose(f, cx, cy, R, t, C, azimuth, elevation, view)


def homography_from_pose(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    H = K @ np.stack([R[:, 0], R[:, 1], t], axis=1)
    return H / H[2, 2]


def look_at(cam_pos, target, up=(0, 0, 1.0)):
    """World->camera rotation for a camera at cam_pos looking at target (OpenCV axes)."""
    cam_pos, target, up = (np.asarray(v, dtype=np.float64) for v in (cam_pos, target, up))
    z = target - cam_pos
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-9:
        x = np.cross(z, (0, 1.0, 0))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    t = -R @ cam_pos
    return R, t
