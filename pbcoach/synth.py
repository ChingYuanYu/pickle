"""Synthetic court renderer for testing calibration and tracking from any camera angle.

Renders a court (plus an adjacent court, fence, net and players) through a pinhole
camera placed anywhere, and returns the exact ground-truth homography and foot positions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import court
from .camera import homography_from_pose, look_at

# Texture extent (meters) and resolution for the ground plane.
X0, X1, Y0, Y1 = -9.0, 16.0, -10.0, 23.5
PPM = 60


@dataclass
class View:
    name: str
    position: tuple
    target: tuple
    hfov_deg: float = 65.0
    roll_deg: float = 0.0


VIEWS = [
    View("baseline_high", (3.05, -8.0, 4.5), (3.05, 6.0, 0.0), 60),
    View("baseline_low", (3.05, -5.0, 1.5), (3.05, 7.5, 0.6), 70),
    View("baseline_far_tripod", (3.05, -16.0, 6.0), (3.05, 6.7, 0.0), 40),
    View("baseline_offcenter", (0.5, -6.0, 3.0), (3.5, 6.7, 0.0), 70),
    View("corner_high", (-5.0, -5.0, 4.0), (3.05, 6.7, 0.0), 70),
    View("corner_low", (-3.5, -3.5, 1.8), (3.05, 6.7, 0.4), 75),
    View("sideline_high", (-7.5, 6.7, 4.0), (3.05, 6.7, 0.0), 80),
    View("sideline_low", (-5.5, 6.0, 1.7), (3.05, 6.7, 0.4), 85),
    View("overhead", (3.05, 6.0, 16.0), (3.05, 6.7, 0.0), 60),
    View("phone_tilted", (2.5, -7.0, 2.2), (3.05, 6.7, 0.0), 65, roll_deg=9.0),
    View("near_half_zoom", (3.05, -5.0, 3.0), (3.05, 3.5, 0.0), 55),
    View("far_corner_fence", (8.5, -4.0, 2.6), (3.05, 7.0, 0.0), 70),
]


@dataclass
class Player:
    x: float
    y: float
    shirt: tuple
    height: float = 1.75


@dataclass
class Scene:
    surface: tuple = (150, 95, 40)        # court color, BGR (blue court)
    apron: tuple = (70, 120, 70)          # surround, BGR (green)
    line: tuple = (235, 235, 235)
    neighbor_court: bool = True
    tennis_lines: bool = True
    players: list = field(default_factory=list)
    noise: float = 4.0
    blur: int = 3
    jpeg: int = 75


def camera_for(view: View, size=(1280, 720)):
    w, h = size
    f = (w / 2) / np.tan(np.radians(view.hfov_deg) / 2)
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    R, t = look_at(view.position, view.target)
    if view.roll_deg:
        a = np.radians(view.roll_deg)
        Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
        R = Rz @ R
        t = Rz @ t
    return K, R, t


def _ground_texture(scene: Scene, rng) -> np.ndarray:
    W = int((X1 - X0) * PPM)
    Hh = int((Y1 - Y0) * PPM)
    tex = np.zeros((Hh, W, 3), np.uint8)
    tex[:] = scene.apron

    def m2p(x, y):
        return int(round((x - X0) * PPM)), int(round((y - Y0) * PPM))

    def draw_court(ox, oy, surface):
        cv2.rectangle(tex, m2p(ox, oy), m2p(ox + court.WIDTH, oy + court.LENGTH), surface, -1)
        lw = max(1, int(round(court.LINE_WIDTH * PPM)))
        for (x1, y1), (x2, y2) in court.SEGMENTS:
            cv2.line(tex, m2p(ox + x1, oy + y1), m2p(ox + x2, oy + y2), scene.line, lw, cv2.LINE_AA)

    if scene.tennis_lines:  # faint, differently colored lines of a shared tennis court
        col = (60, 200, 230)
        lw = max(1, int(round(0.05 * PPM)))
        for x in (-1.6, court.WIDTH + 1.6):
            cv2.line(tex, m2p(x, -3.0), m2p(x, court.LENGTH + 3.0), col, lw, cv2.LINE_AA)
        cv2.line(tex, m2p(-1.6, -3.0), m2p(court.WIDTH + 1.6, -3.0), col, lw, cv2.LINE_AA)
    draw_court(0.0, 0.0, scene.surface)
    if scene.neighbor_court:
        draw_court(court.WIDTH + 3.0, 0.0, scene.surface)
    noise = rng.normal(0, 6, tex.shape[:2])[..., None]
    return np.clip(tex + noise, 0, 255).astype(np.uint8)


def _poly(img, P, pts3d, color, alpha=1.0):
    X = np.hstack([np.asarray(pts3d, float), np.ones((len(pts3d), 1))]) @ P.T
    if (X[:, 2] <= 0.05).any():
        return
    xy = (X[:, :2] / X[:, 2:3]).astype(np.int32)
    if alpha >= 1:
        cv2.fillPoly(img, [cv2.convexHull(xy)], color, cv2.LINE_AA)
    else:
        over = img.copy()
        cv2.fillPoly(over, [cv2.convexHull(xy)], color, cv2.LINE_AA)
        cv2.addWeighted(over, alpha, img, 1 - alpha, 0, img)


def _cylinder(c, r, z0, z1, n=16):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a)], 1)
    return np.vstack([np.hstack([ring, np.full((n, 1), z0)]), np.hstack([ring, np.full((n, 1), z1)])])


def _draw_net(img, P):
    """Posts, translucent mesh and white tape."""
    for x in (-0.3, court.WIDTH + 0.3):
        _poly(img, P, _cylinder((x, court.NET_Y), 0.04, 0, court.NET_HEIGHT_POST, 8), (30, 30, 30))
    xs = np.linspace(-0.3, court.WIDTH + 0.3, 13)
    for xa, xb in zip(xs[:-1], xs[1:]):
        za = court.NET_HEIGHT_CENTER + (court.NET_HEIGHT_POST - court.NET_HEIGHT_CENTER) * abs(xa - court.CENTER_X) / court.CENTER_X
        zb = court.NET_HEIGHT_CENTER + (court.NET_HEIGHT_POST - court.NET_HEIGHT_CENTER) * abs(xb - court.CENTER_X) / court.CENTER_X
        _poly(img, P, [(xa, court.NET_Y, 0), (xb, court.NET_Y, 0), (xb, court.NET_Y, zb - 0.06), (xa, court.NET_Y, za - 0.06)], (25, 25, 25), 0.55)
        _poly(img, P, [(xa, court.NET_Y, za - 0.06), (xb, court.NET_Y, zb - 0.06), (xb, court.NET_Y, zb), (xa, court.NET_Y, za)], (240, 240, 240))


def _draw_player(img, P, p):
    _poly(img, P, _cylinder((p.x + 0.25, p.y + 0.15), 0.3, 0, 0.001), (30, 50, 30), 0.35)   # shadow
    _poly(img, P, _cylinder((p.x - 0.1, p.y), 0.08, 0, 0.85), (90, 80, 70))                  # legs
    _poly(img, P, _cylinder((p.x + 0.1, p.y), 0.08, 0, 0.85), (90, 80, 70))
    _poly(img, P, _cylinder((p.x, p.y), 0.22, 0.8, p.height - 0.25), p.shirt)                # torso
    _poly(img, P, _cylinder((p.x, p.y), 0.11, p.height - 0.25, p.height), (120, 150, 200))   # head


def render(view: View, scene: Scene | None = None, size=(1280, 720), seed: int = 0, background=None):
    """Render one frame. Returns (image BGR, H_true court->image, K, R, t)."""
    rng = np.random.default_rng(seed)
    scene = scene or Scene()
    w, h = size
    K, R, t = camera_for(view, size)
    H = homography_from_pose(K, R, t)
    P = K @ np.hstack([R, t.reshape(3, 1)])

    if background is None:
        tex = _ground_texture(scene, rng)
        T = np.array([[1 / PPM, 0, X0], [0, 1 / PPM, Y0], [0, 0, 1.0]])
        img = cv2.warpPerspective(tex, H @ T, size, flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=scene.apron)
        # Above the horizon: wall / sky gradient instead of the mirrored ground.
        uu, vv = np.meshgrid(np.arange(w), np.arange(h))
        rays = np.stack([uu, vv, np.ones_like(uu)], -1).reshape(-1, 3) @ np.linalg.inv(K).T @ R
        ground = (rays[:, 2] < 0).reshape(h, w)
        sky = np.zeros_like(img)
        grad = np.linspace(200, 140, h)[:, None]
        sky[..., 0] = np.clip(grad + 30, 0, 255)
        sky[..., 1] = np.clip(grad + 10, 0, 255)
        sky[..., 2] = np.clip(grad - 20, 0, 255)
        img = np.where(ground[..., None], img, sky)
        # Fence posts and a top rail around the facility.
        for x in np.arange(X0 + 1, X1, 3.0):
            for y in (Y0 + 1, Y1 - 1):
                _poly(img, P, _cylinder((x, y), 0.04, 0, 3.0, 6), (60, 60, 60))
        for y in np.arange(Y0 + 1, Y1, 3.0):
            for x in (X0 + 1, X1 - 1):
                _poly(img, P, _cylinder((x, y), 0.04, 0, 3.0, 6), (60, 60, 60))
        _draw_net(img, P)
    else:
        img = background.copy()

    # Players, far to near so nearer ones occlude; the net covers players on its far side.
    cam = -R.T @ t
    order = sorted(scene.players, key=lambda p: -np.hypot(p.x - cam[0], p.y - cam[1]))
    beyond = [p for p in order if (p.y - court.NET_Y) * (cam[1] - court.NET_Y) < 0]
    before = [p for p in order if p not in beyond]
    if beyond:
        layer = img.copy()
        for p in beyond:
            _draw_player(layer, P, p)
        changed = (layer != img).any(-1)
        netted = layer.copy()
        _draw_net(netted, P)
        img = np.where(changed[..., None], netted, layer)
    for p in before:
        _draw_player(img, P, p)

    if scene.blur > 1:
        img = cv2.GaussianBlur(img, (scene.blur | 1, scene.blur | 1), 0)
    img = np.clip(img + rng.normal(0, scene.noise, img.shape), 0, 255).astype(np.uint8)
    if scene.jpeg:
        _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, scene.jpeg])
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return img, H, K, R, t


def rally_positions(n_frames: int, seed: int = 0):
    """Plausible doubles movement: two near players, two far, drifting toward the kitchen line."""
    rng = np.random.default_rng(seed)
    starts = np.array([[1.5, 0.5], [4.6, 1.0], [1.5, 12.4], [4.6, 12.0]])
    goals = np.array([[1.6, 4.0], [4.5, 4.2], [1.6, 9.4], [4.5, 9.2]])
    out = []
    for f in range(n_frames):
        a = min(1.0, f / (0.6 * n_frames))
        pos = starts + (goals - starts) * a + rng.normal(0, 0.05, (4, 2))
        pos[:, 0] += 0.4 * np.sin(f / 7.0 + np.arange(4))
        out.append(pos)
    return np.array(out)


SHIRTS = [(40, 40, 200), (40, 160, 230), (200, 80, 40), (160, 60, 160)]
