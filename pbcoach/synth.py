"""Synthetic court renderer for testing calibration and tracking from any camera angle.

Renders a pickleball or tennis court (plus an adjacent court, another sport's lines, fence,
net and players) through a pinhole camera placed anywhere, and returns the exact ground-truth
homography and foot positions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .court import PICKLEBALL, Court
from .camera import homography_from_pose, look_at

PPM = 60   # ground texture resolution (pixels per meter)


def _extent(court: Court):
    """Ground texture extent (meters): the court, its run-off, a neighbouring court and a fence."""
    if court.sport == "pickleball":
        return -9.0, 16.0, -10.0, 23.5
    return -14.0, 2 * court.width + _neighbor_gap(court) + 4.0, -16.0, court.length + 16.0


def _neighbor_gap(court: Court) -> float:
    return 3.0 if court.sport == "pickleball" else 2 * court.runoff_side


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

# Tennis: a 10.97 x 23.77 m court, center (5.49, 11.89).
TENNIS_VIEWS = [
    View("broadcast_high", (5.49, -14.0, 8.0), (5.49, 11.0, 0.0), 45),
    View("baseline_fence_high", (5.49, -6.5, 4.0), (5.49, 12.0, 0.0), 75),
    View("baseline_low", (5.49, -6.0, 1.6), (5.49, 13.0, 0.6), 70),
    View("baseline_offcenter", (2.0, -7.0, 3.0), (6.0, 12.0, 0.0), 70),
    View("corner_high", (-6.0, -6.0, 5.0), (6.5, 10.5, 0.0), 75),
    View("corner_low", (-4.0, -5.0, 2.0), (5.49, 12.0, 0.4), 75),
    View("sideline_high", (-10.0, 11.9, 6.0), (5.49, 11.9, 0.0), 85),
    View("sideline_low", (-9.0, 11.0, 1.8), (5.49, 11.9, 0.3), 95),
    View("overhead", (5.49, 11.0, 34.0), (5.49, 11.9, 0.0), 75),
    View("phone_tilted", (4.5, -8.0, 2.2), (5.49, 12.0, 0.0), 70, roll_deg=8.0),
    View("near_half_zoom", (5.49, -6.0, 3.5), (5.49, 6.0, 0.0), 55),
    View("far_corner_fence", (15.5, -6.0, 3.0), (5.49, 12.0, 0.0), 70),
]


def views_for(court: Court | str = PICKLEBALL) -> list[View]:
    return TENNIS_VIEWS if getattr(court, "sport", court) == "tennis" else VIEWS


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
    other_lines: bool = True              # another sport's yellow lines, as on shared courts
    court: Court = PICKLEBALL
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
    court = scene.court
    X0, X1, Y0, Y1 = _extent(court)
    W = int((X1 - X0) * PPM)
    Hh = int((Y1 - Y0) * PPM)
    tex = np.zeros((Hh, W, 3), np.uint8)
    tex[:] = scene.apron

    def m2p(x, y):
        return int(round((x - X0) * PPM)), int(round((y - Y0) * PPM))

    def draw_court(ox, oy, surface):
        cv2.rectangle(tex, m2p(ox, oy), m2p(ox + court.width, oy + court.length), surface, -1)
        lw = max(1, int(round(court.line_width * PPM)))
        for (x1, y1), (x2, y2) in court.segments:
            cv2.line(tex, m2p(ox + x1, oy + y1), m2p(ox + x2, oy + y2), scene.line, lw, cv2.LINE_AA)

    def draw_other_lines(dx=0.0):  # differently colored lines of the other sport on a shared court
        col = (60, 200, 230)
        lw = max(1, int(round(0.05 * PPM)))
        if court.sport == "pickleball":   # the tennis court around it
            for x in (-1.6, court.width + 1.6):
                cv2.line(tex, m2p(x, -3.0), m2p(x, court.length + 3.0), col, lw, cv2.LINE_AA)
            cv2.line(tex, m2p(-1.6, -3.0), m2p(court.width + 1.6, -3.0), col, lw, cv2.LINE_AA)
            return
        # A pickleball court laid out inside the tennis court, sharing its net.
        p = PICKLEBALL
        ox, oy = dx + court.center_x - p.center_x, court.net_y - p.net_y
        for (x1, y1), (x2, y2) in p.segments:
            cv2.line(tex, m2p(ox + x1, oy + y1), m2p(ox + x2, oy + y2), col, lw, cv2.LINE_AA)

    draw_court(0.0, 0.0, scene.surface)
    if scene.other_lines:
        draw_other_lines()
    if scene.neighbor_court:
        dx = court.width + _neighbor_gap(court)
        draw_court(dx, 0.0, scene.surface)
        if scene.other_lines and court.sport == "tennis":   # the facility's other courts are lined alike
            draw_other_lines(dx)
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


def _draw_net(img, P, court: Court = PICKLEBALL):
    """Posts, translucent mesh and white tape."""
    y, o = court.net_y, court.net_overhang
    for x in (-o, court.width + o):
        _poly(img, P, _cylinder((x, y), 0.04, 0, court.net_height_post, 8), (30, 30, 30))
    xs = np.linspace(-o, court.width + o, 13)
    for xa, xb in zip(xs[:-1], xs[1:]):
        za, zb = court.net_tape_height(xa), court.net_tape_height(xb)
        _poly(img, P, [(xa, y, 0), (xb, y, 0), (xb, y, zb - 0.06), (xa, y, za - 0.06)], (25, 25, 25), 0.55)
        _poly(img, P, [(xa, y, za - 0.06), (xb, y, zb - 0.06), (xb, y, zb), (xa, y, za)], (240, 240, 240))


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
        X0, X1, Y0, Y1 = _extent(scene.court)
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
        _draw_net(img, P, scene.court)
    else:
        img = background.copy()

    # Players, far to near so nearer ones occlude; the net covers players on its far side.
    cam = -R.T @ t
    order = sorted(scene.players, key=lambda p: -np.hypot(p.x - cam[0], p.y - cam[1]))
    net_y = scene.court.net_y
    beyond = [p for p in order if (p.y - net_y) * (cam[1] - net_y) < 0]
    before = [p for p in order if p not in beyond]
    if beyond:
        layer = img.copy()
        for p in beyond:
            _draw_player(layer, P, p)
        changed = (layer != img).any(-1)
        netted = layer.copy()
        _draw_net(netted, P, scene.court)
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


def rally_positions(n_frames: int, seed: int = 0, court: Court = PICKLEBALL, players: int | None = None):
    """Plausible movement, shape (frames, players, 2). Pickleball doubles: two near players, two
    far, drifting toward the kitchen line. Tennis: baseline rallies with side-to-side running;
    in doubles the near team plays one up, one back."""
    rng = np.random.default_rng(seed)
    players = players or court.default_players
    if court.sport == "tennis":
        return _tennis_positions(n_frames, rng, court, players)
    starts = np.array([[1.5, 0.5], [4.6, 1.0], [1.5, 12.4], [4.6, 12.0]])
    goals = np.array([[1.6, 4.0], [4.5, 4.2], [1.6, 9.4], [4.5, 9.2]])
    out = []
    for f in range(n_frames):
        a = min(1.0, f / (0.6 * n_frames))
        pos = starts + (goals - starts) * a + rng.normal(0, 0.05, (4, 2))
        pos[:, 0] += 0.4 * np.sin(f / 7.0 + np.arange(4))
        out.append(pos)
    return np.array(out)


def _tennis_positions(n_frames: int, rng, court: Court, players: int):
    L, c = court.length, court.center_x
    if players <= 2:
        base = np.array([[c, -0.6], [c, L + 0.8]])
        swing = np.array([2.8, 2.4])
    else:
        # Near team: server back, partner at the net; far team: both on the baseline.
        base = np.array([[c - 2.2, -0.3], [c + 2.0, court.net_y - 3.0], [c - 2.5, L + 0.5], [c + 2.5, L + 0.5]])
        swing = np.array([1.5, 1.0, 1.5, 1.5])
    out = []
    for f in range(n_frames):
        pos = base.copy()
        pos[:, 0] += swing * np.sin(f / 9.0 + np.arange(len(base)) * 1.7)
        pos[:, 1] += 0.5 * np.sin(f / 13.0 + np.arange(len(base)))
        out.append(pos + rng.normal(0, 0.05, pos.shape))
    return np.array(out)


SHIRTS = [(40, 40, 200), (40, 160, 230), (200, 80, 40), (160, 60, 160)]


def make_rally_video(view: View, path, n_frames: int, seed: int = 0, fps: int = 15, court: Court = PICKLEBALL,
                     players: int | None = None):
    """Write a rally clip to `path` and the empty court to `path` with suffix .bg.png.

    Returns the true player positions, shape (frames, players, 2), in court meters.
    """
    path = Path(path)
    pos = rally_positions(n_frames, seed, court, players)
    bg, *_ = render(view, Scene(court=court), seed=seed)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (bg.shape[1], bg.shape[0]))
    for f in range(n_frames):
        scene = Scene(players=[Player(x, y, SHIRTS[i]) for i, (x, y) in enumerate(pos[f])],
                      noise=2.0, blur=0, jpeg=0, court=court)
        img, *_ = render(view, scene, seed=seed, background=bg)
        vw.write(img)
    vw.release()
    cv2.imwrite(str(path.with_suffix(".bg.png")), bg)
    return pos
