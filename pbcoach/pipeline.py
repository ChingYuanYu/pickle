"""End-to-end prototype: video in, calibrated court, player tracks in meters, metrics out.

    python -m pbcoach.pipeline match.mp4 --out runs/match1
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from . import court
from .calibrate import Calibration, calibrate_from_points, calibrate_image, median_background
from .metrics import player_metrics, team_metrics
from .players import CourtTracker, OccupancyDetector, YoloDetector, lift_keypoints, locate
from .viz import draw_court, draw_ring, draw_skeleton, minimap, player_color


def read_frames(path: str, stride: int = 1, start: int = 0, end: int | None = None):
    """Yield (frame index, frame, effective fps) for frames start <= i < end, every stride-th."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i = 0
    while True:
        if end is not None and i >= end:
            break
        if i < start:   # sequential reads: seeking is unreliable across codecs
            if not cap.grab():
                break
            i += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        if (i - start) % stride == 0:
            yield i, frame, fps / stride
        i += 1
    cap.release()


def video_frame_count(path: str) -> int:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:  # some containers do not report a frame count: count by reading
        total = 0
        while cap.grab():
            total += 1
    cap.release()
    return total


def find_shots(path: str, end: int | None = None, cut_threshold: float = 30.0) -> list[tuple[int, int]]:
    """Split a video into camera shots at hard cuts, as (start, end) frame ranges.

    Broadcast highlights and edited clips cut between cameras; the court calibration only holds
    within one shot.
    """
    cap = cv2.VideoCapture(path)
    prev, cuts, i = None, [0], 0
    while end is None or i < end:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (160, 90), interpolation=cv2.INTER_AREA).astype(np.int16)
        if prev is not None and float(np.abs(g - prev).mean()) > cut_threshold:
            cuts.append(i)
        prev, i = g, i + 1
    cap.release()
    bounds = cuts + [i]
    return [(a, b) for a, b in zip(bounds, bounds[1:]) if b > a]


def sample_frames(path: str, n: int = 40, start: int = 0, end: int | None = None) -> list[np.ndarray]:
    end = video_frame_count(path) if end is None else end
    wanted = set(np.linspace(start, max(start, end - 1), max(1, min(n, end - start))).astype(int).tolist())
    return [f for i, f, _ in read_frames(path, 1, start, end) if i in wanted]


def smooth_rows(rows: list[dict], fps: float) -> list[dict]:
    """Offline smoothing of each player's path, as strong as that player's measurements are noisy.

    Far players seen by a low camera (where one pixel can be most of a meter) get a running
    median, which drops single-frame jumps, and a wide Gaussian; near players, measured to a
    few centimeters, are barely touched so quick steps survive.
    """
    from scipy.ndimage import gaussian_filter1d, median_filter

    by = {}
    for r in rows:
        by.setdefault(r["player_id"], []).append(r)
    for rs in by.values():
        rs.sort(key=lambda r: r["frame"])
        xy = np.array([(r["x"], r["y"]) for r in rs])
        sds = [r["sd"] for r in rs if r.get("sd") is not None]
        sd = float(np.median(sds)) if sds else 0.1
        if len(xy) >= 3:
            if sd > 0.25:
                k = min(int(round(0.5 * fps)) | 1, len(xy) - 1 + len(xy) % 2)  # odd, not longer than the path
                xy = median_filter(xy, size=(k, 1), mode="nearest")
            sigma_s = float(np.clip(0.4 * sd, 0.03, 0.3))
            xy = gaussian_filter1d(xy, sigma=max(0.3, sigma_s * fps), axis=0, mode="nearest")
        for r, (x, y) in zip(rs, xy):
            r["x"], r["y"] = round(float(x), 3), round(float(y), 3)
    return rows


def build_replay(cal: Calibration, rows: list[dict], seen: dict, fps: float) -> dict:
    """Per-frame player positions and 3D joints for the browser replay (court meters, z up)."""
    by_frame = {}
    for r in rows:
        by_frame.setdefault(r["frame"], []).append(r)
    # Fill short gaps (a player hidden behind a partner for a moment) so nobody blinks out of
    # the replay; filled frames show a plain figure since there is no pose for them.
    all_frames = sorted(by_frame)
    by_player = {}
    for r in rows:
        by_player.setdefault(r["player_id"], []).append(r)
    max_gap = int(3 * fps)
    for pid, rs in by_player.items():
        rs.sort(key=lambda r: r["frame"])
        for a, b in zip(rs, rs[1:]):
            gap = [f for f in all_frames if a["frame"] < f < b["frame"]]
            if gap and b["frame"] - a["frame"] <= max_gap:
                for f in gap:
                    w = (f - a["frame"]) / (b["frame"] - a["frame"])
                    by_frame[f].append({**a, "frame": f, "x": round(a["x"] + w * (b["x"] - a["x"]), 3),
                                        "y": round(a["y"] + w * (b["y"] - a["y"]), 3), "filled": True})
    frames = []
    for f in all_frames:
        ps = []
        for r in by_frame[f]:
            p = {"id": r["player_id"], "team": r["team"], "x": r["x"], "y": r["y"]}
            raw_xy, kp = (None, None) if r.get("filled") else seen.get(f, {}).get(r["player_id"], (None, None))
            if kp is not None:
                J = lift_keypoints(cal, kp, raw_xy)
                J[:, :2] += np.array([r["x"], r["y"]]) - raw_xy   # follow the smoothed position
                J[kp[:, 2] < 0.3] = np.nan
                p["joints"] = [None if not np.isfinite(j).all() else [round(float(v), 3) for v in j] for j in J]
            ps.append(p)
        frames.append({"f": f, "t": by_frame[f][0]["t"], "players": ps})
    cam = cal.pose
    w, h = cal.image_size
    return {"fps": fps, "court": {"width": court.WIDTH, "length": court.LENGTH, "net_y": court.NET_Y,
                                  "nvz_near": court.NVZ_NEAR_Y, "nvz_far": court.NVZ_FAR_Y},
            "camera": {"position": [round(float(v), 3) for v in cam.position],
                       "forward": [round(float(v), 4) for v in cam.R[2]],
                       "up": [round(float(-v), 4) for v in cam.R[1]],
                       "fov_deg": round(float(np.degrees(2 * np.arctan(h / 2 / cam.focal_px))), 2)},
            "colors": {str(r["player_id"]): "#%02x%02x%02x" % player_color(r["player_id"])[::-1] for r in rows},
            "frames": frames}


def render_overlay(video, path: Path, cal: Calibration, rows, seen, stride, start, end):
    """Video with each player's ground ring and skeleton in their color, plus a top-down map."""
    by_frame = {}
    for r in rows:
        by_frame.setdefault(r["frame"], []).append(r)
    trails = {}
    writer = None
    for idx, frame, fps in read_frames(video, stride, start, end):
        vis = draw_court(frame.copy(), cal.H, (0, 220, 255), 1)
        positions = {}
        for r in by_frame.get(idx, []):
            pid = r["player_id"]
            positions[pid] = (r["x"], r["y"], r["team"])
            trails.setdefault(pid, []).append((r["x"], r["y"]))
            draw_ring(vis, cal, (r["x"], r["y"]), player_color(pid))
            kp = seen.get(idx, {}).get(pid, (None, None))[1]
            if kp is not None:
                draw_skeleton(vis, kp, player_color(pid))
            p = cal.to_image([(r["x"], r["y"])])[0]
            if np.isfinite(p).all():
                cv2.putText(vis, str(pid), (int(p[0]) + 6, int(p[1]) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            player_color(pid), 2, cv2.LINE_AA)
        mm = minimap(positions, trails={k: v[-int(3 * fps):] for k, v in trails.items() if k in positions})
        h = vis.shape[0]
        mm = cv2.resize(mm, (int(mm.shape[1] * h / mm.shape[0]), h))
        panel = np.hstack([vis, mm])
        if writer is None:
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (panel.shape[1], panel.shape[0]))
        writer.write(panel)
    if writer is not None:
        writer.release()


def make_detector(kind: str, background: np.ndarray, cal: Calibration):
    if kind in ("auto", "yolo"):
        try:
            return YoloDetector(cal=cal)
        except Exception as e:  # ultralytics/torch missing or weights unavailable
            if kind == "yolo":
                raise
            print(f"YOLO unavailable ({e.__class__.__name__}); using foreground occupancy detector")
    return OccupancyDetector(cal, background)


def run(video: str, out_dir: str, detector: str = "auto", max_frames: int | None = None, stride: int = 1,
        points_json: str | None = None, write_video: bool = True, empty_court: str | None = None,
        progress=None) -> dict:
    """progress: optional callback(fraction, message) for front ends."""
    report = progress or (lambda frac, msg: None)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report(0.0, "Finding camera cuts")
    end = max_frames * stride if max_frames else None
    shots = find_shots(video, end)
    if not shots:
        raise SystemExit(f"could not read {video}")
    start, end = max(shots, key=lambda s: s[1] - s[0])   # analyze the longest continuous shot
    if len(shots) > 1:
        print(f"{len(shots)} camera shots found; analyzing the longest, frames {start}-{end - 1}")
    report(0.03, "Building the empty-court background")
    if empty_court:  # a photo of the empty court from the same camera beats a median of frames
        bg = cv2.imread(empty_court)
    else:
        frames = sample_frames(video, 40, start, end)
        if not frames:
            raise SystemExit(f"could not read {video}")
        bg = median_background(frames)
    h, w = bg.shape[:2]

    report(0.1, "Finding the court and the camera")
    if points_json:
        spec = json.loads(Path(points_json).read_text())
        cal = calibrate_from_points(spec["image_points"], spec["keypoints"], (w, h))
    else:
        cal = calibrate_image(bg)
        if cal is None:
            raise SystemExit("court not found; pass --points with 4+ clicked court keypoints")
    (out / "calibration.json").write_text(json.dumps(cal.summary(), indent=2))
    cv2.imwrite(str(out / "calibration.jpg"), draw_court(bg.copy(), cal.H))
    print(f"calibration: score={cal.score:.2f} view={cal.pose.view} "
          f"camera height={cal.pose.position[2]:.1f} m elevation={cal.pose.elevation_deg:.0f} deg")
    for n in cal.notes:
        print("  note:", n)

    det = make_detector(detector, bg, cal)
    total = end - start
    tracker = None
    rows = []
    seen = {}   # frame -> {player_id: (raw ground xy, keypoints or None)} for rendering and 3D replay
    for idx, frame, fps in read_frames(video, stride, start, end):
        if tracker is None:
            tracker = CourtTracker(fps)
        if total and idx % (5 * stride) == 0:
            report(0.1 + 0.75 * (idx - start) / total, f"Tracking players: frame {idx - start} of {total}")
        tracks = tracker.update(locate(cal, det(frame)), idx)
        for t in tracks:
            rows.append({"frame": idx, "t": round(idx / (fps * stride), 3), "player_id": t.id, "team": t.team,
                         "x": round(float(t.pos[0]), 3), "y": round(float(t.pos[1]), 3),
                         "sd": float(np.sqrt(np.trace(t.det.cov))) if t.det is not None and t.det.cov is not None
                         else None})
            if t.det is not None:
                seen.setdefault(idx, {})[t.id] = (t.det.court_xy, t.det.keypoints)
    fps_eff = tracker.dt ** -1 if tracker else 30.0
    rows = smooth_rows(rows, fps_eff)
    with open(out / "tracks.csv", "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=["frame", "t", "player_id", "team", "x", "y"], extrasaction="ignore")
        wcsv.writeheader()
        wcsv.writerows(rows)
    replay = build_replay(cal, rows, seen, fps_eff)
    (out / "replay.json").write_text(json.dumps(replay, separators=(",", ":")))
    if write_video:
        report(0.88, "Drawing the overlay video")
        render_overlay(video, out / "overlay.mp4", cal, rows, seen, stride, start, end)
    fps_eff = tracker.dt ** -1 if tracker else 30.0
    summary = {"calibration": cal.summary(), "players": player_metrics(rows, fps_eff), "teams": team_metrics(rows)}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    report(1.0, "Done")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--out", default="runs/latest")
    ap.add_argument("--detector", choices=["auto", "yolo", "foreground"], default="auto")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--points", help="JSON with image_points and keypoints for manual calibration")
    ap.add_argument("--empty-court", help="image of the empty court from the same camera position")
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    s = run(a.video, a.out, a.detector, a.max_frames, a.stride, a.points, not a.no_video, a.empty_court)
    print(json.dumps({"players": s["players"], "teams": s["teams"]}, indent=2))


if __name__ == "__main__":
    main()
