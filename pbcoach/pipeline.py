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

from .calibrate import Calibration, calibrate_from_points, calibrate_image, median_background
from .metrics import player_metrics, team_metrics
from .players import CourtTracker, OccupancyDetector, YoloDetector, locate
from .viz import draw_court, minimap


def read_frames(path: str, max_frames: int | None = None, stride: int = 1):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and i >= max_frames * stride):
            break
        if i % stride == 0:
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


def sample_frames(path: str, n: int = 40) -> list[np.ndarray]:
    total = video_frame_count(path)
    wanted = set(np.linspace(0, max(0, total - 1), max(1, min(n, total))).astype(int).tolist())
    out = []
    cap = cv2.VideoCapture(path)
    for i in range(total):  # sequential reads: seeking is unreliable across codecs
        if not cap.grab():
            break
        if i in wanted:
            ok, f = cap.retrieve()
            if ok:
                out.append(f)
    cap.release()
    return out


def make_detector(kind: str, background: np.ndarray, cal: Calibration):
    if kind in ("auto", "yolo"):
        try:
            return YoloDetector()
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
    report(0.0, "Building the empty-court background")
    if empty_court:  # a photo of the empty court from the same camera beats a median of frames
        bg = cv2.imread(empty_court)
    else:
        frames = sample_frames(video)
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
    total = video_frame_count(video)
    if max_frames:
        total = min(total, max_frames * stride) if total else max_frames * stride
    tracker = None
    rows = []
    writer = None
    for idx, frame, fps in read_frames(video, max_frames, stride):
        if tracker is None:
            tracker = CourtTracker(fps)
        if total and idx % (5 * stride) == 0:
            report(0.15 + 0.8 * idx / total, f"Tracking players: frame {idx} of {total}")
        dets = locate(cal, det(frame))
        tracks = tracker.update(dets, idx)
        for t in tracks:
            rows.append({"frame": idx, "t": round(idx / (fps * stride), 3), "player_id": t.id, "team": t.team,
                         "x": round(float(t.pos[0]), 3), "y": round(float(t.pos[1]), 3)})
        if write_video:
            vis = draw_court(frame.copy(), cal.H, (0, 220, 255), 1)
            for d in dets:
                x1, y1, x2, y2 = map(int, d.box)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.circle(vis, tuple(int(v) for v in d.foot_px), 4, (0, 0, 255), -1)
            mm = minimap({t.id: (t.pos[0], t.pos[1], t.team) for t in tracks},
                         trails={t.id: [(x, y) for _, x, y in t.history[-60:]] for t in tracks})
            mm = cv2.resize(mm, (int(mm.shape[1] * h / mm.shape[0]), h))
            panel = np.hstack([vis, mm])
            if writer is None:
                writer = cv2.VideoWriter(str(out / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                         (panel.shape[1], panel.shape[0]))
            writer.write(panel)
    if writer is not None:
        writer.release()

    with open(out / "tracks.csv", "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=["frame", "t", "player_id", "team", "x", "y"])
        wcsv.writeheader()
        wcsv.writerows(rows)
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
