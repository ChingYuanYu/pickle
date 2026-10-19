"""Player position accuracy (meters on court) across camera angles, on synthetic rally videos.

Run: python -m tests.eval_tracking [--frames 45] [--views baseline_high,corner_low] [--keep out_dir]
"""
from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from pbcoach import court, synth
from pbcoach.pipeline import run

R180 = np.array([[-1, 0, court.WIDTH], [0, -1, court.LENGTH], [0, 0, 1.0]])


def make_video(view, path: Path, n_frames: int, seed: int = 0, fps: int = 15):
    pos = synth.rally_positions(n_frames, seed)
    bg, *_ = synth.render(view, synth.Scene(), seed=seed)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (bg.shape[1], bg.shape[0]))
    for f in range(n_frames):
        scene = synth.Scene(players=[synth.Player(x, y, synth.SHIRTS[i]) for i, (x, y) in enumerate(pos[f])],
                            noise=2.0, blur=0, jpeg=0)
        img, *_ = synth.render(view, scene, seed=seed, background=bg)
        vw.write(img)
    vw.release()
    cv2.imwrite(str(path.with_suffix(".bg.png")), bg)
    return pos


def score(tracks_csv: Path, gt: np.ndarray):
    rows = list(csv.DictReader(open(tracks_csv)))
    by_frame = {}
    for r in rows:
        by_frame.setdefault(int(r["frame"]), []).append((float(r["x"]), float(r["y"])))
    best = None
    for flip in (False, True):  # the court is symmetric; allow the 180-degree labeling
        errs, found = [], 0
        for f, g in enumerate(gt):
            est = np.array(by_frame.get(f, []))
            if len(est) == 0:
                continue
            if flip:
                est = np.stack([court.WIDTH - est[:, 0], court.LENGTH - est[:, 1]], 1)
            C = np.linalg.norm(g[:, None] - est[None], axis=2)
            ri, ci = linear_sum_assignment(C)
            d = C[ri, ci]
            errs.extend(d[d < 1.5])
            found += int((d < 0.5).sum())
        res = (float(np.mean(errs)) if errs else np.inf, found / gt[:, :, 0].size)
        if best is None or res[0] < best[0]:
            best = res
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=45)
    ap.add_argument("--views", default="")
    ap.add_argument("--keep", default="")
    a = ap.parse_args()
    views = [v for v in synth.VIEWS if not a.views or v.name in a.views.split(",")]
    tmp = Path(a.keep) if a.keep else Path(tempfile.mkdtemp())
    tmp.mkdir(parents=True, exist_ok=True)
    print(f"{'view':22s} mean_pos_err_m  players_found_within_50cm")
    for v in views:
        vid = tmp / f"{v.name}.mp4"
        gt = make_video(v, vid, a.frames)
        # Short clips: players barely move, so a median of frames would keep them in the
        # background. Use the empty-court frame, as a user would with --empty-court.
        run(str(vid), str(tmp / v.name), detector="foreground", write_video=bool(a.keep),
            empty_court=str(vid.with_suffix(".bg.png")))
        err, found = score(tmp / v.name / "tracks.csv", gt)
        print(f"{v.name:22s} {err:14.3f}  {found:8.1%}")


if __name__ == "__main__":
    main()
