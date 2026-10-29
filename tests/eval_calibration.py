"""Calibration accuracy across camera angles on synthetic renders.

Run: python -m tests.eval_calibration [--sport tennis] [--seeds 3] [--save out_dir]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from pbcoach import court as courts, synth
from pbcoach.calibrate import calibrate_image
from pbcoach.camera import project, project_w
from pbcoach.viz import draw_court

def court_error_m(H_est, H_true, size, court=courts.PICKLEBALL):
    """Mean / max ground error (m) over visible court grid points; symmetric labeling allowed."""
    w, h = size
    R180 = np.array([[-1, 0, court.width], [0, -1, court.length], [0, 0, 1.0]])
    gx, gy = np.meshgrid(np.arange(0, court.width + 0.01, 0.5), np.arange(0, court.length + 0.01, 0.5))
    g = np.stack([gx.ravel(), gy.ravel()], 1)
    img, wv = project_w(H_true, g)
    vis = (wv > 0) & (img[:, 0] >= 0) & (img[:, 0] < w) & (img[:, 1] >= 0) & (img[:, 1] < h)
    if vis.sum() == 0:
        return np.inf, np.inf, 0
    best = None
    for F in (np.eye(3), R180):
        back = project(np.linalg.inv(H_est @ F), img[vis])
        e = np.linalg.norm(back - g[vis], axis=1)
        if best is None or e.mean() < best[0]:
            best = (float(e.mean()), float(e.max()), int(vis.sum()))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sport", choices=sorted(courts.COURTS), default="pickleball")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--views", default="", help="comma-separated view names (default: all)")
    ap.add_argument("--save", type=str, default="")
    ap.add_argument("--players", action="store_true", help="render players occluding lines")
    args = ap.parse_args()
    out = Path(args.save) if args.save else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    court = courts.get(args.sport)
    rows = []
    for view in synth.views_for(court):
        if args.views and view.name not in args.views.split(","):
            continue
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            v = synth.View(view.name, tuple(np.add(view.position, rng.normal(0, 0.4, 3) * (1, 1, 0.3))),
                           view.target, view.hfov_deg + rng.normal(0, 3), view.roll_deg + rng.normal(0, 2))
            scene = synth.Scene(court=court)
            if args.players:
                pos = synth.rally_positions(10, seed, court, 4)[5]
                scene.players = [synth.Player(x, y, synth.SHIRTS[i]) for i, (x, y) in enumerate(pos)]
            img, H_true, K, R, t = synth.render(v, scene, seed=seed)
            t0 = time.time()
            cal = calibrate_image(img, court=court)
            dt = time.time() - t0
            if cal is None:
                rows.append((view.name, seed, np.inf, np.inf, 0, 0.0, dt, "-", np.nan, np.nan))
                continue
            mean_e, max_e, n = court_error_m(cal.H, H_true, (img.shape[1], img.shape[0]), court)
            cam_true = -R.T @ t
            rows.append((view.name, seed, mean_e, max_e, n, cal.score, dt, cal.pose.view,
                         cal.pose.position[2] - cam_true[2], cal.pose.focal_px / K[0, 0]))
            if out:
                vis = draw_court(img.copy(), cal.H, (0, 255, 255), court=court)
                cv2.putText(vis, f"{view.name} err={mean_e:.2f}m score={cal.score:.2f} view={cal.pose.view}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imwrite(str(out / f"{view.name}_{seed}.jpg"), vis)
    print(f"{'view':22s} seed  mean_err_m  max_err_m  score  secs  est_view   height_err_m  focal_ratio")
    for r in rows:
        print(f"{r[0]:22s} {r[1]:4d}  {r[2]:10.3f} {r[3]:10.3f}  {r[5]:5.2f} {r[6]:5.1f}  {r[7]:9s}  {r[8]:12.2f}  {r[9]:6.2f}")
    errs = np.array([r[2] for r in rows])
    ok = errs < 0.15
    print(f"\n{ok.sum()}/{len(rows)} views calibrated within 15 cm mean error; median error {np.median(errs):.3f} m")


if __name__ == "__main__":
    main()
