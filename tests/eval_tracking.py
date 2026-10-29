"""Player position accuracy (meters on court) across camera angles, on synthetic rally videos.

Run: python -m tests.eval_tracking [--sport tennis] [--players 2] [--frames 45] [--views baseline_high,corner_low] [--keep out_dir]
"""
from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from pbcoach import court as courts, synth
from pbcoach.pipeline import run

def score(tracks_csv: Path, gt: np.ndarray, court=courts.PICKLEBALL):
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
                est = np.stack([court.width - est[:, 0], court.length - est[:, 1]], 1)
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
    ap.add_argument("--sport", choices=sorted(courts.COURTS), default="pickleball")
    ap.add_argument("--players", type=int, choices=[2, 4])
    ap.add_argument("--frames", type=int, default=45)
    ap.add_argument("--views", default="")
    ap.add_argument("--keep", default="")
    a = ap.parse_args()
    court = courts.get(a.sport)
    players = a.players or court.default_players
    views = [v for v in synth.views_for(court) if not a.views or v.name in a.views.split(",")]
    tmp = Path(a.keep) if a.keep else Path(tempfile.mkdtemp())
    tmp.mkdir(parents=True, exist_ok=True)
    print(f"{'view':22s} mean_pos_err_m  players_found_within_50cm")
    for v in views:
        vid = tmp / f"{v.name}.mp4"
        gt = synth.make_rally_video(v, vid, a.frames, court=court, players=players)
        # Short clips: players barely move, so a median of frames would keep them in the
        # background. Use the empty-court frame, as a user would with --empty-court.
        run(str(vid), str(tmp / v.name), detector="foreground", write_video=bool(a.keep),
            empty_court=str(vid.with_suffix(".bg.png")), sport=court.sport, players=players)
        err, found = score(tmp / v.name / "tracks.csv", gt, court)
        print(f"{v.name:22s} {err:14.3f}  {found:8.1%}")


if __name__ == "__main__":
    main()
