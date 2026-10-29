"""Web demo: upload a match clip, see the court, tracks and heatmaps, and get coaching feedback.

    uv sync --extra demo        # add --extra yolo for real footage
    uv run pbcoach-demo         # then open http://localhost:7860

Set ANTHROPIC_API_KEY to have Claude write the debrief and answer follow-up questions.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import pandas as pd

from . import coach, report, synth
from . import court as courts
from .replay3d import replay_html
from .metrics import player_metrics, team_metrics
from .pipeline import run

WORK = Path(tempfile.gettempdir()) / "pbcoach-demo"
SPORTS = ["pickleball", "tennis"]

INTRO = """
# 🏓🎾 Pickleball and tennis movement coach (demo)
Pick the sport, then upload a clip from a **fixed camera** (tripod, fence mount, or a tracking camera left still)
that shows the court. Any angle works: behind the baseline, a corner, the side, or overhead. The app finds the
court and the camera, tracks the players and their poses in court meters, rebuilds the rally in 3D, and turns
the movement into coaching feedback. Higher cameras see far players best. Ball and shot analysis are not part
of this prototype yet.
"""


def _players(fmt: str) -> int:
    return 2 if fmt == "singles" else 4


def view_names(sport: str) -> list[str]:
    return [v.name for v in synth.views_for(sport)]


def browser_video(src: Path) -> str | None:
    """Re-encode OpenCV's mp4v output to H.264 so every browser can play it."""
    if not src.exists():
        return None
    try:
        import imageio_ffmpeg
        dst = src.with_name(src.stem + "_h264.mp4")
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "26",
                        "-movflags", "+faststart", str(dst)], check=True)
        return str(dst)
    except Exception:
        return str(src)  # still downloadable; Gradio may convert it if ffmpeg is on PATH


def _camera_md(cal: dict) -> str:
    pose = cal.get("camera", {})
    pos = pose.get("position", [0, 0, 0])
    rows = [("Court fit", f"{cal.get('score', 0):.0%} of painted lines confirmed"),
            ("View", pose.get("view", "?")),
            ("Camera height", f"{pose.get('height_m', 0):.1f} m"),
            ("Camera position (court m)", f"x {pos[0]:.1f}, y {pos[1]:.1f}"),
            ("Looking down at", f"{pose.get('elevation_deg', 0):.0f}°"),
            ("Focal length", f"{pose.get('focal_px', 0):.0f} px")]
    md = "| | |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in rows)
    if cal.get("notes"):
        md += "\n\n" + "\n".join(f"- {n}" for n in cal["notes"])
    return md


def _tables(summary: dict):
    def depth(m):
        if "mean_distance_to_kitchen_line_m" in m:
            return {"avg_m_from_kitchen_line": m["mean_distance_to_kitchen_line_m"]}
        return {"avg_m_behind_baseline": m.get("mean_m_behind_baseline"),
                "avg_m_from_center": m.get("mean_m_from_center")}

    players = pd.DataFrame([
        {"player": pid, "team": m["team"], "seconds": m["seconds_tracked"], "distance_m": m["distance_m"],
         **depth(m), **{f"{z}_%": round(100 * v) for z, v in m["zone_share"].items()}}
        for pid, m in summary["players"].items()]).fillna(0)
    teams = pd.DataFrame([{"team": t, **m} for t, m in summary["teams"].items()])
    return players, teams


def kp_choices(sport: str) -> list[tuple[str, str]]:
    return [(f"{v} ({k})", k) for k, v in courts.get(sport).keypoint_help.items()]


KP_CHOICES = kp_choices("pickleball")


def first_frame(video):
    if not video:
        return None, []
    cap = cv2.VideoCapture(video)
    ok, f = cap.read()
    cap.release()
    return (cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if ok else None), []


def _draw_points(frame, points):
    if frame is None:
        return None
    img = frame.copy()
    for name, x, y in points:
        cv2.circle(img, (int(x), int(y)), 6, (255, 80, 0), -1, cv2.LINE_AA)
        cv2.putText(img, name, (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    return img


def add_point(frame, points, name, sport, evt: gr.SelectData):
    x, y = evt.index
    points = [p for p in points if p[0] != name] + [[name, float(x), float(y)]]
    keys = [k for _, k in kp_choices(sport)]
    nxt = next((k for k in keys[keys.index(name) + 1:] + keys if k not in {p[0] for p in points}), name)
    return _draw_points(frame, points), points, nxt, _points_md(points)


def _points_md(points):
    if not points:
        return "No points yet. Pick a court point from the list, then click where it is in the frame."
    need = max(0, 4 - len(points))
    return (f"{len(points)} point(s): " + ", ".join(p[0] for p in points) +
            (f". Add {need} more to use them." if need else ". These will be used instead of automatic court finding."))


def clear_points(frame):
    return frame, [], _points_md([])


def analyze(video, empty_court, detector, max_seconds, points=None, sport="pickleball", fmt="doubles",
            progress=gr.Progress()):
    if not video:
        raise gr.Error("Upload a video first, or try a sample rally.")
    WORK.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(dir=WORK))
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    court = courts.get(sport)
    points_json = None
    if points and len(points) >= 4 and all(p[0] in court.keypoints for p in points):
        points_json = out / "points.json"
        points_json.write_text(json.dumps({"keypoints": [p[0] for p in points],
                                           "image_points": [[p[1], p[2]] for p in points]}))
    try:
        summary = run(video, str(out), detector=detector, max_frames=int(max_seconds * fps) or None,
                      empty_court=empty_court or None, points_json=str(points_json) if points_json else None,
                      progress=lambda f, m: progress(f, desc=m), sport=sport, players=_players(fmt))
    except SystemExit as e:  # the pipeline reports user-facing failures this way
        raise gr.Error(f"{e} Click 4 or more court points under 'Fix the court' to set it by hand.")
    rows = list(csv.DictReader(open(out / "tracks.csv")))
    keep = coach.player_ids(rows)  # drop short ghost tracks from everything the user sees
    rows = [r for r in rows if int(r["player_id"]) in keep]
    for r in rows:
        r.update(frame=int(r["frame"]), t=float(r["t"]), x=float(r["x"]), y=float(r["y"]))
    summary.update(players=player_metrics(rows, fps, court), teams=team_metrics(rows, court))
    insights = coach.analyze(rows, summary, fps)
    findings = coach.to_markdown(insights)
    debrief = ""
    if coach.llm_available():
        progress(0.97, desc="Writing the coaching debrief")
        try:
            debrief = "### Coach's debrief\n" + coach.ask_coach(summary, insights) + "\n\n---\n"
        except Exception as e:
            debrief = f"_Claude debrief unavailable ({e.__class__.__name__}); showing rule-based findings._\n\n"
    replay = json.loads((out / "replay.json").read_text())
    for f in replay["frames"]:
        f["players"] = [p for p in f["players"] if p["id"] in keep]
    replay["colors"] = {k: v for k, v in replay["colors"].items() if int(k) in keep}
    frame3d = ('<iframe style="width:100%;height:560px;border:0;border-radius:10px" allow="fullscreen" '
               f'srcdoc="{html.escape(replay_html(replay), quote=True)}"></iframe>')
    heat = report.heatmaps(rows, court=court)
    players, teams = _tables(summary)
    state = {"summary": summary, "insights": [vars(i) for i in insights]}
    return (frame3d, debrief + findings,
            browser_video(out / "overlay.mp4"),
            str(out / "calibration.jpg"), _camera_md(summary["calibration"]),
            None if heat is None else cv2.cvtColor(heat, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(report.trails(rows, court), cv2.COLOR_BGR2RGB),
            gr.LinePlot(pd.DataFrame(report.depth_series(rows, court)),
                        y_title=f"meters behind own {court.front_name}"),
            players, teams,
            [str(out / "tracks.csv"), str(out / "metrics.json"), str(out / "calibration.json"),
             str(out / "replay.json")],
            state, [])


def make_sample(view_name, sport="pickleball", fmt="doubles", progress=gr.Progress()):
    progress(0, desc=f"Rendering a synthetic {sport} {fmt} rally")
    WORK.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(dir=WORK)) / f"sample_{sport}_{view_name}.mp4"
    views = synth.views_for(sport)
    view = next((v for v in views if v.name == view_name), views[0])
    synth.make_rally_video(view, path, n_frames=90, court=courts.get(sport), players=_players(fmt))
    # The synthetic players are blocks, not people a person detector would recognize.
    return browser_video(path), str(path.with_suffix(".bg.png")), "foreground"


def chat(message, history, state):
    if not state:
        return history + [{"role": "user", "content": message},
                          {"role": "assistant", "content": "Analyze a clip first, then ask me about it."}], ""
    insights = [coach.Insight(**i) for i in state["insights"]]
    if coach.llm_available():
        try:
            past = [{"role": m["role"], "content": m["content"]} for m in history]
            answer = coach.ask_coach(state["summary"], insights, message, past)
        except Exception as e:
            answer = f"Claude is unavailable ({e.__class__.__name__}).\n\n" + coach.offline_answer(message, insights)
    else:
        answer = coach.offline_answer(message, insights)
    return history + [{"role": "user", "content": message}, {"role": "assistant", "content": answer}], ""


def build() -> gr.Blocks:
    with gr.Blocks(title="Pickleball and tennis movement coach", fill_width=True) as app:
        state = gr.State(None)
        gr.Markdown(INTRO)
        with gr.Row():
            sport = gr.Radio(SPORTS, value="pickleball", label="Sport")
            fmt = gr.Radio(["doubles", "singles"], value="doubles", label="Format")
        with gr.Row(equal_height=False):
            video = gr.Video(label="Match clip", sources=["upload"], height=260, scale=2)
            empty = gr.Image(label="Empty-court photo from the same camera (optional)", type="filepath",
                             height=260, scale=1)
            with gr.Column(scale=1):
                go = gr.Button("Analyze", variant="primary", size="lg")
                with gr.Accordion("Options", open=False):
                    detector = gr.Radio(["auto", "foreground"], value="auto", label="People detector",
                                        info="auto uses YOLO with pose when installed; foreground needs a static camera")
                    max_seconds = gr.Slider(5, 300, value=60, step=5, label="Analyze the first N seconds")
                with gr.Group():
                    view = gr.Dropdown(view_names("pickleball"), value="corner_high",
                                       label="No clip handy? Camera angle")
                    sample = gr.Button("Load a synthetic sample rally")
        points = gr.State([])
        frame0 = gr.State(None)
        with gr.Accordion("Fix the court (optional): click known court points if the detected lines are off",
                          open=False):
            with gr.Row():
                pick = gr.Image(label="First frame: click to place the selected point", interactive=False, scale=3)
                with gr.Column(scale=1):
                    kp_name = gr.Dropdown(KP_CHOICES, value="near_left", label="Court point to place")
                    pts_md = gr.Markdown(_points_md([]))
                    clear = gr.Button("Clear points")
                    gr.Markdown("Four points are enough when they are not all on one line, for example both "
                                "ends of the near kitchen (or service) line and both near baseline corners. "
                                "Points hidden out of frame can be skipped; pick any visible line crossings.")
        with gr.Tabs():
            with gr.Tab("Replay"):
                with gr.Row():
                    with gr.Column(scale=1):
                        frame3d = gr.HTML("<p style='opacity:.6'>The 3D replay appears here after analysis. "
                                          "Drag to orbit, scroll to zoom, or pick a camera preset.</p>")
                    with gr.Column(scale=1):
                        overlay = gr.Video(label="Tracked video: pose, ground rings and top-down map", height=560)
            with gr.Tab("Coaching"):
                feedback = gr.Markdown("Results appear here after analysis.")
                chatbot = gr.Chatbot(label="Ask the coach", height=320)
                ask = gr.Textbox(placeholder="e.g. Why are we losing points in the middle?", show_label=False)
            with gr.Tab("Court & camera"):
                with gr.Row():
                    calib = gr.Image(label="Detected court", scale=2)
                    camera = gr.Markdown()
            with gr.Tab("Movement"):
                heat = gr.Image(label="Where each player spent time")
                with gr.Row():
                    trail = gr.Image(label="Paths over the whole clip", scale=1)
                    line = gr.LinePlot(x="t", y="behind_m", color="player", x_title="seconds",
                                       y_title="meters behind own kitchen line", height=420, scale=2)
            with gr.Tab("Numbers"):
                players = gr.Dataframe(label="Players")
                teams = gr.Dataframe(label="Teams")
                files = gr.File(label="Downloads", file_count="multiple")
        def on_sport(sp, frame):
            return (gr.Radio(value="singles" if sp == "tennis" else "doubles"),
                    gr.Dropdown(choices=view_names(sp), value="corner_high"),
                    gr.Dropdown(choices=kp_choices(sp), value="near_left"), [], frame, _points_md([]))

        sport.change(on_sport, [sport, frame0], [fmt, view, kp_name, points, pick, pts_md])
        sample.click(make_sample, [view, sport, fmt], [video, empty, detector])
        video.change(first_frame, video, [frame0, points]).then(lambda f: (f, _points_md([])), frame0, [pick, pts_md])
        pick.select(add_point, [frame0, points, kp_name, sport], [pick, points, kp_name, pts_md])
        clear.click(clear_points, frame0, [pick, points, pts_md])
        go.click(analyze, [video, empty, detector, max_seconds, points, sport, fmt],
                 [frame3d, feedback, overlay, calib, camera, heat, trail, line, players, teams, files, state, chatbot])
        ask.submit(chat, [ask, chatbot, state], [chatbot, ask])
    return app


def main():
    ap = argparse.ArgumentParser(description="Pickleball and tennis coach web demo")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="create a temporary public gradio.live link")
    a = ap.parse_args()
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    build().queue().launch(server_name=a.host, server_port=a.port, share=a.share, allowed_paths=[str(WORK)])


if __name__ == "__main__":
    main()
