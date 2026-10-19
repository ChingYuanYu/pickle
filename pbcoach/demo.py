"""Web demo: upload a match clip, see the court, tracks and heatmaps, and get coaching feedback.

    uv sync --extra demo        # add --extra yolo for real footage
    uv run pbcoach-demo         # then open http://localhost:7860

Set ANTHROPIC_API_KEY to have Claude write the debrief and answer follow-up questions.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import pandas as pd

from . import coach, report, synth
from .metrics import player_metrics, team_metrics
from .pipeline import run

WORK = Path(tempfile.gettempdir()) / "pbcoach-demo"
VIEW_NAMES = [v.name for v in synth.VIEWS]

INTRO = """
# 🏓 Pickleball movement coach (demo)
Upload a clip from a **fixed camera** (tripod or fence mount) that shows the whole court. Any angle works:
behind the baseline, a corner, the side, or overhead. Higher cameras (3 m or more) see far players best.
The app finds the court and the camera, tracks the four players in court meters, and turns their
movement into coaching feedback. Ball and shot analysis are not part of this prototype yet.
"""


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
    players = pd.DataFrame([
        {"player": pid, "team": m["team"], "seconds": m["seconds_tracked"], "distance_m": m["distance_m"],
         "avg_m_from_kitchen_line": m["mean_distance_to_kitchen_line_m"],
         **{f"{z}_%": round(100 * v) for z, v in m["zone_share"].items()}}
        for pid, m in summary["players"].items()]).fillna(0)
    teams = pd.DataFrame([{"team": t, **m} for t, m in summary["teams"].items()])
    return players, teams


def analyze(video, empty_court, detector, max_seconds, progress=gr.Progress()):
    if not video:
        raise gr.Error("Upload a video first, or try a sample rally.")
    WORK.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(dir=WORK))
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    try:
        summary = run(video, str(out), detector=detector, max_frames=int(max_seconds * fps) or None,
                      empty_court=empty_court or None, progress=lambda f, m: progress(f, desc=m))
    except SystemExit as e:  # the pipeline reports user-facing failures this way
        raise gr.Error(f"{e} An empty-court photo from the same camera often fixes this.")
    rows = list(csv.DictReader(open(out / "tracks.csv")))
    keep = coach.player_ids(rows)  # drop short ghost tracks from everything the user sees
    rows = [r for r in rows if int(r["player_id"]) in keep]
    for r in rows:
        r.update(frame=int(r["frame"]), t=float(r["t"]), x=float(r["x"]), y=float(r["y"]))
    summary.update(players=player_metrics(rows, fps), teams=team_metrics(rows))
    insights = coach.analyze(rows, summary, fps)
    findings = coach.to_markdown(insights)
    debrief = ""
    if coach.llm_available():
        progress(0.97, desc="Writing the coaching debrief")
        try:
            debrief = "### Coach's debrief\n" + coach.ask_coach(summary, insights) + "\n\n---\n"
        except Exception as e:
            debrief = f"_Claude debrief unavailable ({e.__class__.__name__}); showing rule-based findings._\n\n"
    heat = report.heatmaps(rows)
    players, teams = _tables(summary)
    state = {"summary": summary, "insights": [vars(i) for i in insights]}
    return (debrief + findings,
            browser_video(out / "overlay.mp4"),
            str(out / "calibration.jpg"), _camera_md(summary["calibration"]),
            None if heat is None else cv2.cvtColor(heat, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(report.trails(rows), cv2.COLOR_BGR2RGB),
            pd.DataFrame(report.kitchen_distance_series(rows)),
            players, teams,
            [str(out / "tracks.csv"), str(out / "metrics.json"), str(out / "calibration.json")],
            state, [])


def make_sample(view_name, progress=gr.Progress()):
    progress(0, desc="Rendering a synthetic doubles rally")
    WORK.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(dir=WORK)) / f"sample_{view_name}.mp4"
    view = next(v for v in synth.VIEWS if v.name == view_name)
    synth.make_rally_video(view, path, n_frames=90)
    return browser_video(path), str(path.with_suffix(".bg.png"))


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
    with gr.Blocks(title="Pickleball movement coach") as app:
        state = gr.State(None)
        gr.Markdown(INTRO)
        with gr.Row():
            with gr.Column(scale=1):
                video = gr.Video(label="Match clip", sources=["upload"])
                empty = gr.Image(label="Empty-court photo from the same camera (optional)", type="filepath")
                with gr.Accordion("Options", open=False):
                    detector = gr.Radio(["auto", "foreground"], value="auto", label="People detector",
                                        info="auto uses YOLO when installed; foreground needs a static camera")
                    max_seconds = gr.Slider(5, 300, value=60, step=5, label="Analyze the first N seconds")
                go = gr.Button("Analyze", variant="primary")
                with gr.Group():
                    view = gr.Dropdown(VIEW_NAMES, value="corner_high", label="No clip handy? Camera angle")
                    sample = gr.Button("Load a synthetic sample rally")
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("Coaching"):
                        feedback = gr.Markdown("Results appear here after analysis.")
                        chatbot = gr.Chatbot(label="Ask the coach", height=320)
                        ask = gr.Textbox(placeholder="e.g. Why are we losing points in the middle?",
                                         show_label=False)
                    with gr.Tab("Video"):
                        overlay = gr.Video(label="Tracked video with top-down minimap")
                    with gr.Tab("Court & camera"):
                        calib = gr.Image(label="Detected court")
                        camera = gr.Markdown()
                    with gr.Tab("Movement"):
                        heat = gr.Image(label="Where each player spent time")
                        trail = gr.Image(label="Paths over the whole clip")
                        line = gr.LinePlot(x="t", y="behind_kitchen_m", color="player", x_title="seconds",
                                           y_title="meters behind own kitchen line", height=280)
                    with gr.Tab("Numbers"):
                        players = gr.Dataframe(label="Players")
                        teams = gr.Dataframe(label="Teams")
                        files = gr.File(label="Downloads", file_count="multiple")
        sample.click(make_sample, view, [video, empty])
        go.click(analyze, [video, empty, detector, max_seconds],
                 [feedback, overlay, calib, camera, heat, trail, line, players, teams, files, state, chatbot])
        ask.submit(chat, [ask, chatbot, state], [chatbot, ask])
    return app


def main():
    ap = argparse.ArgumentParser(description="Pickleball coach web demo")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="create a temporary public gradio.live link")
    a = ap.parse_args()
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    build().queue().launch(server_name=a.host, server_port=a.port, share=a.share, allowed_paths=[str(WORK)])


if __name__ == "__main__":
    main()
