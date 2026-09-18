"""
app.py - Streamlit dashboard for the Construction Site Restricted Zone
Intrusion Detection system.

    streamlit run app.py

The dashboard streams annotated frames live (which sidesteps every browser
H.264 codec problem), then offers the finished MP4 and the CSV event log for
download once the run completes.
"""

from __future__ import annotations

import io
import json
import os
import tempfile

import cv2
import pandas as pd
import streamlit as st

from src.demo_video import generate_demo_video
from src.evaluation import evaluate, load_groundtruth
from src.pipeline import SafetyPipeline, load_config
from src.video_utils import make_writer, open_video, to_h264, video_info

DEMO_PATH = "videos/demo_construction.mp4"
GT_PATH = "eval/groundtruth_demo.json"

st.set_page_config(page_title="Construction Site Safety Monitor",
                   page_icon="🚧", layout="wide")
st.markdown("""
<style>
/* Background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
}

/* Title styling */
h1 {
    color: #00f5ff;
    text-align: center;
    font-weight: 700;
}

/* Card style */
.card {
    background: rgba(255,255,255,0.05);
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0 0 15px rgba(0,255,255,0.2);
    margin-bottom: 10px;
}

/* KPI styling */
.kpi {
    text-align: center;
    padding: 10px;
    border-radius: 10px;
    background: linear-gradient(145deg, #1e293b, #0f172a);
    box-shadow: 0 0 10px rgba(0,255,255,0.3);
}

/* Button styling */
.stButton>button {
    background: linear-gradient(90deg, #00f5ff, #7c3aed);
    color: white;
    font-weight: bold;
    border-radius: 8px;
    height: 3em;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #020617;
}
/* Sidebar text color FIX */
[data-testid="stSidebar"] * {
    color: white !important;
}

/* Sidebar headings */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #00f5ff !important;
}

/* Sidebar labels */
[data-testid="stSidebar"] label {
    color: #e2e8f0 !important;
}

/* Slider text */
[data-testid="stSidebar"] .stSlider {
    color: white !important;
}

/* Alert styles */
.alert-warning {
    background: linear-gradient(90deg, #facc15, #f97316);
    padding: 10px;
    border-radius: 10px;
    color: black;
    font-weight: bold;
}

.alert-critical {
    background: linear-gradient(90deg, #ff006e, #ff4d6d);
    padding: 10px;
    border-radius: 10px;
    color: white;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

SEV_COLORS = {"WARNING": "#ffd24d", "ALERT": "#ff8c1a", "CRITICAL": "#ff3b30"}


# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _ensure_demo() -> str:
    if not os.path.exists(DEMO_PATH):
        generate_demo_video(DEMO_PATH)
    return DEMO_PATH


def sidebar_controls(cfg: dict) -> dict:
    st.sidebar.title("🚧 Site Safety Monitor")
    st.sidebar.caption("Classical CV · HOG+SVM · CPU only · no deep learning")

    st.sidebar.subheader("Detection")
    cfg["detection"]["hog_hit_threshold"] = st.sidebar.slider(
        "HOG hit threshold (lower = more sensitive)", -1.0, 1.0,
        float(cfg["detection"]["hog_hit_threshold"]), 0.05)
    cfg["motion"]["min_contour_area"] = st.sidebar.slider(
        "Min motion blob area (px)", 100, 3000,
        int(cfg["motion"]["min_contour_area"]), 50)

    st.sidebar.subheader("Intrusion logic")
    k = st.sidebar.slider("Debounce frames (entry/exit)", 1, 10,
                          int(cfg["intrusion"]["entry_debounce_frames"]))
    cfg["intrusion"]["entry_debounce_frames"] = k
    cfg["intrusion"]["exit_debounce_frames"] = k
    cfg["intrusion"]["warning_seconds"] = st.sidebar.slider(
        "WARNING → ALERT at (s)", 1.0, 10.0,
        float(cfg["intrusion"]["warning_seconds"]), 0.5)
    cfg["intrusion"]["alert_seconds"] = st.sidebar.slider(
        "ALERT → CRITICAL at (s)", 2.0, 20.0,
        float(cfg["intrusion"]["alert_seconds"]), 0.5)
    cfg["intrusion"]["use_risk_weighting"] = st.sidebar.checkbox(
        "Escalate faster in high-risk zones", True)

    st.sidebar.subheader("Tracking")
    cfg["tracking"]["presence_memory_frames"] = st.sidebar.slider(
        "Presence memory inside zones (frames)", 5, 120,
        int(cfg["tracking"]["presence_memory_frames"]), 5)

    st.sidebar.subheader("Display")
    cfg["display"]["show_motion_mask"] = st.sidebar.checkbox(
        "Show motion mask inset", False)
    stride = st.sidebar.select_slider("Process every Nth frame",
                                      options=[1, 2, 3, 4], value=1)

    with st.sidebar.expander("Configured hazardous zones", expanded=False):
        for z in cfg["zones"]:
            st.write(f"**{z['name']}** — risk `{z['risk']}`")

    return {"stride": stride}


def metrics_panel(pipe: SafetyPipeline, gt_path: str | None):
    stats = pipe.runtime_stats()
    st.sidebar.subheader("Evaluation metrics")
    st.sidebar.write(
        f"**Detection latency:** {stats['avg_latency_ms']} ms/frame "
        f"(p95 {stats['p95_latency_ms']} ms) → {stats['processing_fps']} FPS")
    st.sidebar.write(f"**Alerts / minute:** {stats['alerts_per_minute']}")

    if gt_path and os.path.exists(gt_path):
        res = evaluate(pipe.monitor.events, load_groundtruth(gt_path),
                       pipe.video_time)
        d = res.as_dict()
        st.sidebar.write(f"**Precision:** {d['Precision']}  |  "
                         f"**Recall:** {d['Recall']}  |  "
                         f"**F1:** {d['F1 Score']}")
        st.sidebar.write(f"**False alarms / min:** {d['False alarms / min']}")
        st.sidebar.write(f"**Alert latency:** {d['Mean detection latency (s)']} s")
        st.sidebar.caption("Scored against eval/groundtruth_demo.json")
    else:
        st.sidebar.caption("Upload/keep the demo clip to score Precision/Recall "
                           "against its ground truth.")


# ----------------------------------------------------------------------
def main():
    cfg = load_config("config.yaml")
    opts = sidebar_controls(cfg)

    st.markdown("""
<h1>🚧 SafeZone AI</h1>
<p style='text-align:center; color:#94a3b8;'>
Smart Construction Site Intrusion Detection System
</p>
""", unsafe_allow_html=True)
    st.caption("Worker detection · unique IDs · hazardous-zone intrusion · "
               "duration timers · severity alerts")

    left, right = st.columns([3, 2])
    with left:
        upload = st.file_uploader("Upload CCTV footage (.mp4 / .avi)",
                                  type=["mp4", "avi", "mov", "mkv"])
    with right:
        use_demo = st.checkbox("Use built-in demo clip", value=upload is None)
        st.caption("The demo clip is generated locally and ships with exact "
                   "ground truth for the metrics panel.")

    video_path, gt_path = None, None
    if upload is not None and not use_demo:
        suffix = os.path.splitext(upload.name)[1] or ".mp4"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(upload.read())
        tmp.close()
        video_path = tmp.name
    elif use_demo:
        video_path = _ensure_demo()
        gt_path = GT_PATH if os.path.exists(GT_PATH) else None

    if not video_path:
        st.info("Upload a video or tick **Use built-in demo clip** to start.")
        return

    if not st.button("▶ Run safety monitoring", type="primary"):
        st.stop()

    cap = open_video(video_path)
    info = video_info(cap, cfg["video"].get("fallback_fps", 20))
    pipe = SafetyPipeline(cfg, fps=info["fps"] / opts["stride"])

    os.makedirs("outputs", exist_ok=True)
    out_raw = "outputs/streamlit_annotated.mp4"
    writer = make_writer(out_raw, info["fps"] / opts["stride"],
                         (pipe.width, pipe.height))

    frame_slot = st.empty()
    banner_slot = st.empty()
    kpi = st.columns(4)
    kpi_slots = [c.empty() for c in kpi]
    zone_slot = st.empty()
    log_slot = st.empty()
    progress = st.progress(0.0, text="Processing…")

    total = max(info["frames"], 1)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if opts["stride"] > 1 and (i % opts["stride"]) != 0:
            continue

        vis = pipe.process(frame)
        writer.write(vis)
        frame_slot.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                         channels="RGB", width="stretch")

        if pipe.frame_index % 3 == 0 or i >= total:
            ev = pipe.monitor.worst_active()
            if ev is not None:
                col = SEV_COLORS.get(ev.severity, "#ff3b30")
                banner_slot.markdown(
                    f"<div style='background:{col};color:#111;padding:10px 14px;"
                    f"border-radius:8px;font-weight:700'>"
                    f"{pipe.monitor.banner_text(ev)} — Worker ID {ev.worker_id} "
                    f"| {ev.zone} | {ev.duration:.1f} sec | {ev.severity}</div>",
                    unsafe_allow_html=True)
            else:
                banner_slot.markdown(
                    "<div style='background:#1f7a3f;color:#fff;padding:10px 14px;"
                    "border-radius:8px;font-weight:700'>SITE CLEAR — no active "
                    "zone intrusion</div>", unsafe_allow_html=True)

            sev = pipe.monitor.severity_counts()
            kpi_slots[0].metric("Total safety violations",
                                pipe.monitor.total_violations)
            kpi_slots[1].metric("Active intrusions", pipe.monitor.active_count)
            kpi_slots[2].metric("Workers tracked", len(pipe.tracker.active))
            kpi_slots[3].metric("Critical events", sev["CRITICAL"])

            zc = pipe.monitor.zone_counts()
            if zc:
                zone_slot.bar_chart(pd.DataFrame(
                    {"Violations": zc}).sort_values("Violations",
                                                    ascending=False))
            rows = pipe.monitor.log_rows()
            if rows:
                log_slot.dataframe(pd.DataFrame(rows[::-1]),
                                   width="stretch", height=260)

        progress.progress(min(i / total, 1.0),
                          text=f"Processing frame {i}/{total}")

    cap.release()
    writer.release()
    pipe.finalise()
    progress.empty()

    st.success(f"Done — {pipe.frame_index} frames, "
               f"{pipe.monitor.total_violations} safety violations logged.")

    # Final tables
    rows = pipe.monitor.log_rows()
    st.subheader("Event log")
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, width="stretch")
        st.download_button("⬇ Download event log (CSV)",
                           df.to_csv(index=False).encode(),
                           file_name="intrusion_events.csv", mime="text/csv")
    else:
        st.info("No zone intrusions were detected in this clip.")

    metrics_panel(pipe, gt_path)

    out_play = to_h264(out_raw)
    if os.path.exists(out_play):
        st.subheader("Annotated output")
        try:
            st.video(out_play)
        except Exception:
            st.caption("Inline playback unavailable — use the download button.")
        with open(out_play, "rb") as fh:
            st.download_button("⬇ Download annotated video", fh.read(),
                               file_name="annotated.mp4", mime="video/mp4")

    report = {"runtime": pipe.runtime_stats(),
              "zone_violations": pipe.monitor.zone_counts(),
              "severity": pipe.monitor.severity_counts()}
    st.download_button("⬇ Download run report (JSON)",
                       json.dumps(report, indent=2).encode(),
                       file_name="report.json", mime="application/json")


if __name__ == "__main__":
    main()
