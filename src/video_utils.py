"""
video_utils.py
--------------
Small robustness helpers around OpenCV video IO.

The important one is :func:`to_h264`. OpenCV writes ``mp4v``, which browsers
(and therefore Streamlit's ``st.video``) frequently refuse to play. If ffmpeg
is present we transcode to H.264 + faststart; if it is not, we fall back
gracefully and the dashboard streams annotated frames instead.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import cv2


def open_video(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")
    return cap


def video_info(cap: cv2.VideoCapture, fallback_fps: float = 20.0) -> dict:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 1 or fps > 120:   # NaN / bogus guard
        fps = fallback_fps
    return {
        "fps": float(fps),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    }


def make_writer(path: str, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"XVID"), fps, size)
    return writer


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def to_h264(src: str, dst: str | None = None) -> str:
    """Transcode to browser-friendly H.264. Returns ``src`` if ffmpeg is absent."""
    if not has_ffmpeg() or not os.path.exists(src):
        return src
    dst = dst or src.replace(".mp4", "_h264.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", src,
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
           "-movflags", "+faststart", dst]
    try:
        subprocess.run(cmd, check=True, timeout=600)
        return dst
    except (subprocess.SubprocessError, OSError):
        return src
