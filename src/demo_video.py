"""
demo_video.py
-------------
Generates the fallback demo clip so the project runs end-to-end even with no
CCTV footage at hand (``python main.py --make-demo``).

It renders a synthetic construction yard - dirt ground, sky, spoil heaps, a
parked excavator, a material stack - plus animated worker silhouettes with
head / torso / swinging arms / striding legs. The silhouettes carry the strong
vertical gradient structure the HOG pedestrian model was trained on, so the
real detector (not a shortcut) fires on them.

The walk paths are scripted, which is what makes the clip usable as an
evaluation set: ``eval/groundtruth_demo.json`` lists exactly when each worker's
feet are inside which zone.
"""

from __future__ import annotations

import math
import os

import cv2
import numpy as np

W, H = 640, 360
FPS = 20


# ----------------------------------------------------------------------
def _background(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bg = np.zeros((H, W, 3), np.uint8)

    # Sky gradient
    for y in range(0, 120):
        t = y / 120.0
        bg[y, :] = (int(190 - 25 * t), int(170 - 20 * t), int(150 - 20 * t))
    # Dirt ground gradient
    for y in range(120, H):
        t = (y - 120) / float(H - 120)
        bg[y, :] = (int(95 + 45 * t), int(110 + 50 * t), int(125 + 55 * t))

    # Distant hoarding / fence line
    cv2.rectangle(bg, (0, 104), (W, 124), (105, 115, 125), -1)
    for x in range(0, W, 26):
        cv2.line(bg, (x, 104), (x, 124), (80, 88, 96), 1)

    # Spoil heaps
    cv2.ellipse(bg, (90, 240), (75, 26), 0, 0, 360, (92, 112, 134), -1)
    cv2.ellipse(bg, (250, 150), (60, 16), 0, 0, 360, (98, 118, 138), -1)

    # Parked excavator (machinery zone)
    cv2.rectangle(bg, (640 - 130, 190), (640 - 40, 250), (40, 150, 200), -1)
    cv2.rectangle(bg, (640 - 130, 250), (640 - 40, 268), (55, 55, 60), -1)
    cv2.line(bg, (640 - 120, 190), (640 - 175, 150), (40, 150, 200), 9)
    cv2.line(bg, (640 - 175, 150), (640 - 150, 205), (40, 150, 200), 7)

    # Material stack (low-risk zone)
    for i in range(3):
        cv2.rectangle(bg, (205, 108 + i * 13), (385, 119 + i * 13),
                      (70, 105, 150), -1)

    # Ground texture
    noise = rng.normal(0, 5, (H, W, 1)).astype(np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return bg


# ----------------------------------------------------------------------
def _draw_worker(img: np.ndarray, cx: int, feet_y: int, height: int,
                 phase: float, vest: tuple = (40, 150, 245)):
    """Draw a person-shaped silhouette whose feet sit at ``feet_y``."""
    h = height
    body = (38, 40, 46)                       # dark work clothes
    head_r = max(3, int(h * 0.075))
    head_cy = feet_y - int(h * 0.92)
    torso_top = feet_y - int(h * 0.82)
    hip_y = feet_y - int(h * 0.48)
    half_w = max(3, int(h * 0.115))

    swing = math.sin(phase)
    stride = int(h * 0.12 * swing)
    arm = int(h * 0.09 * swing)

    # Legs (kept separated so the HOG model sees the classic inverted-V)
    leg_w = max(2, int(h * 0.058))
    cv2.line(img, (cx - leg_w, hip_y), (cx - leg_w + stride, feet_y), body,
             leg_w * 2, cv2.LINE_AA)
    cv2.line(img, (cx + leg_w, hip_y), (cx + leg_w - stride, feet_y), body,
             leg_w * 2, cv2.LINE_AA)

    # Torso + hi-vis vest
    torso = np.array([
        [cx - half_w, torso_top], [cx + half_w, torso_top],
        [cx + int(half_w * 0.85), hip_y], [cx - int(half_w * 0.85), hip_y],
    ], np.int32)
    cv2.fillConvexPoly(img, torso, vest, cv2.LINE_AA)
    cv2.polylines(img, [torso], True, body, 1, cv2.LINE_AA)
    cv2.line(img, (cx - half_w, int(torso_top + (hip_y - torso_top) * 0.55)),
             (cx + half_w, int(torso_top + (hip_y - torso_top) * 0.55)),
             (245, 245, 245), max(1, int(h * 0.028)), cv2.LINE_AA)

    # Arms
    arm_w = max(2, int(h * 0.045))
    cv2.line(img, (cx - half_w, torso_top + 2),
             (cx - half_w - int(h * 0.05), hip_y + arm), body, arm_w, cv2.LINE_AA)
    cv2.line(img, (cx + half_w, torso_top + 2),
             (cx + half_w + int(h * 0.05), hip_y - arm), body, arm_w, cv2.LINE_AA)

    # Head + helmet
    cv2.circle(img, (cx, head_cy), head_r, (55, 58, 64), -1, cv2.LINE_AA)
    cv2.ellipse(img, (cx, head_cy - int(head_r * 0.4)),
                (int(head_r * 1.25), int(head_r * 0.95)), 0, 180, 360,
                (40, 190, 250), -1, cv2.LINE_AA)


# ----------------------------------------------------------------------
# Scripted walk paths: (start_s, end_s, (x0,y0) -> (x1,y1), height, vest)
# y is the FOOT line, which is what the zone test uses.
PATHS = [
    # Worker A: crosses into the Excavation Zone (HIGH) and lingers
    dict(t0=1.0, t1=26.0, height=110, vest=(40, 150, 245),
         waypoints=[(0.0, (500, 300)), (0.25, (300, 305)), (0.45, (150, 310)),
                    (0.80, (120, 320)), (1.0, (110, 318))]),
    # Worker B: walks through the Machinery Zone (MEDIUM), then leaves
    dict(t0=6.0, t1=26.0, height=96, vest=(60, 200, 120),
         waypoints=[(0.0, (620, 200)), (0.30, (520, 245)), (0.55, (430, 255)),
                    (0.80, (330, 250)), (1.0, (285, 236))]),
    # Worker C: brief pass along the Material Storage (LOW) edge
    dict(t0=12.0, t1=24.0, height=70, vest=(200, 120, 220),
         waypoints=[(0.0, (200, 130)), (0.40, (330, 128)), (0.70, (430, 128)),
                    (1.0, (560, 132))]),
]


def _interp(waypoints, u: float) -> tuple[int, int]:
    u = min(max(u, 0.0), 1.0)
    for (u0, p0), (u1, p1) in zip(waypoints, waypoints[1:]):
        if u0 <= u <= u1:
            k = (u - u0) / max(u1 - u0, 1e-6)
            return (int(p0[0] + (p1[0] - p0[0]) * k),
                    int(p0[1] + (p1[1] - p0[1]) * k))
    return waypoints[-1][1]


def generate_demo_video(path: str = "videos/demo_construction.mp4",
                        seconds: float = 28.0, fps: int = FPS) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    base = _background()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (W, H))
    n = int(seconds * fps)

    for i in range(n):
        t = i / fps
        frame = base.copy()
        # Gentle global illumination drift - exercises background adaptation.
        drift = int(6 * math.sin(t * 0.35))
        frame = np.clip(frame.astype(np.int16) + drift, 0, 255).astype(np.uint8)

        # A non-human moving object (dumper truck) -> should be NON_WORKER.
        tx = int((t * 46) % (W + 220)) - 110
        cv2.rectangle(frame, (tx, 168), (tx + 96, 200), (60, 90, 190), -1)
        cv2.rectangle(frame, (tx + 70, 156), (tx + 100, 186), (80, 110, 210), -1)
        cv2.circle(frame, (tx + 22, 202), 9, (35, 35, 38), -1)
        cv2.circle(frame, (tx + 78, 202), 9, (35, 35, 38), -1)

        for p in PATHS:
            if not (p["t0"] <= t <= p["t1"]):
                continue
            u = (t - p["t0"]) / max(p["t1"] - p["t0"], 1e-6)
            cx, fy = _interp(p["waypoints"], u)
            _draw_worker(frame, cx, fy, p["height"], phase=t * 7.0,
                         vest=p["vest"])

        # Foreground occluder (stacked pipes). Drawn AFTER the workers, so a
        # worker walking behind it vanishes for ~1.5 s while still standing
        # inside the excavation zone - this is what exercises PRESENCE MEMORY.
        cv2.rectangle(frame, (196, 262), (266, 332), (58, 78, 104), -1)
        for k in range(3):
            cv2.circle(frame, (214 + k * 22, 278), 11, (70, 92, 120), -1)
            cv2.circle(frame, (214 + k * 22, 278), 11, (35, 45, 60), 2)
            cv2.circle(frame, (214 + k * 22, 308), 11, (70, 92, 120), -1)
            cv2.circle(frame, (214 + k * 22, 308), 11, (35, 45, 60), 2)

        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        writer.write(frame)

    writer.release()
    return path


if __name__ == "__main__":
    print(generate_demo_video())
