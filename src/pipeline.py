"""
pipeline.py
-----------
Glues the five stages together and draws the CCTV-style overlay.

    frame -> MOG2 proposals -> HOG worker classification -> centroid tracking
          -> foot-point zone test -> debounced intrusion + severity -> overlay

Both ``main.py`` (CLI / batch) and ``app.py`` (Streamlit dashboard) drive this
same class, so the demo and the offline run can never diverge.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import cv2
import numpy as np
import yaml

from .human_classifier import HumanClassifier, WORKER, UNKNOWN, NON_WORKER
from .intrusion_logic import IntrusionMonitor, SEVERITY_ORDER
from .motion_detector import MotionDetector
from .tracker import CentroidTracker
from .zone_manager import ZoneManager

FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class SafetyPipeline:
    """Stateful, frame-by-frame construction-site safety monitor."""

    def __init__(self, cfg: dict, fps: float | None = None,
                 clock_start: datetime | None = None):
        self.cfg = cfg
        v = cfg["video"]
        self.width = int(v.get("target_width", 640))
        self.height = int(v.get("target_height", 360))
        self.label = v.get("overlay_label", "CCTV LIVE FEED - CONSTRUCTION SITE")
        self.fps = float(fps or v.get("fallback_fps", 20)) or 20.0

        self.colors = cfg.get("display", {}).get("colors", {})
        self.show_mask = bool(cfg.get("display", {}).get("show_motion_mask", False))

        self.zones = ZoneManager(cfg, self.width, self.height)
        self.motion = MotionDetector(cfg)
        self.classifier = HumanClassifier(cfg)
        self.tracker = CentroidTracker(cfg)
        self.monitor = IntrusionMonitor(cfg)

        self.frame_index = 0
        self.latencies_ms: list[float] = []
        self.clock_start = clock_start or datetime.now().replace(microsecond=0)
        self._known_tracks: set[int] = set()
        self.detections = []

    # ------------------------------------------------------------------
    @property
    def video_time(self) -> float:
        return self.frame_index / self.fps

    def _color(self, key, default=(200, 200, 200)):
        return tuple(int(c) for c in self.colors.get(key, default))

    # ------------------------------------------------------------------
    def process(self, frame: np.ndarray) -> np.ndarray:
        """Process one raw frame; returns the annotated frame."""
        t0 = time.perf_counter()

        frame = cv2.resize(frame, (self.width, self.height))
        vis = frame.copy()

        # --- 1. motion proposals (zone-aware background learning) -------
        mask, proposals = self.motion.detect(frame, self.zones.hazard_mask)

        # --- 2. worker classification (HOG only on proposals) -----------
        self.detections = self.classifier.classify(frame, proposals)
        workers = [d for d in self.detections if d.label == WORKER]

        # --- 3. tracking ------------------------------------------------
        tracks = self.tracker.update(workers)
        now = self.video_time

        # Close intrusions for tracks that died (presence memory exhausted).
        alive = set(tracks.keys())
        for gone in self._known_tracks - alive:
            self.monitor.drop_worker(gone, now)
        self._known_tracks = alive

        # --- 4. foot-point zone test + debounced intrusion --------------
        for tr in tracks.values():
            if not tr.confirmed:
                continue
            zone = self.zones.zone_at(tr.foot_point)
            tr.in_hazard = zone is not None      # drives presence memory
            tr.last_zone = zone.name if zone else None
            self.monitor.update(tr.id, zone, now)

        # --- 5. render ---------------------------------------------------
        self.zones.draw(vis, self.monitor.breached_zones)
        self._draw_objects(vis, tracks)
        self._draw_hud(vis)

        if self.show_mask:
            small = cv2.resize(mask, (self.width // 4, self.height // 4))
            small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
            vis[self.height - small.shape[0] - 4:self.height - 4,
                self.width - small.shape[1] - 4:self.width - 4] = small

        self.frame_index += 1
        self.latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        return vis

    # ------------------------------------------------------------------
    def _draw_objects(self, vis, tracks):
        # Non-worker / unknown blobs, drawn thin for transparency.
        for d in self.detections:
            if d.label == WORKER:
                continue
            x, y, w, h = d.bbox
            col = self._color(d.label, (160, 160, 160))
            cv2.rectangle(vis, (x, y), (x + w, y + h), col, 1)
            cv2.putText(vis, d.label.replace("_", " ").title(), (x, y - 4),
                        FONT, 0.36, col, 1, cv2.LINE_AA)

        for tr in tracks.values():
            if not tr.confirmed:
                continue
            x, y, w, h = tr.bbox
            ev = self.monitor.active_events.get(tr.id)
            col = self._color(ev.severity, (0, 255, 0)) if ev else self._color("WORKER", (0, 255, 0))
            thick = 3 if ev else 2
            cv2.rectangle(vis, (x, y), (x + w, y + h), col, thick)

            # Foot point - the exact point tested against the polygons.
            fx, fy = tr.foot_point
            cv2.circle(vis, (fx, fy), 4, col, -1)
            cv2.circle(vis, (fx, fy), 6, (0, 0, 0), 1)

            label = tr.label
            if ev:
                label = f"{tr.label} | {ev.zone} | {ev.duration:4.1f} sec"
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.45, 1)
            lx = max(0, min(x, vis.shape[1] - tw - 8))
            ly = max(th + 30, y)           # keep clear of the top bar / banner
            cv2.rectangle(vis, (lx, ly - th - 8), (lx + tw + 6, ly - 1), col, -1)
            cv2.putText(vis, label, (lx + 3, ly - 5), FONT, 0.45, (0, 0, 0), 1,
                        cv2.LINE_AA)

            if tr.disappeared > 0:
                cv2.putText(vis, "PRESENCE MEMORY", (x, y + h + 12), FONT, 0.38,
                            (255, 255, 255), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    def _draw_hud(self, vis):
        h, w = vis.shape[:2]
        # Top bar
        cv2.rectangle(vis, (0, 0), (w, 24), (0, 0, 0), -1)
        cv2.putText(vis, self.label, (8, 17), FONT, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
        stamp = (self.clock_start + timedelta(seconds=self.video_time)) \
            .strftime("%Y-%m-%d %H:%M:%S")
        (tw, _), _ = cv2.getTextSize(stamp, FONT, 0.45, 1)
        cv2.putText(vis, stamp, (w - tw - 8, 17), FONT, 0.45, (0, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.circle(vis, (w - tw - 20, 12), 4,
                   (0, 0, 255) if (self.frame_index // 8) % 2 == 0 else (0, 0, 120), -1)

        # Stats strip
        cv2.rectangle(vis, (0, h - 22), (w, h), (0, 0, 0), -1)
        sev = self.monitor.severity_counts()
        txt = (f"Violations: {self.monitor.total_violations}   "
               f"Active: {self.monitor.active_count}   "
               f"W/A/C: {sev['WARNING']}/{sev['ALERT']}/{sev['CRITICAL']}   "
               f"Workers tracked: {len(self.tracker.active)}")
        cv2.putText(vis, txt, (8, h - 7), FONT, 0.42, (230, 230, 230), 1,
                    cv2.LINE_AA)

        # Alert banner for the worst active intrusion
        ev = self.monitor.worst_active()
        if ev is not None:
            col = self._color(ev.severity, (0, 0, 255))
            msg = (f"{self.monitor.banner_text(ev)} - Worker ID {ev.worker_id}"
                   f" | {ev.zone} | {ev.duration:.1f} sec")
            cv2.rectangle(vis, (0, 26), (w, 50), col, -1)
            cv2.putText(vis, msg, (8, 43), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def runtime_stats(self) -> dict:
        lat = np.array(self.latencies_ms) if self.latencies_ms else np.array([0.0])
        minutes = max(self.video_time / 60.0, 1e-6)
        return {
            "frames": self.frame_index,
            "video_seconds": round(self.video_time, 2),
            "avg_latency_ms": round(float(lat.mean()), 2),
            "p95_latency_ms": round(float(np.percentile(lat, 95)), 2),
            "processing_fps": round(1000.0 / max(float(lat.mean()), 1e-6), 2),
            "total_violations": self.monitor.total_violations,
            "alerts_per_minute": round(self.monitor.total_violations / minutes, 2),
        }

    def finalise(self):
        self.monitor.finalise(self.video_time)
