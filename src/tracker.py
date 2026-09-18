"""
tracker.py
----------
Stage 3: lightweight centroid tracker that gives every worker a stable
``Worker ID``.

Association: greedy nearest-neighbour on centroid distance with a hard
``max_distance`` gate. Unmatched tracks age out after ``max_disappeared``
frames.

Innovation implemented here: PRESENCE MEMORY.
A worker who disappears (occlusion behind a machine, crouching inside a pit,
detector miss) *while standing inside a hazardous zone* keeps their track
alive for ``presence_memory_frames`` instead of the normal, much shorter
budget. The safety timer therefore keeps running: the system assumes a worker
inside a pit is still in danger until proven otherwise.
"""

from __future__ import annotations

import numpy as np


class Track:
    """One tracked worker."""

    __slots__ = ("id", "bbox", "centroid", "foot_point", "disappeared",
                 "hits", "confirmed", "in_hazard", "last_zone", "trail",
                 "smoothing")

    def __init__(self, track_id: int, bbox, smoothing: float = 1.0):
        self.id = track_id
        self.smoothing = float(smoothing)
        self.update_box(bbox, smooth=False)
        self.disappeared = 0
        self.hits = 1
        self.confirmed = False
        self.in_hazard = False
        self.last_zone = None
        self.trail: list[tuple[int, int]] = [self.centroid]

    def update_box(self, bbox, smooth: bool = True):
        x, y, w, h = (float(v) for v in bbox)
        if smooth and self.smoothing < 1.0:
            a = self.smoothing
            px, py, pw, ph = self.bbox
            x = a * x + (1 - a) * px
            y = a * y + (1 - a) * py
            w = a * w + (1 - a) * pw
            h = a * h + (1 - a) * ph
        x, y, w, h = int(x), int(y), int(w), int(h)
        self.bbox = (x, y, w, h)
        self.centroid = (x + w // 2, y + h // 2)
        # Foot point: the ground-contact estimate used for all zone tests.
        self.foot_point = (int(x + w / 2), int(y + h))

    @property
    def label(self) -> str:
        return f"Worker ID {self.id}"

    def __repr__(self):
        return f"Track(#{self.id}, {self.bbox}, miss={self.disappeared})"


class CentroidTracker:
    def __init__(self, cfg: dict):
        t = cfg["tracking"]
        self.max_distance = float(t.get("max_distance", 90))
        self.max_disappeared = int(t.get("max_disappeared", 12))
        self.presence_memory = int(t.get("presence_memory_frames", 45))
        self.min_hits = int(t.get("min_hits_to_confirm", 2))
        # EMA factor applied to matched boxes: damps HOG scale jitter so the
        # foot point does not hop across a zone boundary frame to frame.
        self.smoothing = float(t.get("bbox_smoothing", 0.6))
        # A new detection this close to an existing track is treated as a
        # duplicate of that track rather than a second worker.
        self.duplicate_distance = float(t.get("duplicate_distance", 45))

        self._next_id = 1
        self.tracks: dict[int, Track] = {}

    # ------------------------------------------------------------------
    def _register(self, bbox) -> Track:
        tr = Track(self._next_id, bbox, self.smoothing)
        tr.confirmed = self.min_hits <= 1
        self.tracks[self._next_id] = tr
        self._next_id += 1
        return tr

    def _budget(self, track: Track) -> int:
        """How many missed frames this track is allowed - presence memory."""
        return self.presence_memory if track.in_hazard else self.max_disappeared

    # ------------------------------------------------------------------
    def update(self, detections) -> dict[int, Track]:
        """``detections``: list of objects exposing ``.bbox`` (workers only)."""
        boxes = [d.bbox for d in detections]

        if not boxes:
            for tid in list(self.tracks):
                tr = self.tracks[tid]
                tr.disappeared += 1
                if tr.disappeared > self._budget(tr):
                    del self.tracks[tid]
            return self.tracks

        if not self.tracks:
            for b in boxes:
                self._register(b)
            return self.tracks

        track_ids = list(self.tracks.keys())
        t_cent = np.array([self.tracks[i].centroid for i in track_ids], dtype=float)
        d_cent = np.array([(x + w / 2, y + h / 2) for (x, y, w, h) in boxes],
                          dtype=float)

        dist = np.linalg.norm(t_cent[:, None, :] - d_cent[None, :, :], axis=2)

        used_t: set[int] = set()
        used_d: set[int] = set()
        # Greedy: repeatedly take the globally smallest remaining distance.
        for ti, di in zip(*np.unravel_index(np.argsort(dist, axis=None),
                                            dist.shape)):
            if ti in used_t or di in used_d:
                continue
            if dist[ti, di] > self.max_distance:
                break
            tr = self.tracks[track_ids[ti]]
            tr.update_box(boxes[di])
            tr.disappeared = 0
            tr.hits += 1
            if tr.hits >= self.min_hits:
                tr.confirmed = True
            tr.trail.append(tr.centroid)
            if len(tr.trail) > 40:
                tr.trail.pop(0)
            used_t.add(ti)
            used_d.add(di)

        # Unmatched existing tracks age.
        for idx, tid in enumerate(track_ids):
            if idx in used_t:
                continue
            tr = self.tracks[tid]
            tr.disappeared += 1
            if tr.disappeared > self._budget(tr):
                del self.tracks[tid]

        # Unmatched detections become new tracks - unless they sit on top of
        # an existing track (duplicate box on the same worker).
        for di, b in enumerate(boxes):
            if di in used_d:
                continue
            cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
            if any(((cx - t.centroid[0]) ** 2 + (cy - t.centroid[1]) ** 2) ** 0.5
                   < self.duplicate_distance for t in self.tracks.values()):
                continue
            self._register(b)

        return self.tracks

    # ------------------------------------------------------------------
    @property
    def active(self) -> list[Track]:
        """Tracks worth drawing / reasoning about."""
        return [t for t in self.tracks.values() if t.confirmed]

    def reset(self):
        self.tracks.clear()
        self._next_id = 1
