"""
human_classifier.py
-------------------
Stage 2: decide whether a motion blob is a WORKER (person), a NON_WORKER
object (vehicle, material being moved, swinging load, animal...) or UNKNOWN.

Method: ``cv2.HOGDescriptor`` with the built-in linear-SVM pedestrian model.
No deep learning, CPU only.

Efficiency: HOG is run *only* on padded crops around motion proposals, not on
the full frame. On a 640x360 feed that is typically 1-4 small crops per frame.

Labels
------
WORKER      : HOG fired AND the box passes geometric sanity filters.
UNKNOWN     : HOG did not fire but the blob is still person-shaped.
NON_WORKER  : clearly not a person (wrong aspect ratio / too small).

Only WORKER detections are forwarded to the tracker and can raise safety
alerts - UNKNOWN and NON_WORKER are drawn for transparency but never trigger
an intrusion event.
"""

from __future__ import annotations

import cv2
import numpy as np

WORKER = "WORKER"
NON_WORKER = "NON_WORKER"
UNKNOWN = "UNKNOWN"

# HOG's default people detector expects at least a 64x128 window.
_MIN_W, _MIN_H = 64, 128


class Detection:
    """A single classified object in one frame."""

    __slots__ = ("bbox", "label", "score")

    def __init__(self, bbox, label: str, score: float = 0.0):
        self.bbox = tuple(int(v) for v in bbox)   # (x, y, w, h)
        self.label = label
        self.score = float(score)

    # -- helpers -------------------------------------------------------
    @property
    def centroid(self) -> tuple[int, int]:
        x, y, w, h = self.bbox
        return x + w // 2, y + h // 2

    @property
    def foot_point(self) -> tuple[int, int]:
        """Ground contact point - the only correct point for zone tests."""
        x, y, w, h = self.bbox
        return int(x + w / 2), int(y + h)

    def __repr__(self):
        return f"Detection({self.label}, {self.bbox}, {self.score:.2f})"


class HumanClassifier:
    def __init__(self, cfg: dict):
        d = cfg["detection"]
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        self.win_stride = tuple(d.get("hog_win_stride", [8, 8]))
        self.padding = tuple(d.get("hog_padding", [16, 16]))
        self.scale = float(d.get("hog_scale", 1.05))
        self.hit_threshold = float(d.get("hog_hit_threshold", 0.0))
        self.roi_padding = int(d.get("roi_padding", 24))

        self.min_area = int(d.get("min_person_area", 700))
        self.min_height = int(d.get("min_box_height", 45))
        self.ar_min = float(d.get("aspect_ratio_min", 1.5))
        self.ar_max = float(d.get("aspect_ratio_max", 4.5))
        self.unk_min = float(d.get("unknown_aspect_min", 1.2))
        self.unk_max = float(d.get("unknown_aspect_max", 5.5))
        self.nms_overlap = float(d.get("nms_overlap", 0.45))
        # The default HOG people model returns a window with ~15% padding on
        # every side. Shrinking it back to the body is essential here: the
        # zone test uses the FOOT POINT, and an over-tall box pushes the feet
        # up to 30 px below where the worker actually stands.
        shrink = d.get("hog_box_shrink", {}) or {}
        self.shrink_w = float(shrink.get("width", 0.60))
        self.shrink_h = float(shrink.get("height", 0.72))
        self.shrink_top = float(shrink.get("top_offset", 0.13))

    # ------------------------------------------------------------------
    @staticmethod
    def _aspect(bbox) -> float:
        _, _, w, h = bbox
        return h / max(w, 1)

    def _passes_person_geometry(self, bbox) -> bool:
        _, _, w, h = bbox
        ar = self._aspect(bbox)
        return (self.ar_min <= ar <= self.ar_max
                and w * h >= self.min_area
                and h >= self.min_height)

    def _looks_person_like(self, bbox) -> bool:
        _, _, w, h = bbox
        ar = self._aspect(bbox)
        return self.unk_min <= ar <= self.unk_max and w * h >= self.min_area * 0.6

    # ------------------------------------------------------------------
    def _hog_on_roi(self, frame: np.ndarray, box) -> list[tuple[tuple, float]]:
        """Run HOG inside a padded crop; return boxes in FRAME coordinates."""
        H, W = frame.shape[:2]
        x, y, w, h = box
        p = self.roi_padding
        x0, y0 = max(0, x - p), max(0, y - p)
        x1, y1 = min(W, x + w + p), min(H, y + h + p)
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            return []

        rh, rw = roi.shape[:2]
        # HOG needs a window of at least 64x128 - upscale small crops.
        sx = max(1.0, _MIN_W / rw)
        sy = max(1.0, _MIN_H / rh)
        s = max(sx, sy)
        if s > 1.0:
            s = min(s, 6.0)          # guard against insane upscales
            roi = cv2.resize(roi, (int(rw * s), int(rh * s)),
                             interpolation=cv2.INTER_LINEAR)
        else:
            s = 1.0

        try:
            rects, weights = self.hog.detectMultiScale(
                roi,
                winStride=self.win_stride,
                padding=self.padding,
                scale=self.scale,
                hitThreshold=self.hit_threshold,
            )
        except cv2.error:
            return []

        out = []
        for (rx, ry, rw_, rh_), score in zip(rects, np.ravel(weights)):
            fx = x0 + rx / s
            fy = y0 + ry / s
            fw = rw_ / s
            fh = rh_ / s
            # Remove the HOG window padding -> tight body box.
            fx = int(fx + fw * (1 - self.shrink_w) / 2)
            fy = int(fy + fh * self.shrink_top)
            fw = int(fw * self.shrink_w)
            fh = int(fh * self.shrink_h)
            fx, fy = max(0, fx), max(0, fy)
            fw, fh = min(fw, W - fx), min(fh, H - fy)
            if fw > 4 and fh > 4:
                out.append(((fx, fy, fw, fh), float(score)))
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def _iou(a, b) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1, y1 = max(ax, bx), max(ay, by)
        x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0
        return inter / float(aw * ah + bw * bh - inter)

    @staticmethod
    def _containment(a, b) -> float:
        """Intersection over the *smaller* box - catches nested duplicates
        produced by different levels of the HOG scale pyramid, which plain IoU
        misses and which would otherwise spawn two tracks on one worker."""
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1, y1 = max(ax, bx), max(ay, by)
        x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0
        return inter / float(min(aw * ah, bw * bh))

    def _nms(self, items: list[tuple[tuple, float]]) -> list[tuple[tuple, float]]:
        items = sorted(items, key=lambda t: t[1], reverse=True)
        kept: list[tuple[tuple, float]] = []
        for box, score in items:
            dup = any(self._iou(box, k[0]) >= self.nms_overlap
                      or self._containment(box, k[0]) >= 0.55 for k in kept)
            if not dup:
                kept.append((box, score))
        return kept

    # ------------------------------------------------------------------
    def classify(self, frame: np.ndarray, proposals) -> list[Detection]:
        """Classify every motion proposal. Returns a list of Detection."""
        hog_hits: list[tuple[tuple, float]] = []
        for box in proposals:
            hog_hits.extend(self._hog_on_roi(frame, box))
        hog_hits = self._nms(hog_hits)

        detections: list[Detection] = []
        workers: list[tuple] = []
        for box, score in hog_hits:
            if self._passes_person_geometry(box):
                detections.append(Detection(box, WORKER, score))
                workers.append(box)

        # Motion blobs not explained by a WORKER box -> UNKNOWN / NON_WORKER
        for box in proposals:
            if any(self._iou(box, w) > 0.25 for w in workers):
                continue
            label = UNKNOWN if self._looks_person_like(box) else NON_WORKER
            detections.append(Detection(box, label, 0.0))

        return detections

    # ------------------------------------------------------------------
    @staticmethod
    def workers(detections: list[Detection]) -> list[Detection]:
        return [d for d in detections if d.label == WORKER]
