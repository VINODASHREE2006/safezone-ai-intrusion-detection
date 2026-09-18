"""
motion_detector.py
------------------
Stage 1 of the pipeline: MOG2 background subtraction used as a *region
proposal* mechanism, so the expensive HOG person detector only ever runs on a
handful of small patches instead of the whole frame.

Innovation implemented here: ZONE-AWARE ADAPTIVE BACKGROUND LEARNING.
Two MOG2 models are maintained on the same frames:

  * bg_normal  - normal learning rate, used for pixels OUTSIDE hazard zones.
  * bg_hazard  - near-frozen learning rate, used for pixels INSIDE hazard
                 zones.

A worker who stands still inside an excavation pit is slowly absorbed into a
normally-learning background model and simply vanishes - which is exactly the
situation a safety system must never miss. Freezing the model inside hazard
zones keeps that worker in the foreground indefinitely, while the rest of the
scene still adapts to lighting changes, swaying barriers, dust, etc.
"""

from __future__ import annotations

import cv2
import numpy as np


class MotionDetector:
    """MOG2-based foreground segmentation + contour proposals."""

    def __init__(self, cfg: dict):
        m = cfg["motion"]
        self.shadow_value = int(m.get("shadow_value", 127))
        self.binary_threshold = int(m.get("binary_threshold", 200))
        self.lr_normal = float(m.get("learning_rate_normal", 0.005))
        self.lr_hazard = float(m.get("learning_rate_hazard", 0.00005))
        self.min_area = int(m.get("min_contour_area", 400))
        self.max_area_ratio = float(m.get("max_contour_area_ratio", 0.55))
        self.dilate_iterations = int(m.get("dilate_iterations", 2))

        def _make():
            return cv2.createBackgroundSubtractorMOG2(
                history=int(m.get("history", 500)),
                varThreshold=float(m.get("var_threshold", 32)),
                detectShadows=bool(m.get("detect_shadows", True)),
            )

        self.bg_normal = _make()
        self.bg_hazard = _make()

        ko = tuple(m.get("morph_open_kernel", [3, 3]))
        kc = tuple(m.get("morph_close_kernel", [9, 9]))
        self.k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ko)
        self.k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kc)

        self.last_mask: np.ndarray | None = None

    # ------------------------------------------------------------------
    def _clean(self, raw: np.ndarray) -> np.ndarray:
        """Drop MOG2 shadow pixels (value 127) and binarise."""
        mask = raw.copy()
        mask[mask == self.shadow_value] = 0
        _, mask = cv2.threshold(mask, self.binary_threshold, 255,
                                cv2.THRESH_BINARY)
        return mask

    # ------------------------------------------------------------------
    def apply(self, frame: np.ndarray,
              hazard_mask: np.ndarray | None = None) -> np.ndarray:
        """Return a cleaned binary foreground mask for ``frame``.

        ``hazard_mask`` is a uint8 mask (255 inside hazardous zones) produced
        by :class:`~src.zone_manager.ZoneManager`.
        """
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)

        fg_normal = self._clean(self.bg_normal.apply(blurred,
                                                     learningRate=self.lr_normal))
        fg_hazard = self._clean(self.bg_hazard.apply(blurred,
                                                     learningRate=self.lr_hazard))

        if hazard_mask is not None and hazard_mask.any():
            mask = np.where(hazard_mask > 0, fg_hazard, fg_normal).astype(np.uint8)
        else:
            mask = fg_normal

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.k_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.k_close)
        mask = cv2.dilate(mask, self.k_close, iterations=self.dilate_iterations)

        self.last_mask = mask
        return mask

    # ------------------------------------------------------------------
    def proposals(self, mask: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Contour bounding boxes ``(x, y, w, h)`` that survive area filters."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        frame_area = mask.shape[0] * mask.shape[1]
        boxes: list[tuple[int, int, int, int]] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            if area > frame_area * self.max_area_ratio:
                continue          # global lighting change / camera shake
            boxes.append(tuple(int(v) for v in cv2.boundingRect(c)))
        return boxes

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray, hazard_mask: np.ndarray | None = None):
        """Convenience wrapper -> ``(mask, proposal_boxes)``."""
        mask = self.apply(frame, hazard_mask)
        return mask, self.proposals(mask)
