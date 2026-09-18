"""
zone_manager.py
---------------
Hazardous zone definition, point-in-zone testing and rendering.

Zones are polygons defined in ``config.yaml`` with NORMALISED coordinates
(0-1), so the same configuration works at any processing resolution.

Each zone carries a name ("Excavation Zone") and a risk level
(LOW / MEDIUM / HIGH), which drives both its colour and the severity
escalation weight used by the intrusion logic.

Zone membership is tested with ``cv2.pointPolygonTest`` on the worker's FOOT
POINT (x + w/2, y + h) - a person's bounding-box centre can sit over a pit
while their feet are safely on the edge, so the foot point is the only
defensible choice for ground-plane safety logic.
"""

from __future__ import annotations

import cv2
import numpy as np

RISK_LEVELS = ("LOW", "MEDIUM", "HIGH")


class Zone:
    def __init__(self, name: str, risk: str, points_norm, frame_w: int,
                 frame_h: int, color, weight: float):
        self.name = name
        self.risk = risk.upper()
        self.weight = float(weight)
        self.color = tuple(int(c) for c in color)
        self.points_norm = np.array(points_norm, dtype=np.float32)
        self.polygon = np.array(
            [[int(px * frame_w), int(py * frame_h)] for px, py in points_norm],
            dtype=np.int32,
        )

    # ------------------------------------------------------------------
    def contains(self, point) -> bool:
        return cv2.pointPolygonTest(self.polygon,
                                    (float(point[0]), float(point[1])),
                                    False) >= 0

    def distance(self, point) -> float:
        """Signed distance in px (positive = inside)."""
        return cv2.pointPolygonTest(self.polygon,
                                    (float(point[0]), float(point[1])), True)

    @property
    def anchor(self) -> tuple[int, int]:
        x, y, w, h = cv2.boundingRect(self.polygon)
        return x + 4, y + 16

    def __repr__(self):
        return f"Zone({self.name}, {self.risk})"


class ZoneManager:
    def __init__(self, cfg: dict, frame_w: int, frame_h: int):
        self.frame_w, self.frame_h = frame_w, frame_h
        colors = cfg.get("display", {}).get("colors", {})
        weights = cfg["intrusion"].get("zone_risk_weight", {})
        self.fill_alpha = float(cfg.get("display", {})
                                .get("zone_fill_alpha", 0.22))
        self.show_fill = bool(cfg.get("display", {}).get("show_zone_fill", True))

        self.zones: list[Zone] = []
        for z in cfg.get("zones", []):
            risk = str(z.get("risk", "MEDIUM")).upper()
            if risk not in RISK_LEVELS:
                risk = "MEDIUM"
            self.zones.append(
                Zone(
                    name=z["name"],
                    risk=risk,
                    points_norm=z["points"],
                    frame_w=frame_w,
                    frame_h=frame_h,
                    color=colors.get(risk, [0, 140, 255]),
                    weight=float(weights.get(risk, 1.0)),
                )
            )
        self._hazard_mask = self._build_mask()

    # ------------------------------------------------------------------
    def _build_mask(self) -> np.ndarray:
        mask = np.zeros((self.frame_h, self.frame_w), dtype=np.uint8)
        for z in self.zones:
            cv2.fillPoly(mask, [z.polygon], 255)
        return mask

    @property
    def hazard_mask(self) -> np.ndarray:
        """uint8 mask, 255 inside any hazardous zone (for adaptive BG)."""
        return self._hazard_mask

    # ------------------------------------------------------------------
    def zone_at(self, point) -> Zone | None:
        """Highest-risk zone containing ``point`` (usually the foot point)."""
        hits = [z for z in self.zones if z.contains(point)]
        if not hits:
            return None
        order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        return max(hits, key=lambda z: order.get(z.risk, 0))

    def near_zone(self, point, margin_px: float = 35.0) -> Zone | None:
        """Zone whose boundary is within ``margin_px`` of an outside point."""
        best, best_d = None, -1e9
        for z in self.zones:
            d = z.distance(point)
            if d < 0 and d > -margin_px and d > best_d:
                best, best_d = z, d
        return best

    def by_name(self, name: str) -> Zone | None:
        for z in self.zones:
            if z.name == name:
                return z
        return None

    # ------------------------------------------------------------------
    def draw(self, frame: np.ndarray, breached: set[str] | None = None,
             thickness: int = 2) -> np.ndarray:
        """Draw all zones. Zones in ``breached`` are highlighted."""
        breached = breached or set()
        if self.show_fill:
            overlay = frame.copy()
            for z in self.zones:
                cv2.fillPoly(overlay, [z.polygon], z.color)
            cv2.addWeighted(overlay, self.fill_alpha, frame,
                            1 - self.fill_alpha, 0, frame)

        for z in self.zones:
            hot = z.name in breached
            cv2.polylines(frame, [z.polygon], True, z.color,
                          thickness + (2 if hot else 0), cv2.LINE_AA)
            tag = f"{z.name} [{z.risk}]"
            if hot:
                tag += "  !! BREACH !!"
            ax, ay = z.anchor
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(frame, (ax - 3, ay - th - 5), (ax + tw + 3, ay + 4),
                          (0, 0, 0), -1)
            cv2.putText(frame, tag, (ax, ay), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        z.color, 1, cv2.LINE_AA)
        return frame

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.zones)

    def __iter__(self):
        return iter(self.zones)
