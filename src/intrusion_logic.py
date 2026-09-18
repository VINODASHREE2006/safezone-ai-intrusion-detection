"""
intrusion_logic.py
------------------
Stage 4: turn per-frame "worker X's feet are inside zone Y" observations into
stable safety events.

Three things happen here:

1. DEBOUNCING - a worker must be inside a zone for K consecutive frames before
   an intrusion opens, and outside for K consecutive frames before it closes.
   Without this, a single jittery detection flickers the alarm on and off many
   times per second and the log becomes useless.

2. DURATION TRACKING - each open intrusion carries a timer driven by *video*
   time, so results are identical whether the clip is processed at 8 FPS or
   30 FPS.

3. SEVERITY - duration is multiplied by the zone risk weight to get an
   "effective exposure", which maps to WARNING / ALERT / CRITICAL. A HIGH-risk
   excavation pit therefore escalates twice as fast as a LOW-risk storage area.

Presence memory lives in the tracker: while a track is kept alive inside a
zone, its timer here keeps running.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

WARNING = "WARNING"
ALERT = "ALERT"
CRITICAL = "CRITICAL"

SEVERITY_ORDER = {WARNING: 1, ALERT: 2, CRITICAL: 3}

ALERT_TEXT = {
    WARNING: "WARNING: Worker near hazardous zone",
    ALERT: "ALERT: Worker entered restricted zone",
    CRITICAL: "CRITICAL: Worker in high-risk zone",
}


@dataclass
class IntrusionEvent:
    worker_id: int
    zone: str
    risk: str
    start_time: float          # seconds into the video
    end_time: float | None = None
    duration: float = 0.0
    severity: str = WARNING
    timestamp: str = ""        # wall-clock-style HH:MM:SS of video time
    frames: int = 0
    active: bool = True

    def as_row(self) -> dict:
        d = asdict(self)
        d["duration"] = round(self.duration, 2)
        return d


def _fmt_clock(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


class IntrusionMonitor:
    """Per-worker debounced zone-occupancy state machine."""

    def __init__(self, cfg: dict):
        i = cfg["intrusion"]
        self.k_in = int(i.get("entry_debounce_frames", 3))
        self.k_out = int(i.get("exit_debounce_frames", 3))
        self.warn_s = float(i.get("warning_seconds", 3.0))
        self.alert_s = float(i.get("alert_seconds", 6.0))
        self.use_weight = bool(i.get("use_risk_weighting", True))

        # worker_id -> state dict
        self._state: dict[int, dict] = {}
        self.events: list[IntrusionEvent] = []
        self.active_events: dict[int, IntrusionEvent] = {}

    # ------------------------------------------------------------------
    def severity_for(self, duration: float, weight: float) -> str:
        eff = duration * (weight if self.use_weight else 1.0)
        if eff < self.warn_s:
            return WARNING
        if eff < self.alert_s:
            return ALERT
        return CRITICAL

    # ------------------------------------------------------------------
    def _state_for(self, wid: int) -> dict:
        return self._state.setdefault(
            wid, {"zone": None, "cand": None, "in_n": 0, "out_n": 0}
        )

    # ------------------------------------------------------------------
    def update(self, worker_id: int, zone, now: float) -> IntrusionEvent | None:
        """Feed one observation.

        ``zone`` is a :class:`~src.zone_manager.Zone` or ``None``.
        ``now`` is the current video timestamp in seconds.
        Returns the worker's active event, if any.
        """
        st = self._state_for(worker_id)
        zname = zone.name if zone is not None else None

        if st["zone"] is None:
            # ---- currently OUTSIDE: look for a debounced entry ----------
            if zname is None:
                st["cand"], st["in_n"] = None, 0
            else:
                if st["cand"] == zname:
                    st["in_n"] += 1
                else:
                    st["cand"], st["in_n"] = zname, 1
                if st["in_n"] >= self.k_in:
                    st["zone"], st["out_n"] = zname, 0
                    ev = IntrusionEvent(
                        worker_id=worker_id,
                        zone=zname,
                        risk=zone.risk,
                        start_time=now,
                        severity=WARNING,
                        timestamp=_fmt_clock(now),
                    )
                    self.events.append(ev)
                    self.active_events[worker_id] = ev
        else:
            # ---- currently INSIDE: update timer, look for exit ----------
            ev = self.active_events.get(worker_id)
            if ev is not None:
                ev.duration = max(0.0, now - ev.start_time)
                ev.frames += 1
                ev.severity = self.severity_for(ev.duration, self._weight(zone, ev))
            if zname == st["zone"]:
                st["out_n"] = 0
            else:
                st["out_n"] += 1
                if st["out_n"] >= self.k_out:
                    self.close(worker_id, now)
                    # A debounced entry into a *different* zone starts fresh.
                    if zname is not None:
                        st["cand"], st["in_n"] = zname, 1

        return self.active_events.get(worker_id)

    # ------------------------------------------------------------------
    @staticmethod
    def _weight(zone, ev: IntrusionEvent) -> float:
        if zone is not None and zone.name == ev.zone:
            return zone.weight
        return {"LOW": 1.0, "MEDIUM": 1.5, "HIGH": 2.0}.get(ev.risk, 1.0)

    # ------------------------------------------------------------------
    def close(self, worker_id: int, now: float):
        """Close an open intrusion (exit debounced, or track lost)."""
        st = self._state.get(worker_id)
        ev = self.active_events.pop(worker_id, None)
        if ev is not None:
            ev.end_time = now
            ev.duration = max(ev.duration, now - ev.start_time)
            ev.severity = self.severity_for(ev.duration,
                                            {"LOW": 1.0, "MEDIUM": 1.5,
                                             "HIGH": 2.0}.get(ev.risk, 1.0))
            ev.active = False
        if st is not None:
            st.update({"zone": None, "cand": None, "in_n": 0, "out_n": 0})

    def drop_worker(self, worker_id: int, now: float):
        """Track died (presence memory exhausted) -> close its intrusion."""
        self.close(worker_id, now)
        self._state.pop(worker_id, None)

    def finalise(self, now: float):
        for wid in list(self.active_events):
            self.close(wid, now)

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------
    @property
    def total_violations(self) -> int:
        return len(self.events)

    @property
    def active_count(self) -> int:
        return len(self.active_events)

    @property
    def breached_zones(self) -> set[str]:
        return {e.zone for e in self.active_events.values()}

    def zone_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.events:
            out[e.zone] = out.get(e.zone, 0) + 1
        return out

    def severity_counts(self) -> dict[str, int]:
        out = {WARNING: 0, ALERT: 0, CRITICAL: 0}
        for e in self.events:
            out[e.severity] = out.get(e.severity, 0) + 1
        return out

    def worst_active(self) -> IntrusionEvent | None:
        if not self.active_events:
            return None
        return max(self.active_events.values(),
                   key=lambda e: (SEVERITY_ORDER.get(e.severity, 0), e.duration))

    def log_rows(self) -> list[dict]:
        """Rows for the dashboard table / CSV export."""
        rows = []
        for e in self.events:
            rows.append({
                "Timestamp": e.timestamp,
                "Worker ID": e.worker_id,
                "Zone": e.zone,
                "Risk": e.risk,
                "Duration (s)": round(e.duration, 2),
                "Severity": e.severity,
                "Status": "ACTIVE" if e.active else "CLOSED",
            })
        return rows

    @staticmethod
    def banner_text(ev: IntrusionEvent) -> str:
        return ALERT_TEXT.get(ev.severity, ALERT_TEXT[WARNING])
