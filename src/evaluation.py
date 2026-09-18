"""
evaluation.py
-------------
Turns the raw event log into the metrics reported in the README and the
Streamlit sidebar.

Ground truth format (JSON, see ``eval/groundtruth_demo.json``)::

    {
      "video": "videos/demo_construction.mp4",
      "intrusions": [
        {"zone": "Excavation Zone", "start": 4.5, "end": 11.0},
        {"zone": "Machinery Zone",  "start": 13.0, "end": 18.5}
      ]
    }

Matching rule: a detected event is a TRUE POSITIVE if it names the same zone
as an un-consumed ground-truth interval AND their time ranges overlap
(each GT interval can be matched at most once - repeated re-triggers inside
the same interval count as false alarms, which is the honest way to score a
system whose job is one alert per real breach).

    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)
    F1        = 2PR / (P + R)
    False alarms / minute = FP / video_minutes
    Detection latency = time from GT interval start to the first matching
                        detected event (includes the K-frame debounce cost),
                        reported as an absolute offset, alongside per-frame processing latency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class EvalResult:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    false_alarms_per_min: float = 0.0
    mean_detection_latency_s: float = 0.0
    video_minutes: float = 0.0

    def as_dict(self) -> dict:
        return {
            "True Positives": self.tp,
            "False Positives": self.fp,
            "False Negatives": self.fn,
            "Precision": round(self.precision, 3),
            "Recall": round(self.recall, 3),
            "F1 Score": round(self.f1, 3),
            "False alarms / min": round(self.false_alarms_per_min, 2),
            "Mean detection latency (s)": round(self.mean_detection_latency_s, 2),
        }


def load_groundtruth(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("intrusions", data if isinstance(data, list) else [])


def _overlaps(a0, a1, b0, b1) -> bool:
    return max(a0, b0) <= min(a1, b1)


def evaluate(events, groundtruth: list[dict], video_seconds: float) -> EvalResult:
    """``events``: list of IntrusionEvent (or dicts with the same fields)."""
    det = []
    for e in events:
        if isinstance(e, dict):
            det.append((e["zone"], float(e["start_time"]),
                        float(e.get("end_time") or e["start_time"] + e.get("duration", 0))))
        else:
            end = e.end_time if e.end_time is not None else e.start_time + e.duration
            det.append((e.zone, float(e.start_time), float(end)))
    det.sort(key=lambda t: t[1])

    gt = [(g["zone"], float(g["start"]), float(g["end"])) for g in groundtruth]
    matched = [False] * len(gt)
    latencies: list[float] = []
    tp = fp = 0

    for zone, s, en in det:
        hit = None
        for gi, (gz, gs, ge) in enumerate(gt):
            if matched[gi] or gz != zone:
                continue
            if _overlaps(s, en, gs, ge):
                hit = gi
                break
        if hit is None:
            fp += 1
        else:
            matched[hit] = True
            tp += 1
            latencies.append(abs(s - gt[hit][1]))

    fn = matched.count(False)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    minutes = max(video_seconds / 60.0, 1e-6)

    return EvalResult(
        tp=tp, fp=fp, fn=fn,
        precision=precision, recall=recall, f1=f1,
        false_alarms_per_min=fp / minutes,
        mean_detection_latency_s=(sum(latencies) / len(latencies)) if latencies else 0.0,
        video_minutes=minutes,
    )
