#!/usr/bin/env python3
"""
eval/make_groundtruth.py
------------------------
Derives ground-truth intrusion intervals for the synthetic demo clip.

The demo workers follow scripted waypoints, so for every frame we know exactly
where each worker's FEET are. Replaying that script against the configured
zone polygons gives an exact, non-hand-waved ground truth:

    python eval/make_groundtruth.py

writes ``eval/groundtruth_demo.json``.

For real CCTV footage you would annotate intervals by hand in the same format.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.demo_video import FPS, PATHS, _interp          # noqa: E402
from src.pipeline import load_config                     # noqa: E402
from src.zone_manager import ZoneManager                 # noqa: E402

MIN_DURATION = 0.3   # ignore sub-debounce grazes


def main(seconds: float = 28.0, out: str = "eval/groundtruth_demo.json"):
    cfg = load_config("config.yaml")
    zm = ZoneManager(cfg, cfg["video"]["target_width"],
                     cfg["video"]["target_height"])

    intervals = []
    for wi, p in enumerate(PATHS, start=1):
        current, start = None, 0.0
        for i in range(int(seconds * FPS) + 1):
            t = i / FPS
            zone = None
            if p["t0"] <= t <= p["t1"]:
                u = (t - p["t0"]) / max(p["t1"] - p["t0"], 1e-6)
                foot = _interp(p["waypoints"], u)
                z = zm.zone_at(foot)
                zone = z.name if z else None
            if zone != current:
                if current is not None and t - start >= MIN_DURATION:
                    intervals.append({"worker": wi, "zone": current,
                                      "start": round(start, 2),
                                      "end": round(t, 2)})
                current, start = zone, t
        if current is not None:
            intervals.append({"worker": wi, "zone": current,
                              "start": round(start, 2),
                              "end": round(seconds, 2)})

    intervals.sort(key=lambda d: d["start"])
    data = {"video": "videos/demo_construction.mp4",
            "fps": FPS, "duration_s": seconds, "intrusions": intervals}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    print(f"Wrote {out} with {len(intervals)} ground-truth intrusions:")
    for iv in intervals:
        print(f"  worker {iv['worker']} | {iv['zone']:18s} | "
              f"{iv['start']:6.2f} -> {iv['end']:6.2f} s")


if __name__ == "__main__":
    main()
