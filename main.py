#!/usr/bin/env python3
"""
main.py - command line runner for the Construction Site Restricted Zone
Intrusion Detection system.

Examples
--------
    # build the fallback demo clip
    python main.py --make-demo

    # process a video, write an annotated MP4 + CSV event log
    python main.py --video videos/demo_construction.mp4

    # also score against ground truth
    python main.py --video videos/demo_construction.mp4 \
                   --groundtruth eval/groundtruth_demo.json

    # live preview window (needs a desktop session)
    python main.py --video videos/site.mp4 --show
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import cv2

from src.demo_video import generate_demo_video
from src.evaluation import evaluate, load_groundtruth
from src.pipeline import SafetyPipeline, load_config
from src.video_utils import make_writer, open_video, to_h264, video_info

DEFAULT_DEMO = "videos/demo_construction.mp4"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Construction site intrusion detection")
    p.add_argument("--video", default=None, help="input .mp4/.avi (default: demo clip)")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--output", default="outputs/annotated.mp4")
    p.add_argument("--csv", default="outputs/events.csv")
    p.add_argument("--groundtruth", default=None, help="JSON ground truth for metrics")
    p.add_argument("--make-demo", action="store_true", help="(re)generate the demo clip and exit")
    p.add_argument("--show", action="store_true", help="show a live preview window")
    p.add_argument("--limit", type=int, default=0, help="stop after N frames (0 = all)")
    p.add_argument("--no-h264", action="store_true", help="skip ffmpeg re-encode")
    return p.parse_args(argv)


def run(args) -> int:
    if args.make_demo:
        path = generate_demo_video(DEFAULT_DEMO)
        print(f"Demo video written to {path}")
        return 0

    video = args.video or DEFAULT_DEMO
    if not os.path.exists(video):
        print(f"[i] {video} not found - generating the fallback demo clip.")
        video = generate_demo_video(DEFAULT_DEMO)

    cfg = load_config(args.config)
    cap = open_video(video)
    info = video_info(cap, cfg["video"].get("fallback_fps", 20))
    pipe = SafetyPipeline(cfg, fps=info["fps"])

    os.makedirs("outputs", exist_ok=True)
    writer = make_writer(args.output, info["fps"], (pipe.width, pipe.height))

    print(f"Processing {video}  ({info['frames']} frames @ {info['fps']:.1f} fps)")
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        vis = pipe.process(frame)
        writer.write(vis)
        n += 1
        if args.show:
            cv2.imshow("Construction Site Safety Monitor", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if n % 100 == 0:
            print(f"  frame {n}  violations={pipe.monitor.total_violations} "
                  f"active={pipe.monitor.active_count}")
        if args.limit and n >= args.limit:
            break

    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()
    pipe.finalise()

    out_path = args.output
    if not args.no_h264:
        out_path = to_h264(args.output)

    # ---- event log -----------------------------------------------------
    rows = pipe.monitor.log_rows()
    if rows:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    stats = pipe.runtime_stats()
    report = {"video": video, "runtime": stats,
              "zone_violations": pipe.monitor.zone_counts(),
              "severity": pipe.monitor.severity_counts()}

    # ---- metrics -------------------------------------------------------
    if args.groundtruth and os.path.exists(args.groundtruth):
        res = evaluate(pipe.monitor.events, load_groundtruth(args.groundtruth),
                       pipe.video_time)
        report["metrics"] = res.as_dict()

    with open("outputs/report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    # ---- console summary ------------------------------------------------
    print("\n================ SAFETY REPORT ================")
    print(f"Annotated video : {out_path}")
    print(f"Event log       : {args.csv if rows else '(no events)'}")
    print(f"Frames          : {stats['frames']}  ({stats['video_seconds']} s of video)")
    print(f"Avg latency     : {stats['avg_latency_ms']} ms/frame "
          f"({stats['processing_fps']} FPS on CPU)")
    print(f"Violations      : {stats['total_violations']}  "
          f"({stats['alerts_per_minute']} per minute)")
    print(f"By zone         : {pipe.monitor.zone_counts()}")
    print(f"By severity     : {pipe.monitor.severity_counts()}")
    if "metrics" in report:
        print("---- Accuracy vs ground truth ----")
        for k, v in report["metrics"].items():
            print(f"{k:28s}: {v}")
    print("\nEvent log:")
    for r in rows:
        print(f"  {r['Timestamp']} | Worker ID {r['Worker ID']} | {r['Zone']} "
              f"| {r['Duration (s)']} sec | {r['Severity']}")
    print("===============================================")
    return 0


if __name__ == "__main__":
    sys.exit(run(parse_args()))
