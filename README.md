# Construction Site — Restricted Zone Intrusion Detection

AI-based CCTV safety monitoring for construction sites, built entirely with
**classical computer vision**: MOG2 background subtraction for region
proposals, **HOG + linear SVM** for worker detection, centroid tracking for
identity, and polygon hazard zones with debounced, timed intrusion logic.

**No deep learning. No GPU. Runs on CPU at ~50 FPS on a 640×360 feed.**

---

## 1. Construction safety context

Struck-by incidents and falls into excavations are among the largest causes of
death on construction sites worldwide. Almost all of them share a precursor
that is visible on CCTV seconds before the incident: *a worker on foot inside a
zone they should not be in* — a live excavation, a swing radius around an
excavator, a crane load path, a trench edge.

Sites already have cameras. What they do not have is anything watching them.
A supervisor cannot stare at sixteen tiles for eight hours, and cloud-based
deep-learning analytics are often impossible on site: no reliable bandwidth,
no GPU, and real objections to streaming worker footage off-premises.

## 2. Problem statement

> Given a CCTV-style video of a construction site, detect workers on foot,
> determine when their **feet** enter predefined hazardous zones, and raise
> graded safety alerts — accurately enough to be trusted, cheaply enough to run
> on an edge box at the site office.

Constraints deliberately imposed on this project:

| Constraint | Consequence |
|---|---|
| No deep learning (no YOLO / TensorFlow / PyTorch) | HOG + linear SVM person detector |
| CPU only, edge-friendly | Detector runs only on motion proposals, not full frames |
| Must be stable for a live demo | Debouncing, box smoothing, presence memory, fallback demo clip |

## 3. Approach

```
 frame
   │
   ├─ 1. PRE-PROCESS      resize 640×360, timestamp, "CCTV LIVE FEED" overlay
   │
   ├─ 2. MOTION           MOG2 (detectShadows=True) → drop shadow pixels (127)
   │                      → threshold → open/close/dilate → contours
   │                      → candidate regions        [src/motion_detector.py]
   │
   ├─ 3. CLASSIFY         HOG + getDefaultPeopleDetector() run ONLY inside the
   │                      padded candidate crops, NMS + containment de-dup,
   │                      aspect-ratio (1.5–4.5) and area filters
   │                      → WORKER / NON_WORKER / UNKNOWN [src/human_classifier.py]
   │
   ├─ 4. TRACK            centroid association (distance gate + max_disappeared)
   │                      → stable "Worker ID n"              [src/tracker.py]
   │
   ├─ 5. ZONE TEST        FOOT POINT (x + w/2, y + h) vs polygon zones via
   │                      cv2.pointPolygonTest                [src/zone_manager.py]
   │
   └─ 6. INTRUSION        K-frame entry/exit debounce, duration timer,
                          risk-weighted severity, event log [src/intrusion_logic.py]
```

Only objects classified **WORKER** can raise an intrusion. NON_WORKER
(vehicles, moving material) and UNKNOWN blobs are drawn for transparency but
never trigger an alert.

### Why the foot point?
A worker standing at the edge of a trench has a bounding box whose *centre*
hovers over the pit while their feet are on solid ground — and vice versa when
they are inside it but leaning out. Ground-plane safety logic is only
defensible if it uses the ground-contact point, so every zone test in this
system uses `(x + w/2, y + h)`, drawn on screen as a small circle so a reviewer
can see exactly what was tested.

## 4. Features

- Upload `.mp4` / `.avi` CCTV footage through a Streamlit dashboard, or use the
  built-in generated demo clip.
- Timestamped CCTV-style overlay with a live "REC" indicator.
- Motion-proposal + HOG worker detection, three-way object classification.
- Centroid tracking with persistent **Worker ID n** labels.
- Polygon hazard zones from `config.yaml` with names and risk levels, drawn
  **red (HIGH) / orange (MEDIUM) / yellow (LOW)**, highlighted on breach.
- Debounced intrusion detection (K = 3 frames in, 3 frames out) — no flicker.
- Per-intrusion duration timer rendered on the worker's box:
  `Worker ID 2 | Excavation Zone | 4.2 sec`.
- Risk-weighted severity: **WARNING → ALERT → CRITICAL**, with a colour-coded
  banner (`CRITICAL: Worker in high-risk zone`).
- Live KPIs: total violations, active intrusions, workers tracked, zone-wise
  violation chart, full event log, CSV / MP4 / JSON export.
- Metrics (precision, recall, F1, false alarms per minute, latency) in the
  sidebar and from the CLI.

### Severity model

`effective_exposure = duration × zone_risk_weight`

| Effective exposure | Severity | Colour | Message |
|---|---|---|---|
| &lt; 3 s | WARNING | yellow | WARNING: Worker near hazardous zone |
| 3–6 s | ALERT | orange | ALERT: Worker entered restricted zone |
| &gt; 6 s | CRITICAL | red | CRITICAL: Worker in high-risk zone |

Weights: `LOW 1.0`, `MEDIUM 1.5`, `HIGH 2.0`. A worker standing in a HIGH-risk
excavation therefore reaches CRITICAL in 3 seconds, while the same dwell time
in a LOW-risk storage area is still only a WARNING. The logged severity for an
event is the worst level it reached.

## 5. Innovation

Two safety-motivated ideas go beyond the standard "detect + point-in-polygon"
recipe. Both exist because the *failure modes of background subtraction are
most dangerous exactly where the danger is*.

### 5.1 Zone-aware adaptive background learning
A MOG2 model absorbs anything that stops moving. A worker who stands still at
the bottom of an excavation for twenty seconds is quietly learned into the
background and disappears from the system — the single worst possible failure
for a safety monitor.

Two MOG2 models are therefore run on the same frames:

| Model | Learning rate | Applies to |
|---|---|---|
| `bg_normal` | 0.005 | pixels **outside** hazard zones |
| `bg_hazard` | 0.00005 (near-frozen) | pixels **inside** hazard zones |

The final mask is composited from the two using the hazard-zone mask. The site
at large keeps adapting to sunlight, dust and swaying barriers; inside the
zones that can kill someone, a stationary worker stays in the foreground.

### 5.2 Presence memory
When a track stops being matched, a normal tracker drops it after
`max_disappeared` frames (12 here). But a worker who vanishes *while inside a
hazard zone* is far more likely to be occluded, crouching or bending down than
to have teleported out — and their safety timer must keep running.

So the survival budget is state-dependent: `max_disappeared` (12 frames)
outside a zone, `presence_memory_frames` (45 frames ≈ 2.2 s) inside one. The
ID, the intrusion event and the timer all survive the occlusion. The demo clip
deliberately walks a worker behind a stack of pipes while they are inside the
excavation zone; the overlay prints `PRESENCE MEMORY` while the track coasts,
and the single 19.5-second event is *not* split in two.

Supporting robustness measures: HOG-window padding is trimmed (the default
model returns a box ~15 % larger than the body, which drops the foot point
below the real ground contact), matched boxes are EMA-smoothed so the foot
point cannot hop across a zone boundary from jitter, and multi-scale duplicate
boxes are merged by an intersection-over-smaller test so one worker cannot
become two IDs.

## 6. Metrics

Measured on the bundled 28-second demo clip (560 frames @ 20 FPS, 3 workers,
3 zones, 1 moving vehicle as a distractor, 1 occlusion event), scored against
`eval/groundtruth_demo.json`. Reproduce with:

```bash
python main.py --groundtruth eval/groundtruth_demo.json
```

| Metric | Value |
|---|---|
| True positives / False positives / False negatives | 3 / 0 / 0 |
| **Precision** | **1.00** |
| **Recall** | **1.00** |
| **F1 score** | **1.00** |
| False alarms per minute | 0.00 |
| Alert latency (GT entry → alert raised) | 0.13 s |
| Processing latency | 19.5 ms/frame avg, 34.5 ms p95 |
| Throughput | ≈ 51 FPS, single CPU core, 640×360 |

**Read these honestly.** The demo clip is synthetic, which is what makes exact
frame-level ground truth possible, but it is also clean: no rain, no crowding,
no camera shake, no heavy occlusion between workers. On real site footage,
expect a HOG-based pipeline to land nearer **0.75–0.85 precision and
0.6–0.75 recall**, with recall dropping fastest for crouching, partially
occluded, or far-field workers, and precision dropping around high-contrast
vertical objects (barrier posts, ladders). The measured *latency* numbers
transfer far better than the accuracy numbers — those are a property of the
algorithm, not the clip.

Scoring rule: a detected event is a true positive if it names the same zone as
an unconsumed ground-truth interval and overlaps it in time. Each ground-truth
interval can be matched once, so re-triggering on the same breach counts
against precision — which is the metric that actually matters for an alarm
nobody should learn to ignore.

To score your own footage, annotate intervals in the same JSON shape:

```json
{"intrusions": [{"zone": "Excavation Zone", "start": 4.5, "end": 11.0}]}
```

## 7. How to run

```bash
# 1. environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. build the fallback demo clip + its ground truth
python main.py --make-demo
python eval/make_groundtruth.py

# 3a. dashboard (recommended for the demo)
streamlit run app.py

# 3b. or command line
python main.py --video videos/demo_construction.mp4 \
               --groundtruth eval/groundtruth_demo.json
```

Outputs land in `outputs/`: `annotated.mp4` (plus an `_h264` copy if ffmpeg is
installed), `events.csv`, `report.json`.

Useful CLI flags: `--show` (live preview window), `--limit N` (first N frames),
`--output`, `--csv`, `--no-h264`.

### Defining your own zones
Edit the `zones:` block in `config.yaml`. Points are **normalised** (0–1), so a
zone definition survives any change of resolution:

```yaml
zones:
  - name: "Excavation Zone"
    risk: "HIGH"
    points: [[0.03, 0.52], [0.34, 0.46], [0.44, 0.99], [0.02, 0.99]]
```

Grab coordinates by pausing a frame of your own footage and dividing pixel
positions by the frame width/height.

## 8. Project structure

```
project/
├── app.py                     # Streamlit dashboard
├── main.py                    # CLI runner + reporting
├── config.yaml                # every threshold, zone and parameter
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── motion_detector.py     # MOG2 + zone-aware adaptive background
│   ├── human_classifier.py    # HOG+SVM worker / non-worker / unknown
│   ├── tracker.py             # centroid tracking + presence memory
│   ├── zone_manager.py        # polygons, foot-point tests, rendering
│   ├── intrusion_logic.py     # debounce, timers, severity, event log
│   ├── pipeline.py            # orchestration + overlay drawing
│   ├── evaluation.py          # precision / recall / F1 / latency
│   ├── video_utils.py         # IO + H.264 re-encode
│   └── demo_video.py          # fallback demo clip generator
├── eval/
│   ├── make_groundtruth.py
│   └── groundtruth_demo.json
├── videos/                    # input footage (gitignored)
└── outputs/                   # annotated video, CSV, report (gitignored)
```

## 9. Limitations

Stated plainly, because a safety system that oversells itself is worse than
none:

1. **HOG detects upright, mostly-unoccluded pedestrians.** Crouching, kneeling,
   bending or prone workers — common on a real site, and often the highest-risk
   postures — are frequently missed. This system flags intrusions; it is not a
   fall detector.
2. **No re-identification.** If a worker leaves the frame and returns, they get
   a new ID, and the intrusion history does not follow them.
3. **Single fixed camera, single ground plane.** Zones are drawn in image
   space, so the camera must not move. A PTZ camera, or a knocked tripod,
   invalidates every zone until they are redrawn.
4. **No homography / depth.** A worker far behind a zone can project onto it in
   the image. Zones should be drawn conservatively, and a real deployment would
   calibrate a ground-plane homography.
5. **Motion-gated detection.** A worker who is perfectly still *outside* a
   hazard zone eventually fades into the background (inside zones, the
   near-frozen model prevents this — see §5.1).
6. **Sensitive to lighting and weather.** Hard shadows, rain, dust and night
   IR footage all degrade MOG2. Shadow suppression helps, it does not solve it.
7. **Not a certified safety device.** This is decision *support* — an aid to a
   supervisor, an audit trail, and a source of near-miss statistics. It must
   never replace physical barriers, permits, banksmen or exclusion procedures.
8. **Metrics above are from a synthetic clip.** Re-measure on your own site
   footage before quoting any number to anyone.

## 10. Publishing to GitHub (manual)

Nothing is pushed automatically. Create an empty repository on GitHub first,
then:

```bash
cd project
git init
git add .
git commit -m "Construction site restricted zone intrusion detection (classical CV)"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

`videos/` and `outputs/` are gitignored, so clone users regenerate the demo
clip with `python main.py --make-demo`. If you want to ship a sample clip
anyway, force-add it: `git add -f videos/demo_construction.mp4`.
