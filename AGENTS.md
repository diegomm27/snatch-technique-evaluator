# AGENTS.md

## Project Overview

**Snatch Technique Evaluator** is a Python toolkit for analyzing Olympic weightlifting snatch technique from side-view videos. It uses YOLO pose estimation to track the lifter's body and optical-flow barbell tracking to generate annotated videos with real-time bar-path graphs.

## Architecture

```
snatch_technique_evaluator/
├── __main__.py           # Entry point: python -m snatch_technique_evaluator
├── app.py                # Tkinter desktop launcher + CLI argument parsing
├── analysis.py           # Core pipeline: tracking, phase detection, rendering, scoring
├── tracking.py           # BarbellTracker (optical flow + template matching)
├── pose_backends/
│   ├── base.py           # PoseBackend abstract base + PoseResult dataclass
│   └── yolo_backend.py   # Ultralytics YOLO wrapper
└── references/
    └── default_reference.json  # Default scoring reference profile
```

## Key Components

### Pose Estimation (`pose_backends/`)
- `PoseBackend` — abstract base for pose inference
- `YoloPoseBackend` — wraps Ultralytics YOLO11 (nano or extra-large variants)
- Returns 12 COCO-format keypoints (shoulders, elbows, wrists, hips, knees, ankles)

### Barbell Tracking (`tracking.py`)
- `BarbellTracker` — tracks barbell using `cv2.calcOpticalFlowPyrLK` with template matching fallback
- Auto-reclick prompt after 5 lost frames
- Returns smoothed point with confidence score

### Analysis Pipeline (`analysis.py`)
1. **Phase detection** — detects setup, first pull, second pull, turnover, catch, recovery, finish from bar velocity, hip height, and extension score
2. **Kinematic computation** — torso angle, knee/hip/elbow angles, bar offset, foot displacement
3. **Lift state machine** — tracks live state transitions and completion detection
4. **Scoring** — compares normalized lift features against reference profile using z-scores with tolerance floors
5. **Rendering** — upscales video, draws skeleton overlay, bar path, bar-path chart, phase labels, and score report

### Entry Points
- **GUI**: `python -m snatch_technique_evaluator` or `launch` subcommand
- **CLI**: `python -m snatch_technique_evaluator analyze --video <path>`

## Dependencies

- `ultralytics` — YOLO11 pose estimation
- `opencv-python` — video I/O, optical flow, drawing
- `numpy` — array math and smoothing
- `ttkbootstrap` — desktop UI theme

## Development

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Linting: `ruff check .`

## Output Artifacts

All outputs go to `outputs/<timestamp>/`:
- `annotated.mp4` — upscaled video with pose overlay, bar-path graph, phase markers
- `frames.csv` — per-frame data (bar position, angles, phase, confidence, warnings)
- `metrics.json` — summary with phase frame ranges, score, warnings, catch metrics

## Important Notes

- Expects **side-view** snatch videos
- Requires one initial barbell click to seed the tracker
- Auto-detects CUDA GPU, falls back to CPU silently
- Auto-pauses on stable recovery position
- Scoring uses z-scores against reference profile with tolerance floors to avoid over-penalizing pose jitter
- The reference profile building pipeline (`build_reference_profile`, `discover_reference_videos`) is available in `analysis.py` but not exposed via CLI or UI
