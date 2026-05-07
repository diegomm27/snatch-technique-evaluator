# Snatch Technique Evaluator

A Python toolkit for analyzing Olympic weightlifting snatch technique from side-view videos. It uses **YOLO pose estimation** to track the lifter's body and **optical-flow barbell tracking** to generate an annotated video with a real-time bar-path graph.

---

## What it does

1. **Pose estimation** — Runs [Ultralytics YOLO](https://docs.ultralytics.com) on every frame to detect the lifter's keypoints (shoulders, hips, knees, ankles, wrists, elbows).
2. **Barbell tracking** — After an initial click on the barbell, it tracks the bar's center through the lift using optical flow and template matching.
3. **Phase detection** — Automatically detects lift phases (setup, first pull, second pull, turnover, catch, recovery, finish) from body kinematics.
4. **Bar-path graph** — Renders a fixed bar-path chart directly on the video, showing the bar's horizontal and vertical displacement relative to the starting position.
5. **Annotated output** — Exports an upscaled annotated video (`annotated.mp4` or `.avi`), a per-frame CSV (`frames.csv`), and a metrics JSON (`metrics.json`).

---

## How it works

### YOLO pose backend
The app uses **Ultralytics YOLO** (`yolo11n-pose.pt` by default) as its pose estimator. YOLO runs inference on each frame and returns 17 COCO-format keypoints. From these keypoints the analyzer computes:

- Torso angle
- Knee and hip angles (left / right)
- Bar horizontal offset from the lifter's midline
- Foot displacement

### Model variants
You can choose between two YOLO11 pose models in the UI:

| Model | Speed | Accuracy | Best for |
|---|---|---|---|
| `yolo11n-pose.pt` (default) | Fast | Good | Most users, real-time preview |
| `yolo11x-pose.pt` | Slow | Best | Maximum accuracy, post-processing analysis |

The nano variant (`n`) is selected by default because it runs smoothly on both GPU and CPU. The extra-large variant (`x`) is the most accurate but significantly slower — useful when you want the best possible keypoint detection.

### Tracking pipeline
```
Video frame
    │
    ▼
YOLO pose inference ──► Body keypoints
    │
    ▼
Barbell tracker (optical flow + template match)
    │
    ▼
FrameRecord (pose + bar position + computed metrics)
    │
    ▼
Phase detection ──► Bar-path chart rendering ──► Annotated video frame
```

### Auto-pause
When the analyzer detects a stable recovery position, playback automatically pauses so you can review the lift and bar path without missing the end.

---

## Setup

Requires **Python 3.11+**.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

The first run will automatically download the default YOLO model (`yolo11n-pose.pt`) via Ultralytics.

### GPU support (NVIDIA CUDA)

YOLO inference is much faster on GPU. The app auto-detects CUDA and uses it automatically. If your GPU is incompatible or drivers are missing, it **silently falls back to CPU** so the app always works.

**1. Check your GPU and CUDA version:**
```powershell
nvidia-smi
```

**2. Install the correct PyTorch build for your GPU:**

| GPU Generation | Examples | PyTorch Command |
|---|---|---|
| RTX 50-series (Blackwell) | RTX 5070, 5080, 5090 | `pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128` |
| RTX 40/30-series (Ada/Ampere) | RTX 4090, 3090 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126` |
| RTX 20/GTX 16-series (Turing) | RTX 2080, GTX 1660 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126` |
| Older (Pascal / Maxwell) | GTX 1080, 980 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126` |

> **RTX 50-series note:** Blackwell GPUs (sm_120) require PyTorch 2.7+ with CUDA 12.8. As of early 2025 this is only available in [nightly builds](https://download.pytorch.org/whl/nightly/cu128). If the nightly install fails, the app will still work on CPU.

**3. Verify it works:**
```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```
You should see a version like `2.6.0+cu126` and `True NVIDIA GeForce RTX ...`.

**4. Install the package:**
```powershell
pip install -e .
```

The app now automatically picks `cuda:0` when it works, otherwise falls back to CPU with a one-line warning.

---

## Usage

### Desktop launcher (recommended)

```powershell
python -m snatch_technique_evaluator
```

1. Click **Browse Video** and select a side-view snatch clip.
2. Pick the YOLO model variant:
   - `yolo11n-pose.pt` — fast, good accuracy (default)
   - `yolo11x-pose.pt` — slower, best accuracy
3. Click **Track Bar Path**.
4. Click the barbell center in the first frame.
5. The video plays in **fullscreen** with the bar-path graph overlaid. It auto-pauses when recovery is stable.
6. Output details (folder path, warnings, completion frame) are printed in the console log below the button.

### CLI

```powershell
# Launch GUI
python -m snatch_technique_evaluator launch

# Analyze a video directly
python -m snatch_technique_evaluator analyze --video "C:\path\to\lift.mp4"
```

---

## Outputs

All artifacts are written to `outputs/<timestamp>/`:

| File | Description |
|------|-------------|
| `annotated.mp4` / `annotated.avi` | Upscaled video with pose overlay, bar-path graph, and phase markers |
| `frames.csv` | Per-frame data: bar position, angles, phase, confidence, warnings |
| `metrics.json` | Summary: phase frame ranges, score, warnings, catch metrics |

The bar-path graph in the video is:
- **Fixed to the right edge** (does not follow the lifter)
- **Zoomed in** on realistic bar displacement ranges
- **Clean** — no titles, labels, or footers, just the path, grid, and axis guides

---

## Project structure

```
snatch_technique_evaluator/
├── __main__.py           # Entry point: python -m snatch_technique_evaluator
├── app.py                # Tkinter desktop launcher + CLI argument parsing
├── analysis.py           # Core pipeline: tracking, phase detection, rendering
├── tracking.py           # BarbellTracker (optical flow / template matching)
└── pose_backends/
    ├── base.py           # PoseBackend abstract base + PoseResult
    └── yolo_backend.py   # Ultralytics YOLO wrapper
```

---

## Dependencies

- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLO pose estimation
- [OpenCV](https://opencv.org/) — Video I/O, optical flow, drawing
- [NumPy](https://numpy.org/) — Array math and smoothing
- [ttkbootstrap](https://github.com/israel-dryer/ttkbootstrap) — Desktop UI theme

---

## Notes

- The analyzer expects a **side-view** snatch video for accurate bar-path and angle measurements.
- An initial barbell click is required to seed the tracker.
- Auto-pause and device selection are automatic — no manual configuration needed.

## License

MIT
