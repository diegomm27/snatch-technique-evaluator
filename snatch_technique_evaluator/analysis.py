from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .pose_backends import PoseBackend, PoseResult, YoloPoseBackend
from .tracking import BarbellTracker, BarTrackState

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = "1.0"
REFERENCE_SAMPLE_COUNT = 25
SKELETON_EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]
PHASES = [
    "setup",
    "first_pull",
    "second_pull",
    "turnover",
    "catch",
    "recovery",
    "finish",
]
SCORE_LABELS = {
    "bar_path": "Bar Path",
    "torso_hip": "Torso And Hip",
    "turnover_catch": "Turnover And Catch",
    "overhead_recovery": "Overhead And Recovery",
    "symmetry_footwork": "Symmetry And Footwork",
}
SCORING_WEIGHTS = {
    "bar_path": 31.25,
    "torso_hip": 18.75,
    "turnover_catch": 25.0,
    "overhead_recovery": 18.75,
    "symmetry_footwork": 6.25,
}
FEATURE_TOLERANCE_FLOORS = {
    "bar_path_x_norm": 0.08,
    "bar_path_y_norm": 0.10,
    "hip_height_norm": 0.08,
    "shoulder_height_norm": 0.08,
    "torso_angle_norm": 0.08,
    "knee_angle_mean_deg": 8.0,
    "hip_angle_mean_deg": 8.0,
    "elbow_angle_mean_deg": 10.0,
    "overhead_alignment_norm": 0.10,
    "left_right_knee_angle_diff_deg": 8.0,
    "left_right_hip_angle_diff_deg": 8.0,
    "foot_displacement_norm": 0.08,
    "foot_displacement_px": 12.0,
}
THRESHOLDS = {
    "tracking_reclick_frames": 5,
    "bar_far_from_body_px": 60.0,
    "hips_rise_ratio": 1.35,
    "overhead_alignment_px": 55.0,
    "asymmetry_angle_deg": 12.0,
    "completion_stable_frames": 15,
    "completion_bar_velocity_px": 2.5,
    "completion_hip_velocity_px": 2.5,
    "catch_drop_ratio": 0.08,
    "recovery_rise_ratio": 0.05,
    "overhead_margin_px": 18.0,
    "missing_frame_ratio_low_confidence": 0.35,
}
MIN_RENDER_HEIGHT = 1080
CHART_GRID_LINES = 6
CHART_SMOOTHING_WINDOW = 7
CHART_DENSIFY_STEPS = 6
CHART_WIDTH = 300
CHART_HEIGHT = 360
CHART_USER_OFFSET_X = 72
CHART_MARGIN = 16
CHART_X_RANGE_PX = 65.0
CHART_Y_RANGE_PX = 434.0


@dataclass(slots=True)
class AnalyzerConfig:
    video_path: Path
    pose_backend_name: str
    model_name: str
    device: str
    output_dir: Path | None = None
    reference_path: Path | None = None
    auto_pause: bool = False
    pause_mode: str = "after-recovery-stable"
    interactive: bool = True
    show_live_window: bool = True
    persist_outputs: bool = True
    initial_point: tuple[int, int] | None = None
    point_provider: Callable[[np.ndarray, str], tuple[int, int] | None] | None = None
    progress_callback: Callable[[dict[str, Any]], None] | None = None


@dataclass(slots=True)
class FrameRecord:
    frame_index: int
    timestamp_s: float
    pose_confidence: float
    bar_x: float | None
    bar_y: float | None
    bar_confidence: float
    bar_fallback_used: bool
    ankle_mid_x: float | None
    ankle_mid_y: float | None
    shoulder_mid_x: float | None
    shoulder_mid_y: float | None
    hip_mid_x: float | None
    hip_mid_y: float | None
    bar_horizontal_offset: float | None
    torso_angle_deg: float | None
    left_knee_angle_deg: float | None
    right_knee_angle_deg: float | None
    left_hip_angle_deg: float | None
    right_hip_angle_deg: float | None
    left_elbow_angle_deg: float | None
    right_elbow_angle_deg: float | None
    extension_score: float | None
    left_right_knee_angle_diff_deg: float | None
    left_right_hip_angle_diff_deg: float | None
    left_right_shoulder_y_diff_px: float | None
    foot_displacement_px: float | None
    body_scale_px: float | None = None
    foot_displacement_norm: float | None = None
    bar_path_x_norm: float | None = None
    bar_path_y_norm: float | None = None
    hip_height_norm: float | None = None
    shoulder_height_norm: float | None = None
    torso_angle_norm: float | None = None
    knee_angle_mean_deg: float | None = None
    hip_angle_mean_deg: float | None = None
    elbow_angle_mean_deg: float | None = None
    overhead_alignment_norm: float | None = None
    phase: str = "analyzing"
    live_state: str = "waiting_for_setup"
    normalized_phase_progress: float | None = None
    global_progress: float | None = None
    reference_deviation: float | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PhaseMarkers:
    setup_end: int
    first_pull_start: int
    second_pull_start: int
    turnover_start: int
    catch_frame: int
    recovery_start: int
    finish_frame: int


@dataclass(slots=True)
class CompletionStatus:
    completion_frame: int | None = None
    completion_reason: str | None = None
    stable_frames: int = 0
    auto_paused: bool = False


@dataclass(slots=True)
class AnalysisArtifacts:
    fps: float
    width: int
    height: int
    frame_count: int
    processed_frames: int
    records: list[FrameRecord]
    pose_dump: list[dict[str, list[float] | None]]
    phases: PhaseMarkers
    completion: CompletionStatus
    warnings: list[str]
    score: dict[str, Any]
    reference_profile_path: str | None
    reference_deviations: list[dict[str, Any]]
    output_dir: Path
    aborted: bool


def resolve_device(requested_device: str) -> str:
    requested = requested_device.strip().lower()
    if requested == "auto":
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda"):
        try:
            import torch
        except ImportError:
            return "cpu"
        if not torch.cuda.is_available():
            return "cpu"
    return requested_device


def default_reference_path() -> Path:
    return Path(__file__).resolve().parent / "references" / "default_reference.json"


def make_output_dir(video_path: Path, output_root: Path | None = None) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base = output_root or (PROJECT_ROOT / "outputs")
    output_dir = base / f"{video_path.stem}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "snapshots").mkdir(parents=True, exist_ok=True)
    return output_dir


def open_annotated_video_writer(
    output_path: Path, fps: float, width: int, height: int
) -> tuple[cv2.VideoWriter, str, Path]:
    # Try MP4-friendly codecs first.  mp4v is the most broadly-available
    # codec that does not depend on external libraries such as openh264.
    for codec in ("mp4v", "avc1"):
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
            True,
        )
        if writer.isOpened():
            return writer, codec, output_path
        writer.release()

    # Fallback to AVI with XVID or MJPG if no MP4 codec works.
    avi_path = output_path.with_suffix(".avi")
    for codec in ("XVID", "MJPG"):
        writer = cv2.VideoWriter(
            str(avi_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
            True,
        )
        if writer.isOpened():
            return writer, codec, avi_path
        writer.release()

    raise RuntimeError(
        f"Unable to open annotated video writer for {output_path} (or {avi_path}) with codecs: mp4v, avc1, XVID, MJPG"
    )


def _round_even(value: float) -> int:
    rounded = int(round(value))
    return rounded if rounded % 2 == 0 else rounded + 1


def annotated_video_dimensions(source_width: int, source_height: int) -> tuple[int, int]:
    render_height = max(source_height, MIN_RENDER_HEIGHT)
    scale = render_height / max(source_height, 1)
    render_frame_width = _round_even(source_width * scale)
    return render_frame_width, render_height


def _render_layout(source_width: int, source_height: int) -> tuple[int, int, float]:
    render_height = max(source_height, MIN_RENDER_HEIGHT)
    scale = render_height / max(source_height, 1)
    render_frame_width = _round_even(source_width * scale)
    return render_frame_width, render_height, scale


def open_video(video_path: Path) -> tuple[cv2.VideoCapture, float, int, int, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    ok, first_frame = capture.read()
    if not ok or first_frame is None:
        capture.release()
        raise RuntimeError(f"Unable to read the first frame from: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or first_frame.shape[1])
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or first_frame.shape[0])
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return capture, fps, width, height, frame_count


def request_point_click(
    frame_bgr: np.ndarray,
    window_name: str = "Click barbell/plate center",
) -> tuple[int, int] | None:
    clicked: dict[str, tuple[int, int] | None] = {"point": None}
    display = frame_bgr.copy()
    instructions = "Click barbell/plate center. Enter/Space confirm, Q/Esc cancel."

    # Scale UI elements based on frame height so they look consistent at any resolution
    frame_h, frame_w = frame_bgr.shape[:2]
    base_h = 1080.0
    scale = max(0.6, frame_h / base_h)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.72 * scale
    thickness = max(1, int(round(2 * scale)))
    text_color = (235, 245, 255)
    accent_color = (0, 220, 120)
    panel_color = (18, 22, 28)

    def on_mouse(event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked["point"] = (x, y)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    try:
        while True:
            frame = display.copy()

            # Draw semi-transparent instruction panel
            (tw, th), _ = cv2.getTextSize(instructions, font, font_scale, thickness)
            pad_x = int(round(16 * scale))
            pad_y = int(round(12 * scale))
            margin = int(round(10 * scale))
            x1 = margin
            y1 = margin
            x2 = min(x1 + tw + pad_x * 2, frame_w - margin)
            y2 = y1 + th + pad_y * 2

            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), panel_color, -1)
            cv2.addWeighted(overlay, 0.78, frame, 0.22, 0.0, frame)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 90, 105), 1, cv2.LINE_AA)

            cv2.putText(
                frame,
                instructions,
                (x1 + pad_x, y2 - pad_y + int(th * 0.15)),
                font,
                font_scale,
                text_color,
                thickness,
                cv2.LINE_AA,
            )

            point = clicked["point"]
            if point is not None:
                r = max(6, int(round(8 * scale)))
                cv2.circle(frame, point, r, accent_color, thickness, cv2.LINE_AA)
                cv2.drawMarker(
                    frame,
                    point,
                    accent_color,
                    cv2.MARKER_CROSS,
                    max(14, int(round(18 * scale))),
                    thickness,
                    cv2.LINE_AA,
                )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                return None
            if key in (13, 32) and point is not None:
                return point
    finally:
        cv2.destroyWindow(window_name)


def _draw_exit_hint(canvas: np.ndarray) -> None:
    """Draw a small 'Press Q to exit' pill in the top-left corner."""
    h, w = canvas.shape[:2]
    base_h = 1080.0
    scale = max(0.5, h / base_h)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.58 * scale
    thickness = max(1, int(round(1.5 * scale)))
    text = "Press Q to exit"
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    pad_x = int(round(10 * scale))
    pad_y = int(round(6 * scale))
    margin = int(round(10 * scale))
    x1 = margin
    y1 = margin
    x2 = min(x1 + tw + pad_x * 2, w - margin)
    y2 = y1 + th + pad_y * 2

    overlay = canvas.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (12, 16, 22), -1)
    cv2.addWeighted(overlay, 0.80, canvas, 0.20, 0.0, canvas)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (64, 72, 84), 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        text,
        (x1 + pad_x, y2 - pad_y + int(th * 0.15)),
        font,
        font_scale,
        (200, 210, 220),
        thickness,
        cv2.LINE_AA,
    )


def create_pose_backend(backend_name: str, model_name: str, device: str) -> tuple[PoseBackend, str]:
    normalized = backend_name.lower()
    if normalized == "yolo":
        try:
            return YoloPoseBackend(model_name=model_name, device=device), device
        except Exception:
            if device.startswith("cuda"):
                import warnings

                warnings.warn(
                    f"CUDA failed for device '{device}', falling back to CPU.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return YoloPoseBackend(model_name=model_name, device="cpu"), "cpu"
            raise
    raise RuntimeError(f"Unsupported pose backend: {backend_name}")


def midpoint(
    first: tuple[float, float, float] | None,
    second: tuple[float, float, float] | None,
) -> tuple[float, float] | None:
    if first is None or second is None:
        return None
    return ((first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0)


def average_visible(
    first: tuple[float, float, float] | None,
    second: tuple[float, float, float] | None,
) -> tuple[float, float] | None:
    visible = [point for point in (first, second) if point is not None]
    if not visible:
        return None
    return (
        float(sum(point[0] for point in visible) / len(visible)),
        float(sum(point[1] for point in visible) / len(visible)),
    )


def angle_at_joint(
    a: tuple[float, float, float] | None,
    b: tuple[float, float, float] | None,
    c: tuple[float, float, float] | None,
) -> float | None:
    if a is None or b is None or c is None:
        return None

    ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=np.float32)
    bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=np.float32)
    norm_product = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if norm_product < 1e-6:
        return None

    cosine = float(np.clip(np.dot(ba, bc) / norm_product, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def torso_angle_from_vertical(
    hips: tuple[float, float] | None,
    shoulders: tuple[float, float] | None,
) -> float | None:
    if hips is None or shoulders is None:
        return None
    vector = np.array([shoulders[0] - hips[0], shoulders[1] - hips[1]], dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        return None
    up = np.array([0.0, -1.0], dtype=np.float32)
    cosine = float(np.clip(np.dot(vector / norm, up), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def phase_for_frame(index: int, phases: PhaseMarkers) -> str:
    if index < phases.first_pull_start:
        return "setup"
    if index < phases.second_pull_start:
        return "first_pull"
    if index < phases.turnover_start:
        return "second_pull"
    if index < phases.catch_frame:
        return "turnover"
    if index <= phases.catch_frame:
        return "catch"
    if index < phases.recovery_start:
        return "catch"
    if index < phases.finish_frame:
        return "recovery"
    return "finish"


def _fill_nan_series(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    indices = np.arange(len(values))
    mask = ~np.isnan(values)
    if not mask.any():
        return np.zeros_like(values)
    values[~mask] = np.interp(indices[~mask], indices[mask], values[mask])
    return values


def _first_sustained(condition: np.ndarray, start: int, length: int) -> int | None:
    if len(condition) == 0:
        return None
    streak = 0
    for index in range(start, len(condition)):
        if bool(condition[index]):
            streak += 1
            if streak >= length:
                return index - length + 1
        else:
            streak = 0
    return None


def detect_phases(records: list[FrameRecord]) -> PhaseMarkers:
    if not records:
        return PhaseMarkers(0, 0, 0, 0, 0, 0, 0)

    bar_y = np.array(
        [record.bar_y if record.bar_y is not None else np.nan for record in records],
        dtype=np.float32,
    )
    hip_y = np.array(
        [record.hip_mid_y if record.hip_mid_y is not None else np.nan for record in records],
        dtype=np.float32,
    )
    extension = np.array(
        [record.extension_score if record.extension_score is not None else np.nan for record in records],
        dtype=np.float32,
    )

    bar_y = _fill_nan_series(bar_y)
    hip_y = _fill_nan_series(hip_y)
    extension = _fill_nan_series(extension)

    upward_velocity = np.concatenate([[0.0], bar_y[:-1] - bar_y[1:]])
    acceleration = np.concatenate([[0.0], upward_velocity[1:] - upward_velocity[:-1]])
    velocity_threshold = max(1.0, float(np.percentile(upward_velocity, 70)))
    extension_threshold = max(140.0, float(np.percentile(extension, 70)))
    accel_threshold = max(0.5, float(np.percentile(acceleration, 75)))

    first_pull_start = _first_sustained(upward_velocity > velocity_threshold * 0.45, start=1, length=3)
    if first_pull_start is None:
        first_pull_start = 0

    second_pull_start = _first_sustained(
        (upward_velocity > velocity_threshold) | (extension > extension_threshold) | (acceleration > accel_threshold),
        start=first_pull_start + 1,
        length=2,
    )
    if second_pull_start is None:
        second_pull_start = min(len(records) - 1, first_pull_start + 3)

    bar_peak = int(np.argmin(bar_y))
    turnover_start = max(second_pull_start, bar_peak - 3)

    catch_search_start = min(len(records) - 1, bar_peak)
    catch_search_end = min(len(records), catch_search_start + max(6, len(records) // 8))
    catch_slice = hip_y[catch_search_start:catch_search_end]
    if catch_slice.size == 0:
        catch_frame = bar_peak
    else:
        catch_frame = catch_search_start + int(np.argmax(catch_slice))

    recovery_start = _first_sustained(
        (hip_y[:-1] - hip_y[1:]) > 0.5,
        start=min(len(records) - 2, catch_frame + 1),
        length=3,
    )
    if recovery_start is None:
        recovery_start = min(len(records) - 1, catch_frame + 4)

    finish_frame = min(len(records) - 1, recovery_start + max(5, len(records) // 10))
    return PhaseMarkers(
        setup_end=max(0, first_pull_start - 1),
        first_pull_start=first_pull_start,
        second_pull_start=second_pull_start,
        turnover_start=turnover_start,
        catch_frame=catch_frame,
        recovery_start=recovery_start,
        finish_frame=finish_frame,
    )


def build_warnings(records: list[FrameRecord], phases: PhaseMarkers) -> list[str]:
    warnings: list[str] = []
    if not records:
        return warnings

    setup = records[0]
    first_pull_records = records[phases.first_pull_start : max(phases.second_pull_start, phases.first_pull_start + 1)]
    if first_pull_records and setup.hip_mid_y is not None and setup.shoulder_mid_y is not None:
        final_pull = first_pull_records[-1]
        if final_pull.hip_mid_y is not None and final_pull.shoulder_mid_y is not None:
            hip_rise = final_pull.hip_mid_y - setup.hip_mid_y
            shoulder_rise = final_pull.shoulder_mid_y - setup.shoulder_mid_y
            if abs(shoulder_rise) > 1e-3 and hip_rise / shoulder_rise > THRESHOLDS["hips_rise_ratio"]:
                warnings.append("Hips rising too early in the first pull.")

    post_second_pull = records[phases.second_pull_start : phases.catch_frame + 1]
    offsets = [
        abs(record.bar_horizontal_offset) for record in post_second_pull if record.bar_horizontal_offset is not None
    ]
    if offsets and max(offsets) > THRESHOLDS["bar_far_from_body_px"]:
        warnings.append("Bar loops away from the body after the second pull.")

    catch_record = records[phases.catch_frame]
    if (
        catch_record.bar_x is not None
        and catch_record.shoulder_mid_x is not None
        and catch_record.hip_mid_x is not None
        and catch_record.ankle_mid_x is not None
    ):
        body_line_x = np.mean(
            [
                catch_record.shoulder_mid_x,
                catch_record.hip_mid_x,
                catch_record.ankle_mid_x,
            ]
        )
        if abs(catch_record.bar_x - body_line_x) > THRESHOLDS["overhead_alignment_px"]:
            warnings.append("Overhead bar alignment is unstable at the catch.")

    if (
        catch_record.left_right_knee_angle_diff_deg is not None
        and catch_record.left_right_knee_angle_diff_deg > THRESHOLDS["asymmetry_angle_deg"]
    ):
        warnings.append("Large left/right knee asymmetry at the catch.")

    return warnings


def _dash_line(
    frame_bgr: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
    dash: int = 10,
    gap: int = 8,
) -> None:
    start_xy = np.array(start, dtype=np.float32)
    end_xy = np.array(end, dtype=np.float32)
    delta = end_xy - start_xy
    length = float(np.linalg.norm(delta))
    if length < 1e-6:
        return
    direction = delta / length
    cursor = 0.0
    while cursor < length:
        segment_start = start_xy + direction * cursor
        segment_end = start_xy + direction * min(length, cursor + dash)
        cv2.line(
            frame_bgr,
            tuple(np.round(segment_start).astype(int)),
            tuple(np.round(segment_end).astype(int)),
            color,
            thickness,
            cv2.LINE_AA,
        )
        cursor += dash + gap


def _smooth_curve(values: np.ndarray, window: int = CHART_SMOOTHING_WINDOW) -> np.ndarray:
    if len(values) <= 2 or window <= 1:
        return values
    if window % 2 == 0:
        window += 1
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def _densify_points(points: list[tuple[float, float]], steps: int = CHART_DENSIFY_STEPS) -> list[tuple[float, float]]:
    if len(points) <= 1:
        return points
    dense: list[tuple[float, float]] = []
    for start, end in zip(points, points[1:]):
        for step in range(steps):
            ratio = step / float(steps)
            dense.append(
                (
                    start[0] + (end[0] - start[0]) * ratio,
                    start[1] + (end[1] - start[1]) * ratio,
                )
            )
    dense.append(points[-1])
    return dense


def _chart_transform(
    path_points: list[tuple[int, int]],
    chart_left: int,
    chart_top: int,
    chart_right: int,
    chart_bottom: int,
) -> tuple[list[tuple[int, int]], int, int] | None:
    if not path_points:
        return None

    base_x = float(path_points[0][0])
    base_y = float(path_points[0][1])
    series = np.array(
        [(point[0] - base_x, base_y - point[1]) for point in path_points],
        dtype=np.float32,
    )
    xs = _smooth_curve(series[:, 0])
    ys = _smooth_curve(series[:, 1])
    smooth_points = _densify_points(list(zip(xs.tolist(), ys.tolist())))

    min_x = -CHART_X_RANGE_PX
    max_x = CHART_X_RANGE_PX
    min_y = -CHART_Y_RANGE_PX * 0.12
    max_y = CHART_Y_RANGE_PX
    range_x = max_x - min_x
    range_y = max_y - min_y

    chart_width = max(1, chart_right - chart_left)
    chart_height = max(1, chart_bottom - chart_top)

    def project(point: tuple[float, float]) -> tuple[int, int]:
        x_value, y_value = point
        x_ratio = float(np.clip((x_value - min_x) / range_x, 0.0, 1.0))
        y_ratio = float(np.clip((y_value - min_y) / range_y, 0.0, 1.0))
        x = chart_left + int(round(x_ratio * chart_width))
        y = chart_bottom - int(round(y_ratio * chart_height))
        return x, y

    projected = [project(point) for point in smooth_points]
    y_axis_x, x_axis_y = project((0.0, 0.0))
    return projected, y_axis_x, x_axis_y


def _draw_info_overlay(
    frame_bgr: np.ndarray,
    lines: list[str],
    scale: float,
) -> None:
    font_scale = max(0.72, 0.68 * scale)
    line_height = int(round(30 * max(scale, 1.0)))
    padding_x = int(round(18 * max(scale, 1.0)))
    padding_y = int(round(16 * max(scale, 1.0)))
    width = min(frame_bgr.shape[1] - 24, int(round(frame_bgr.shape[1] * 0.46)))
    box_height = padding_y * 2 + line_height * len(lines)

    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (12, 12), (12 + width, 12 + box_height), (22, 24, 28), -1)
    cv2.addWeighted(overlay, 0.70, frame_bgr, 0.30, 0.0, frame_bgr)
    cv2.rectangle(frame_bgr, (12, 12), (12 + width, 12 + box_height), (96, 104, 114), 1)

    for row, text in enumerate(lines):
        color = (92, 214, 138) if "Snatch complete" in text else (242, 245, 247)
        cv2.putText(
            frame_bgr,
            text,
            (
                12 + padding_x,
                12 + padding_y + (row + 1) * line_height - int(line_height * 0.28),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            max(1, int(round(2 * max(scale, 1.0)))),
            cv2.LINE_AA,
        )


def _draw_bar_path_chart(
    canvas: np.ndarray,
    path_points: list[tuple[int, int]],
    record: FrameRecord,
    scale_x: float,
    scale_y: float,
) -> None:
    # Fixed right-side position so the chart does not move with the lifter
    chart_left = canvas.shape[1] - CHART_WIDTH - CHART_MARGIN
    chart_top = CHART_MARGIN
    chart_right = chart_left + CHART_WIDTH
    chart_bottom = chart_top + CHART_HEIGHT

    overlay = canvas.copy()
    cv2.rectangle(overlay, (chart_left, chart_top), (chart_right, chart_bottom), (18, 23, 29), -1)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0.0, canvas)
    cv2.rectangle(
        canvas,
        (chart_left, chart_top),
        (chart_right, chart_bottom),
        (86, 96, 108),
        1,
        cv2.LINE_AA,
    )

    # No text / titles — plot fills nearly the whole box
    plot_left = chart_left + 10
    plot_top = chart_top + 10
    plot_right = chart_right - 10
    plot_bottom = chart_bottom - 10

    cv2.rectangle(canvas, (plot_left, plot_top), (plot_right, plot_bottom), (56, 64, 72), 1)
    for row in range(1, CHART_GRID_LINES):
        y = plot_top + int(round(((plot_bottom - plot_top) * row) / CHART_GRID_LINES))
        _dash_line(canvas, (plot_left, y), (plot_right, y), (54, 60, 68), 1)
    for column in range(1, CHART_GRID_LINES):
        x = plot_left + int(round(((plot_right - plot_left) * column) / CHART_GRID_LINES))
        _dash_line(canvas, (x, plot_top), (x, plot_bottom), (54, 60, 68), 1)

    transformed = _chart_transform(path_points, plot_left, plot_top, plot_right, plot_bottom)
    if transformed is not None:
        projected, y_axis_x, x_axis_y = transformed
        _dash_line(
            canvas,
            (plot_left, x_axis_y),
            (plot_right, x_axis_y),
            (122, 130, 138),
            2,
            dash=14,
            gap=8,
        )
        _dash_line(
            canvas,
            (y_axis_x, plot_top),
            (y_axis_x, plot_bottom),
            (122, 130, 138),
            2,
            dash=14,
            gap=8,
        )

        cv2.polylines(
            canvas,
            [np.array(projected, dtype=np.int32)],
            False,
            (255, 188, 72),
            3,
            cv2.LINE_AA,
        )
        cv2.circle(canvas, projected[0], 6, (137, 207, 240), -1, cv2.LINE_AA)
        cv2.circle(canvas, projected[-1], 7, (82, 220, 145), -1, cv2.LINE_AA)


def _draw_bar_path_on_frame(
    canvas: np.ndarray,
    path_points: list[tuple[int, int]],
    path_phases: list[str],
    scale_x: float,
    scale_y: float,
    render_scale: float,
) -> None:
    """Draw the tracked bar path directly on the video frame."""
    if len(path_points) < 2:
        return

    scaled = [(int(round(x * scale_x)), int(round(y * scale_y))) for x, y in path_points]
    pts = np.array(scaled, dtype=np.int32).reshape((-1, 1, 2))
    thickness = max(3, int(round(3 * render_scale)))
    glow = max(8, int(round(8 * render_scale)))

    # White glow for visibility against any background
    cv2.polylines(canvas, [pts], False, (255, 255, 255), glow, cv2.LINE_AA)
    # Bright neon-green path (BGR) matching the reference image style
    cv2.polylines(canvas, [pts], False, (0, 255, 80), thickness, cv2.LINE_AA)

    # Start dot
    cv2.circle(
        canvas,
        scaled[0],
        max(6, int(round(6 * render_scale))),
        (0, 255, 0),
        -1,
        cv2.LINE_AA,
    )
    # Current position dot
    cv2.circle(
        canvas,
        scaled[-1],
        max(8, int(round(8 * render_scale))),
        (0, 0, 255),
        -1,
        cv2.LINE_AA,
    )

    # Vertical dashed reference line at the starting x-position
    start_x = scaled[0][0]
    step = max(16, int(round(16 * render_scale)))
    for y in range(0, canvas.shape[0], step):
        y_end = min(y + step // 2, canvas.shape[0])
        cv2.line(
            canvas,
            (start_x, y),
            (start_x, y_end),
            (160, 160, 160),
            max(1, int(round(1 * render_scale))),
            cv2.LINE_AA,
        )

    # Phase labels at transitions
    if path_phases and len(path_phases) == len(path_points):
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = max(0.48, 0.48 * render_scale)
        thick = max(1, int(round(1 * render_scale)))
        shown: set[str] = set()
        for i in range(1, len(scaled)):
            if path_phases[i] != path_phases[i - 1]:
                phase = path_phases[i]
                if phase in shown or phase == "setup":
                    continue
                shown.add(phase)
                pt = scaled[i]
                label = phase.replace("_", " ").upper()
                (tw, th), _ = cv2.getTextSize(label, font, fs, thick)
                x1 = max(6, pt[0] - tw // 2 - 5)
                y1 = max(6, pt[1] - th - 10)
                x2 = min(canvas.shape[1] - 6, x1 + tw + 10)
                y2 = min(canvas.shape[0] - 6, y1 + th + 8)
                overlay = canvas.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (18, 18, 18), -1)
                cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
                cv2.putText(
                    canvas,
                    label,
                    (x1 + 5, y2 - 3),
                    font,
                    fs,
                    (235, 235, 235),
                    thick,
                    cv2.LINE_AA,
                )


def annotate_frame(
    frame_bgr: np.ndarray,
    pose: PoseResult | None,
    record: FrameRecord,
    phase_label: str,
    path_points: list[tuple[int, int]],
    global_warnings: list[str],
    score_text: str | None = None,
    completion_text: str | None = None,
    score_report: dict[str, Any] | None = None,
    path_phases: list[str] | None = None,
) -> np.ndarray:
    source_height, source_width = frame_bgr.shape[:2]
    render_width, render_height, render_scale = _render_layout(source_width, source_height)
    vis = cv2.resize(
        frame_bgr,
        (render_width, render_height),
        interpolation=cv2.INTER_LANCZOS4 if render_scale >= 1.0 else cv2.INTER_AREA,
    )
    scale_x = render_width / max(source_width, 1)
    scale_y = render_height / max(source_height, 1)
    line_thickness = max(2, int(round(2 * max(render_scale, 1.0))))
    point_radius = max(4, int(round(4 * max(render_scale, 1.0))))
    if pose is not None:
        for start_name, end_name in SKELETON_EDGES:
            start = pose.keypoints.get(start_name)
            end = pose.keypoints.get(end_name)
            if start is None or end is None:
                continue
            cv2.line(
                vis,
                (int(round(start[0] * scale_x)), int(round(start[1] * scale_y))),
                (int(round(end[0] * scale_x)), int(round(end[1] * scale_y))),
                (0, 220, 0),
                line_thickness,
                cv2.LINE_AA,
            )
        for keypoint in pose.keypoints.values():
            if keypoint is None:
                continue
            cv2.circle(
                vis,
                (int(round(keypoint[0] * scale_x)), int(round(keypoint[1] * scale_y))),
                point_radius,
                (0, 255, 255),
                -1,
                cv2.LINE_AA,
            )

    overlay_lines = [
        f"Phase: {phase_label}",
        f"Live state: {record.live_state}",
        f"Pose conf: {record.pose_confidence:.2f} | Bar conf: {record.bar_confidence:.2f}",
        f"Bar offset: {_fmt(record.bar_horizontal_offset)} px | Torso: {_fmt(record.torso_angle_deg)} deg",
        f"Knee L/R: {_fmt(record.left_knee_angle_deg)} / {_fmt(record.right_knee_angle_deg)}",
        f"Hip L/R: {_fmt(record.left_hip_angle_deg)} / {_fmt(record.right_hip_angle_deg)}",
    ]
    if score_text:
        overlay_lines.append(score_text)
    if completion_text:
        overlay_lines.append(completion_text)
    elif record.warnings:
        overlay_lines.append("Frame warning: " + " | ".join(record.warnings[:2]))
    elif global_warnings:
        overlay_lines.append("Session warning: " + " | ".join(global_warnings[:2]))
    canvas = vis
    _draw_bar_path_chart(canvas, path_points, record, scale_x, scale_y)

    if score_report and score_report.get("score") is not None:
        _draw_score_report(canvas, score_report)

    return canvas


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def _draw_score_report(frame_bgr: np.ndarray, score_report: dict[str, Any]) -> None:
    breakdown = score_report.get("score_breakdown_100", {})
    findings = score_report.get("findings", [])
    x2 = frame_bgr.shape[1] - 16
    y1 = 16
    width = min(320, frame_bgr.shape[1] - 32)
    row_height = 24
    box_height = 82 + len(breakdown) * row_height + min(2, len(findings)) * 22
    x1 = max(16, x2 - width)
    y2 = min(frame_bgr.shape[0] - 16, y1 + box_height)

    panel = frame_bgr.copy()
    cv2.rectangle(panel, (x1, y1), (x2, y2), (28, 20, 18), -1)
    cv2.addWeighted(panel, 0.72, frame_bgr, 0.28, 0.0, frame_bgr)
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (214, 185, 119), 1)

    overall_score = score_report.get("score")
    cv2.putText(
        frame_bgr,
        "Report",
        (x1 + 14, y1 + 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (245, 236, 222),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr,
        f"{overall_score:.1f}/100",
        (x2 - 138, y1 + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        _score_color(overall_score),
        2,
        cv2.LINE_AA,
    )

    cursor_y = y1 + 56
    for key, label in SCORE_LABELS.items():
        value = breakdown.get(key)
        if value is None:
            continue
        cv2.putText(
            frame_bgr,
            label,
            (x1 + 14, cursor_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (235, 228, 214),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            f"{value:.0f}",
            (x2 - 50, cursor_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            _score_color(value),
            1,
            cv2.LINE_AA,
        )
        cursor_y += row_height

    for finding in findings[:2]:
        if cursor_y > y2 - 8:
            break
        cv2.putText(
            frame_bgr,
            finding[:44],
            (x1 + 14, cursor_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (194, 181, 166),
            1,
            cv2.LINE_AA,
        )
        cursor_y += 22


def _score_color(score: float) -> tuple[int, int, int]:
    if score >= 85:
        return (92, 214, 138)
    if score >= 70:
        return (95, 201, 243)
    if score >= 50:
        return (84, 191, 255)
    return (90, 134, 255)


def save_snapshot(frame_bgr: np.ndarray, output_dir: Path) -> Path:
    snapshot_dir = output_dir / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    output_path = snapshot_dir / f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(str(output_path), frame_bgr)
    return output_path


def keypoint_array_to_dict(pose: PoseResult | None) -> dict[str, list[float] | None]:
    if pose is None:
        return {}
    return {
        name: None if point is None else [float(point[0]), float(point[1]), float(point[2])]
        for name, point in pose.keypoints.items()
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


class LiftStateMachine:
    def __init__(self, fps: float) -> None:
        self.fps = fps
        self.state = "waiting_for_setup"
        self.completion = CompletionStatus()
        self.baseline_bar_y: float | None = None
        self.baseline_hip_y: float | None = None
        self.min_bar_y: float | None = None
        self.catch_hip_y: float | None = None

    def update(self, records: list[FrameRecord], record: FrameRecord) -> CompletionStatus:
        if self.baseline_bar_y is None and record.bar_y is not None:
            self.baseline_bar_y = record.bar_y
        if self.baseline_hip_y is None and record.hip_mid_y is not None:
            self.baseline_hip_y = record.hip_mid_y

        previous = records[-2] if len(records) >= 2 else None
        bar_velocity = self._velocity(previous.bar_y if previous else None, record.bar_y)
        hip_velocity = self._velocity(previous.hip_mid_y if previous else None, record.hip_mid_y)
        overhead = self._is_overhead(record)

        if record.bar_y is not None:
            self.min_bar_y = record.bar_y if self.min_bar_y is None else min(self.min_bar_y, record.bar_y)

        if self.state == "waiting_for_setup":
            if bar_velocity > 1.0:
                self.state = "first_pull"
        elif self.state == "first_pull":
            if bar_velocity > 2.5 or (record.extension_score is not None and record.extension_score > 150.0):
                self.state = "second_pull"
        elif self.state == "second_pull":
            if self.min_bar_y is not None and record.bar_y is not None and record.bar_y - self.min_bar_y > 6.0:
                self.state = "turnover"
            elif overhead and self._hip_is_descending_from_baseline(record):
                self.state = "turnover"
        elif self.state == "turnover":
            if overhead and self._hip_is_descending_from_baseline(record):
                self.state = "catch"
                self.catch_hip_y = record.hip_mid_y
        elif self.state == "catch":
            if self.catch_hip_y is None and record.hip_mid_y is not None:
                self.catch_hip_y = record.hip_mid_y
            if overhead and self._hip_rising_from_catch(record):
                self.state = "recovery"
        elif self.state == "recovery":
            if (
                overhead
                and abs(bar_velocity) <= THRESHOLDS["completion_bar_velocity_px"]
                and abs(hip_velocity) <= THRESHOLDS["completion_hip_velocity_px"]
            ):
                self.completion.stable_frames += 1
            else:
                self.completion.stable_frames = 0

            if self.completion.completion_frame is None and self.completion.stable_frames >= int(
                THRESHOLDS["completion_stable_frames"]
            ):
                self.state = "complete"
                self.completion.completion_frame = record.frame_index
                self.completion.completion_reason = (
                    "catch detected, recovered to standing, and stable overhead support confirmed"
                )
        record.live_state = self.state
        return self.completion

    @staticmethod
    def _velocity(previous: float | None, current: float | None) -> float:
        if previous is None or current is None:
            return 0.0
        return previous - current

    def _hip_is_descending_from_baseline(self, record: FrameRecord) -> bool:
        if self.baseline_hip_y is None or record.hip_mid_y is None or record.body_scale_px is None:
            return False
        return (record.hip_mid_y - self.baseline_hip_y) >= record.body_scale_px * THRESHOLDS["catch_drop_ratio"]

    def _hip_rising_from_catch(self, record: FrameRecord) -> bool:
        if self.catch_hip_y is None or record.hip_mid_y is None or record.body_scale_px is None:
            return False
        return (self.catch_hip_y - record.hip_mid_y) >= record.body_scale_px * THRESHOLDS["recovery_rise_ratio"]

    def _is_overhead(self, record: FrameRecord) -> bool:
        if record.bar_y is None or record.shoulder_mid_y is None:
            return False
        if record.bar_x is None or record.shoulder_mid_x is None:
            return False
        margin = THRESHOLDS["overhead_margin_px"]
        return record.bar_y < (record.shoulder_mid_y - margin)


def finalize_record_features(records: list[FrameRecord], phases: PhaseMarkers) -> None:
    if not records:
        return

    total_frames = max(1, len(records) - 1)
    for record in records:
        record.phase = phase_for_frame(record.frame_index, phases)
        record.global_progress = float(record.frame_index / total_frames)

        start, end = phase_frame_bounds(phases, record.phase)
        phase_length = max(1, end - start)
        record.normalized_phase_progress = float(np.clip((record.frame_index - start) / phase_length, 0.0, 1.0))

        if record.body_scale_px is not None and record.body_scale_px > 1e-6:
            if record.foot_displacement_px is not None:
                record.foot_displacement_norm = float(record.foot_displacement_px / record.body_scale_px)
            if record.bar_horizontal_offset is not None:
                record.bar_path_x_norm = float(record.bar_horizontal_offset / record.body_scale_px)
            if record.bar_y is not None and record.ankle_mid_y is not None:
                record.bar_path_y_norm = float((record.ankle_mid_y - record.bar_y) / record.body_scale_px)
            if record.hip_mid_y is not None and record.ankle_mid_y is not None:
                record.hip_height_norm = float((record.ankle_mid_y - record.hip_mid_y) / record.body_scale_px)
            if record.shoulder_mid_y is not None and record.ankle_mid_y is not None:
                record.shoulder_height_norm = float((record.ankle_mid_y - record.shoulder_mid_y) / record.body_scale_px)
            if record.bar_x is not None and record.shoulder_mid_x is not None:
                record.overhead_alignment_norm = float((record.bar_x - record.shoulder_mid_x) / record.body_scale_px)
        if record.torso_angle_deg is not None:
            record.torso_angle_norm = record.torso_angle_deg / 90.0
        knee_values = [
            value for value in (record.left_knee_angle_deg, record.right_knee_angle_deg) if value is not None
        ]
        if knee_values:
            record.knee_angle_mean_deg = float(sum(knee_values) / len(knee_values))
        hip_values = [value for value in (record.left_hip_angle_deg, record.right_hip_angle_deg) if value is not None]
        if hip_values:
            record.hip_angle_mean_deg = float(sum(hip_values) / len(hip_values))
        elbow_values = [
            value for value in (record.left_elbow_angle_deg, record.right_elbow_angle_deg) if value is not None
        ]
        if elbow_values:
            record.elbow_angle_mean_deg = float(sum(elbow_values) / len(elbow_values))


def phase_frame_bounds(phases: PhaseMarkers, phase: str) -> tuple[int, int]:
    if phase == "setup":
        return 0, phases.first_pull_start
    if phase == "first_pull":
        return phases.first_pull_start, phases.second_pull_start
    if phase == "second_pull":
        return phases.second_pull_start, phases.turnover_start
    if phase == "turnover":
        return phases.turnover_start, phases.catch_frame
    if phase == "catch":
        return phases.catch_frame, phases.recovery_start
    if phase == "recovery":
        return phases.recovery_start, phases.finish_frame
    return phases.finish_frame, max(phases.finish_frame + 1, phases.finish_frame)


def load_reference_profile(reference_path: Path | None) -> dict[str, Any] | None:
    if reference_path is None or not reference_path.exists():
        return None
    profile = json.loads(reference_path.read_text(encoding="utf-8"))
    schema_version = profile.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise RuntimeError(f"Reference profile schema mismatch. Expected {SCHEMA_VERSION}, found {schema_version}.")
    return profile


def interpolate_series(values: list[float | None], sample_count: int) -> list[float | None]:
    if sample_count <= 0:
        return []
    numeric = np.array(
        [np.nan if value is None else float(value) for value in values],
        dtype=np.float32,
    )
    mask = ~np.isnan(numeric)
    if not mask.any():
        return [None] * sample_count
    if len(numeric) == 1:
        return [float(numeric[0])] * sample_count
    numeric = _fill_nan_series(numeric)
    source_x = np.linspace(0.0, 1.0, num=len(numeric))
    target_x = np.linspace(0.0, 1.0, num=sample_count)
    resampled = np.interp(target_x, source_x, numeric)
    return [float(value) for value in resampled]


def build_normalized_lift(records: list[FrameRecord], phases: PhaseMarkers) -> dict[str, Any]:
    phase_features: dict[str, dict[str, list[float | None]]] = {}
    phase_durations: dict[str, float] = {}
    feature_names = [
        "bar_path_x_norm",
        "bar_path_y_norm",
        "hip_height_norm",
        "shoulder_height_norm",
        "torso_angle_norm",
        "knee_angle_mean_deg",
        "hip_angle_mean_deg",
        "elbow_angle_mean_deg",
        "overhead_alignment_norm",
        "left_right_knee_angle_diff_deg",
        "left_right_hip_angle_diff_deg",
        "foot_displacement_norm",
        "foot_displacement_px",
    ]

    for phase in PHASES:
        start, end = phase_frame_bounds(phases, phase)
        slice_records = records[start : max(end + 1, start + 1)]
        phase_durations[phase] = float(len(slice_records) / max(1, len(records)))
        phase_features[phase] = {}
        for feature_name in feature_names:
            values = [getattr(record, feature_name) for record in slice_records]
            phase_features[phase][feature_name] = interpolate_series(values, REFERENCE_SAMPLE_COUNT)

    catch_record = records[phases.catch_frame]
    finish_record = records[phases.finish_frame]
    recovery_records = records[phases.recovery_start : phases.finish_frame + 1]
    recovery_bar_velocity = []
    recovery_hip_velocity = []
    for previous, current in zip(recovery_records, recovery_records[1:]):
        recovery_bar_velocity.append(
            abs((previous.bar_y or 0.0) - (current.bar_y or 0.0))
            if previous.bar_y is not None and current.bar_y is not None
            else None
        )
        recovery_hip_velocity.append(
            abs((previous.hip_mid_y or 0.0) - (current.hip_mid_y or 0.0))
            if previous.hip_mid_y is not None and current.hip_mid_y is not None
            else None
        )

    return {
        "phase_features": phase_features,
        "phase_durations": phase_durations,
        "catch_metrics": {
            "bar_path_x_norm": catch_record.bar_path_x_norm,
            "bar_path_y_norm": catch_record.bar_path_y_norm,
            "hip_height_norm": catch_record.hip_height_norm,
            "torso_angle_norm": catch_record.torso_angle_norm,
            "knee_angle_mean_deg": catch_record.knee_angle_mean_deg,
            "elbow_angle_mean_deg": catch_record.elbow_angle_mean_deg,
            "overhead_alignment_norm": catch_record.overhead_alignment_norm,
        },
        "recovery_stability": {
            "finish_overhead_alignment_norm": finish_record.overhead_alignment_norm,
            "bar_velocity_abs_mean": _nanmean(recovery_bar_velocity),
            "hip_velocity_abs_mean": _nanmean(recovery_hip_velocity),
        },
    }


def _nanmean(values: list[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return float(sum(numeric) / len(numeric))


def aggregate_reference_payload(
    successful_videos: list[dict[str, Any]],
    backend_name: str,
    model_name: str,
    device: str,
) -> dict[str, Any]:
    references = [video["normalized_lift"] for video in successful_videos]
    feature_names = list(references[0]["phase_features"]["setup"].keys())
    phase_features: dict[str, dict[str, dict[str, list[float | None]]]] = {}

    for phase in PHASES:
        phase_features[phase] = {}
        for feature_name in feature_names:
            curves = [reference["phase_features"][phase][feature_name] for reference in references]
            phase_features[phase][feature_name] = {
                "mean": aggregate_curve(curves, np.mean),
                "std": aggregate_curve(curves, np.std),
            }

    phase_durations = {}
    for phase in PHASES:
        values = [reference["phase_durations"][phase] for reference in references]
        phase_durations[phase] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    metric_groups = ["catch_metrics", "recovery_stability"]
    aggregates: dict[str, dict[str, dict[str, float | None]]] = {}
    for group in metric_groups:
        aggregates[group] = {}
        for key in references[0][group].keys():
            values = [reference[group][key] for reference in references if reference[group][key] is not None]
            if values:
                aggregates[group][key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }
            else:
                aggregates[group][key] = {"mean": None, "std": None}

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pose_backend": backend_name,
        "pose_model": model_name,
        "device": device,
        "reference_video_count": len(successful_videos),
        "sample_count_per_phase": REFERENCE_SAMPLE_COUNT,
        "thresholds": THRESHOLDS,
        "phase_features": phase_features,
        "phase_duration_proportions": phase_durations,
        "catch_metrics": aggregates["catch_metrics"],
        "recovery_stability": aggregates["recovery_stability"],
        "quality_flags": [
            {
                "video_path": item["video_path"],
                "warnings": item["warnings"],
                "processed_frames": item["processed_frames"],
            }
            for item in successful_videos
        ],
    }


def aggregate_curve(curves: list[list[float | None]], reducer) -> list[float | None]:
    result: list[float | None] = []
    for sample_index in range(REFERENCE_SAMPLE_COUNT):
        values = [
            curve[sample_index] for curve in curves if sample_index < len(curve) and curve[sample_index] is not None
        ]
        result.append(float(reducer(values)) if values else None)
    return result


def compute_reference_deviations(
    records: list[FrameRecord],
    phases: PhaseMarkers,
    reference_profile: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if reference_profile is None:
        return [], {
            "score": None,
            "score_breakdown": {},
            "score_breakdown_100": {},
            "low_confidence_categories": [],
            "findings": [],
            "reference_video_count": 0,
            "scoring_notes": ["Reference scoring disabled because no valid reference profile was loaded."],
        }

    normalized = build_normalized_lift(records, phases)
    deviations: list[dict[str, Any]] = []
    group_scores: dict[str, float] = {}
    group_scores_100: dict[str, float] = {}
    group_low_confidence: list[str] = []
    scoring_notes: list[str] = []
    reference_video_count = int(reference_profile.get("reference_video_count") or 0)
    if reference_video_count < 3:
        scoring_notes.append(
            f"Reference profile contains {reference_video_count} accepted video(s); "
            "feature tolerance floors were used to avoid over-penalizing normal pose and phase jitter."
        )

    group_feature_map = {
        "bar_path": [
            ("first_pull", "bar_path_x_norm"),
            ("second_pull", "bar_path_y_norm"),
            ("turnover", "bar_path_x_norm"),
        ],
        "torso_hip": [
            ("first_pull", "torso_angle_norm"),
            ("first_pull", "hip_height_norm"),
        ],
        "turnover_catch": [
            ("turnover", "elbow_angle_mean_deg"),
            ("catch", "hip_height_norm"),
            ("catch", "overhead_alignment_norm"),
        ],
        "overhead_recovery": [
            ("catch", "overhead_alignment_norm"),
            ("recovery", "overhead_alignment_norm"),
            ("recovery", "bar_path_y_norm"),
        ],
        "symmetry_footwork": [
            ("catch", "left_right_knee_angle_diff_deg"),
            ("recovery", "foot_displacement_norm"),
        ],
    }

    per_group_deviation: dict[str, list[float]] = {key: [] for key in group_feature_map}
    per_group_missing: dict[str, int] = {key: 0 for key in group_feature_map}
    per_group_total: dict[str, int] = {key: 0 for key in group_feature_map}

    for group_name, features in group_feature_map.items():
        for phase, feature_name in features:
            reference_curve, reference_feature_name = reference_feature_curve(reference_profile, phase, feature_name)
            observed_feature_name = reference_feature_name if reference_feature_name != feature_name else feature_name
            observed_curve = normalized["phase_features"][phase].get(observed_feature_name, [])
            if reference_curve is None:
                sample_count = len(normalized["phase_features"][phase].get(feature_name, []))
                per_group_missing[group_name] += sample_count
                per_group_total[group_name] += sample_count
                continue
            if not observed_curve:
                per_group_missing[group_name] += len(reference_curve["mean"])
                per_group_total[group_name] += len(reference_curve["mean"])
                continue
            for sample_index, observed in enumerate(observed_curve):
                per_group_total[group_name] += 1
                reference_mean = reference_curve["mean"][sample_index]
                reference_std = reference_curve["std"][sample_index]
                if observed is None or reference_mean is None:
                    per_group_missing[group_name] += 1
                    continue
                deviation = abs(float(observed) - float(reference_mean))
                tolerance_floor = FEATURE_TOLERANCE_FLOORS.get(reference_feature_name, 0.05)
                scale = max(float(reference_std or 0.0), tolerance_floor)
                z_score = deviation / scale
                per_group_deviation[group_name].append(z_score)
                deviations.append(
                    {
                        "group": group_name,
                        "phase": phase,
                        "feature": feature_name,
                        "reference_feature": reference_feature_name,
                        "sample_index": sample_index,
                        "observed": observed,
                        "reference_mean": reference_mean,
                        "reference_std": reference_std,
                        "tolerance_floor": tolerance_floor,
                        "z_score": z_score,
                    }
                )

    for group_name, z_scores in per_group_deviation.items():
        missing_ratio = per_group_missing[group_name] / max(1, per_group_total[group_name])
        if missing_ratio > THRESHOLDS["missing_frame_ratio_low_confidence"]:
            group_low_confidence.append(group_name)
        average_deviation = _nanmean(z_scores)
        penalty = min(1.0, (4.0 if average_deviation is None else average_deviation) / 4.0)
        group_scores[group_name] = round(SCORING_WEIGHTS[group_name] * (1.0 - penalty), 2)
        group_scores_100[group_name] = round(100.0 * (1.0 - penalty), 1)

    deviations.sort(key=lambda item: item["z_score"], reverse=True)
    findings = unique_findings(deviations, limit=5)
    overall_score = round(sum(group_scores.values()), 2)
    return deviations, {
        "score": overall_score,
        "score_breakdown": group_scores,
        "score_breakdown_100": group_scores_100,
        "low_confidence_categories": group_low_confidence,
        "findings": findings,
        "reference_video_count": reference_video_count,
        "scoring_notes": scoring_notes,
    }


def reference_feature_curve(
    reference_profile: dict[str, Any],
    phase: str,
    feature_name: str,
) -> tuple[dict[str, list[float | None]] | None, str]:
    phase_features = reference_profile.get("phase_features", {}).get(phase, {})
    if feature_name in phase_features:
        return phase_features[feature_name], feature_name
    if feature_name == "foot_displacement_norm" and "foot_displacement_px" in phase_features:
        return phase_features["foot_displacement_px"], "foot_displacement_px"
    return None, feature_name


def unique_findings(deviations: list[dict[str, Any]], limit: int) -> list[str]:
    findings: list[str] = []
    seen: set[str] = set()
    for deviation in deviations:
        if float(deviation.get("z_score") or 0.0) < 1.0:
            continue
        finding = deviation_to_finding(deviation)
        if finding in seen:
            continue
        findings.append(finding)
        seen.add(finding)
        if len(findings) >= limit:
            break
    return findings


def deviation_to_finding(deviation: dict[str, Any]) -> str:
    phase = deviation["phase"].replace("_", " ")
    feature = deviation["feature"]
    if feature == "bar_path_x_norm":
        return f"Bar drifted away from the reference path during {phase}."
    if feature == "bar_path_y_norm":
        return f"Bar elevation timing differed from the reference during {phase}."
    if feature == "hip_height_norm":
        return f"Catch or recovery depth was outside the reference range during {phase}."
    if feature == "torso_angle_norm":
        return "Torso angle in the first pull differed from the reference pattern."
    if feature == "overhead_alignment_norm":
        return f"Overhead bar alignment was less stable than the reference during {phase}."
    if feature == "elbow_angle_mean_deg":
        return "Turnover elbow timing lagged the reference pattern."
    if feature in {"foot_displacement_norm", "foot_displacement_px"}:
        return f"Foot displacement exceeded the reference range during {phase}."
    return f"{feature} deviated from the reference during {phase}."


def apply_per_frame_reference_deviation(
    records: list[FrameRecord],
    phases: PhaseMarkers,
    reference_profile: dict[str, Any] | None,
) -> None:
    if reference_profile is None:
        return

    for record in records:
        if record.normalized_phase_progress is None:
            continue
        sample_index = min(
            REFERENCE_SAMPLE_COUNT - 1,
            int(round(record.normalized_phase_progress * (REFERENCE_SAMPLE_COUNT - 1))),
        )
        feature_names = [
            "bar_path_x_norm",
            "bar_path_y_norm",
            "hip_height_norm",
            "overhead_alignment_norm",
        ]
        deviations = []
        for feature_name in feature_names:
            reference, reference_feature_name = reference_feature_curve(reference_profile, record.phase, feature_name)
            if reference is None:
                continue
            observed = getattr(record, reference_feature_name)
            reference_mean = reference["mean"][sample_index]
            reference_std = reference["std"][sample_index]
            if observed is None or reference_mean is None:
                continue
            tolerance_floor = FEATURE_TOLERANCE_FLOORS.get(reference_feature_name, 0.05)
            deviations.append(
                abs(float(observed) - float(reference_mean)) / max(float(reference_std or 0.0), tolerance_floor)
            )
        if deviations:
            record.reference_deviation = float(sum(deviations) / len(deviations))


class SnatchAnalysisSession:
    def __init__(self, config: AnalyzerConfig) -> None:
        self.config = config
        self.video_path = config.video_path
        self.output_dir = config.output_dir or make_output_dir(config.video_path)
        self.device = resolve_device(config.device)
        self.pose_backend, actual_device = create_pose_backend(config.pose_backend_name, config.model_name, self.device)
        self.device = actual_device
        self.reference_profile = load_reference_profile(config.reference_path)
        self.window_name = "Snatch Detector V1"

    def run(self) -> dict[str, object]:
        artifacts = self.run_analysis()
        return self._persist_outputs(artifacts)

    def run_analysis(self) -> AnalysisArtifacts:
        capture, fps, width, height, frame_count = open_video(self.video_path)
        records: list[FrameRecord] = []
        pose_dump: list[dict[str, list[float] | None]] = []
        path_points: list[tuple[int, int]] = []
        path_phases: list[str] = []
        initial_ankle_mid_x: float | None = None
        bar_tracker: BarbellTracker | None = None
        writer = None
        aborted = False
        last_display: np.ndarray | None = None
        completion = CompletionStatus()
        live_machine = LiftStateMachine(fps)

        try:
            ok, first_frame = capture.read()
            if not ok or first_frame is None:
                raise RuntimeError("Unable to read the first frame during analysis startup.")

            initial_point = self._request_point(first_frame, "Click barbell/plate center")
            if initial_point is None:
                raise RuntimeError("Barbell selection cancelled.")

            bar_tracker = BarbellTracker(first_frame, initial_point)
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if self.config.show_live_window:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            frame_index = 0
            paused = False
            auto_pause_latched = False
            while True:
                if not paused:
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        break

                    pose = self.pose_backend.infer(frame)
                    if frame_index == 0:
                        track_state = BarTrackState(
                            point=bar_tracker.current_point(),
                            smoothed_point=bar_tracker.current_point(),
                            confidence=1.0,
                            fallback_used=False,
                            lost_frames=0,
                        )
                    else:
                        track_state = bar_tracker.update(frame)

                    if track_state.lost_frames >= THRESHOLDS["tracking_reclick_frames"]:
                        reclick = self._request_point(
                            frame,
                            "Tracking lost - re-click barbell/plate center",
                        )
                        if reclick is None:
                            raise RuntimeError("Tracking was lost and re-click was cancelled.")
                        bar_tracker.reinitialize(frame, reclick)
                        track_state = BarTrackState(
                            point=bar_tracker.current_point(),
                            smoothed_point=bar_tracker.current_point(),
                            confidence=1.0,
                            fallback_used=True,
                            lost_frames=0,
                        )

                    record = self._build_record(
                        frame_index=frame_index,
                        fps=fps,
                        pose=pose,
                        track_state=track_state,
                        initial_ankle_mid_x=initial_ankle_mid_x,
                    )
                    if initial_ankle_mid_x is None and record.ankle_mid_x is not None:
                        initial_ankle_mid_x = record.ankle_mid_x
                        record.foot_displacement_px = 0.0

                    records.append(record)
                    completion = live_machine.update(records, record)
                    pose_dump.append(keypoint_array_to_dict(pose))
                    if record.bar_x is not None and record.bar_y is not None:
                        path_points.append((int(record.bar_x), int(record.bar_y)))
                        path_phases.append(record.phase)

                    score_text = None
                    if completion.completion_frame is not None:
                        score_text = "Snatch complete"
                    last_display = annotate_frame(
                        frame_bgr=frame,
                        pose=pose,
                        record=record,
                        phase_label="analyzing",
                        path_points=path_points[-120:],
                        global_warnings=[],
                        completion_text=score_text,
                        score_report=None,
                        path_phases=path_phases[-120:],
                    )
                    if (
                        self.config.auto_pause
                        and self.config.pause_mode == "after-recovery-stable"
                        and completion.completion_frame is not None
                        and not auto_pause_latched
                    ):
                        paused = True
                        auto_pause_latched = True
                        completion.auto_paused = True

                    self._emit_progress(
                        frame_index=frame_index,
                        frame_count=frame_count,
                        live_state=record.live_state,
                        paused=paused,
                    )

                    frame_index += 1

                if self.config.show_live_window and last_display is not None:
                    _draw_exit_hint(last_display)
                    cv2.imshow(self.window_name, last_display)
                    key = cv2.waitKey(1 if not paused else 30) & 0xFF
                    if key in (ord("q"), 27):
                        aborted = True
                        break
                    if key == ord(" "):
                        paused = not paused
                    if key == ord("s"):
                        save_snapshot(last_display, self.output_dir)
                elif paused:
                    paused = False

            phases = detect_phases(records)
            finalize_record_features(records, phases)
            warnings = build_warnings(records, phases)
            if completion.completion_frame is None:
                completion.completion_reason = "Recovery stability was not confirmed before the video ended."
            self._apply_phase_warnings(records, phases)
            apply_per_frame_reference_deviation(records, phases, self.reference_profile)
            reference_deviations, score = compute_reference_deviations(records, phases, self.reference_profile)

            return AnalysisArtifacts(
                fps=fps,
                width=width,
                height=height,
                frame_count=frame_count,
                processed_frames=len(records),
                records=records,
                pose_dump=pose_dump,
                phases=phases,
                completion=completion,
                warnings=warnings,
                score=score,
                reference_profile_path=None if self.config.reference_path is None else str(self.config.reference_path),
                reference_deviations=reference_deviations[:25],
                output_dir=self.output_dir,
                aborted=aborted,
            )
        finally:
            capture.release()
            if writer is not None:
                writer.release()
            self.pose_backend.close()
            cv2.destroyAllWindows()

    def _auto_seed_point(self, frame_bgr: np.ndarray) -> tuple[int, int] | None:
        return (frame_bgr.shape[1] // 2, frame_bgr.shape[0] // 2)

    def _request_point(self, frame_bgr: np.ndarray, window_name: str) -> tuple[int, int] | None:
        if self.config.point_provider is not None:
            return self.config.point_provider(frame_bgr, window_name)
        if self.config.initial_point is not None and "re-click" not in window_name.lower():
            return self.config.initial_point
        if self.config.interactive:
            return request_point_click(frame_bgr, window_name=window_name)
        return self._auto_seed_point(frame_bgr)

    def _emit_progress(
        self,
        frame_index: int,
        frame_count: int,
        live_state: str,
        paused: bool,
    ) -> None:
        if self.config.progress_callback is None:
            return
        progress = None
        if frame_count > 0:
            progress = max(0, min(100, int(((frame_index + 1) / frame_count) * 100)))
        self.config.progress_callback(
            {
                "frame_index": frame_index,
                "frame_count": frame_count,
                "progress": progress,
                "live_state": live_state,
                "paused": paused,
            }
        )

    def _apply_phase_warnings(self, records: list[FrameRecord], phases: PhaseMarkers) -> None:
        for record in records:
            if record.phase == "catch":
                if (
                    record.left_right_knee_angle_diff_deg is not None
                    and record.left_right_knee_angle_diff_deg > THRESHOLDS["asymmetry_angle_deg"]
                ):
                    record.warnings.append("Catch knee asymmetry")

    def _persist_outputs(self, artifacts: AnalysisArtifacts) -> dict[str, object]:
        annotated_width, annotated_height = annotated_video_dimensions(artifacts.width, artifacts.height)
        if self.config.persist_outputs:
            writer, annotated_video_codec, annotated_video_path = open_annotated_video_writer(
                self.output_dir / "annotated.mp4",
                artifacts.fps,
                annotated_width,
                annotated_height,
            )
            try:
                self._write_annotated_video(writer, artifacts)
            finally:
                writer.release()
            self._write_frames_csv(artifacts.records)
            summary = self._write_metrics_json(artifacts)
            summary["annotated_video_codec"] = annotated_video_codec
            summary["annotated_video_filename"] = annotated_video_path.name
        else:
            summary = self._build_metrics_summary(artifacts)
        summary["annotated_video_width"] = annotated_width
        summary["annotated_video_height"] = annotated_height
        if self.config.persist_outputs:
            (self.output_dir / "metrics.json").write_text(json.dumps(_json_ready(summary), indent=2), encoding="utf-8")
        return summary

    def _build_record(
        self,
        frame_index: int,
        fps: float,
        pose: PoseResult | None,
        track_state: BarTrackState,
        initial_ankle_mid_x: float | None,
    ) -> FrameRecord:
        keypoints = {} if pose is None else pose.keypoints
        left_ankle = keypoints.get("left_ankle")
        right_ankle = keypoints.get("right_ankle")
        left_shoulder = keypoints.get("left_shoulder")
        right_shoulder = keypoints.get("right_shoulder")
        left_hip = keypoints.get("left_hip")
        right_hip = keypoints.get("right_hip")
        left_knee = keypoints.get("left_knee")
        right_knee = keypoints.get("right_knee")
        left_elbow = keypoints.get("left_elbow")
        right_elbow = keypoints.get("right_elbow")
        left_wrist = keypoints.get("left_wrist")
        right_wrist = keypoints.get("right_wrist")

        ankle_mid = midpoint(left_ankle, right_ankle) or average_visible(left_ankle, right_ankle)
        shoulder_mid = midpoint(left_shoulder, right_shoulder) or average_visible(left_shoulder, right_shoulder)
        hip_mid = midpoint(left_hip, right_hip) or average_visible(left_hip, right_hip)
        bar_point = track_state.smoothed_point or track_state.point

        left_knee_angle = angle_at_joint(left_hip, left_knee, left_ankle)
        right_knee_angle = angle_at_joint(right_hip, right_knee, right_ankle)
        left_hip_angle = angle_at_joint(left_shoulder, left_hip, left_knee)
        right_hip_angle = angle_at_joint(right_shoulder, right_hip, right_knee)
        left_elbow_angle = angle_at_joint(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = angle_at_joint(right_shoulder, right_elbow, right_wrist)
        extension_components = [
            value
            for value in [
                left_knee_angle,
                right_knee_angle,
                left_hip_angle,
                right_hip_angle,
            ]
            if value is not None
        ]
        extension_score = float(sum(extension_components) / len(extension_components)) if extension_components else None

        foot_displacement = None
        if ankle_mid is not None and initial_ankle_mid_x is not None:
            foot_displacement = float(ankle_mid[0] - initial_ankle_mid_x)

        left_right_knee_angle_diff = None
        if left_knee_angle is not None and right_knee_angle is not None:
            left_right_knee_angle_diff = abs(left_knee_angle - right_knee_angle)

        left_right_hip_angle_diff = None
        if left_hip_angle is not None and right_hip_angle is not None:
            left_right_hip_angle_diff = abs(left_hip_angle - right_hip_angle)

        shoulder_y_diff = None
        if left_shoulder is not None and right_shoulder is not None:
            shoulder_y_diff = abs(left_shoulder[1] - right_shoulder[1])

        bar_offset = None
        if bar_point is not None and ankle_mid is not None:
            bar_offset = float(bar_point[0] - ankle_mid[0])

        body_scale = None
        if ankle_mid is not None and shoulder_mid is not None:
            body_scale = abs(float(ankle_mid[1] - shoulder_mid[1]))
        elif ankle_mid is not None and hip_mid is not None:
            body_scale = abs(float(ankle_mid[1] - hip_mid[1]))

        warnings: list[str] = []
        if bar_point is None:
            warnings.append("bar tracking uncertain")
        elif track_state.fallback_used:
            warnings.append("bar tracking fallback")
        if pose is None or pose.pose_confidence < 0.1:
            warnings.append("pose confidence low")

        return FrameRecord(
            frame_index=frame_index,
            timestamp_s=frame_index / max(fps, 1e-6),
            pose_confidence=0.0 if pose is None else pose.pose_confidence,
            bar_x=None if bar_point is None else float(bar_point[0]),
            bar_y=None if bar_point is None else float(bar_point[1]),
            bar_confidence=track_state.confidence,
            bar_fallback_used=track_state.fallback_used,
            ankle_mid_x=None if ankle_mid is None else float(ankle_mid[0]),
            ankle_mid_y=None if ankle_mid is None else float(ankle_mid[1]),
            shoulder_mid_x=None if shoulder_mid is None else float(shoulder_mid[0]),
            shoulder_mid_y=None if shoulder_mid is None else float(shoulder_mid[1]),
            hip_mid_x=None if hip_mid is None else float(hip_mid[0]),
            hip_mid_y=None if hip_mid is None else float(hip_mid[1]),
            bar_horizontal_offset=bar_offset,
            torso_angle_deg=torso_angle_from_vertical(hip_mid, shoulder_mid),
            left_knee_angle_deg=left_knee_angle,
            right_knee_angle_deg=right_knee_angle,
            left_hip_angle_deg=left_hip_angle,
            right_hip_angle_deg=right_hip_angle,
            left_elbow_angle_deg=left_elbow_angle,
            right_elbow_angle_deg=right_elbow_angle,
            extension_score=extension_score,
            left_right_knee_angle_diff_deg=left_right_knee_angle_diff,
            left_right_hip_angle_diff_deg=left_right_hip_angle_diff,
            left_right_shoulder_y_diff_px=shoulder_y_diff,
            foot_displacement_px=foot_displacement,
            body_scale_px=body_scale,
            warnings=warnings,
        )

    def _write_annotated_video(self, writer: cv2.VideoWriter, artifacts: AnalysisArtifacts) -> None:
        capture, _, _, _, _ = open_video(self.video_path)
        path_points: list[tuple[int, int]] = []
        path_phases: list[str] = []
        score_text = None
        if artifacts.score["score"] is not None:
            score_text = f"Score: {artifacts.score['score']:.1f}/100"

        try:
            for frame_index, record in enumerate(artifacts.records):
                ok, frame = capture.read()
                if not ok or frame is None:
                    break

                pose = None
                if frame_index < len(artifacts.pose_dump):
                    pose_keypoints = {
                        key: None if value is None else (float(value[0]), float(value[1]), float(value[2]))
                        for key, value in artifacts.pose_dump[frame_index].items()
                    }
                    if pose_keypoints:
                        pose = PoseResult(
                            keypoints=pose_keypoints,
                            bbox_xyxy=(0.0, 0.0, 0.0, 0.0),
                            pose_confidence=record.pose_confidence,
                            visible_area=0.0,
                        )

                if record.bar_x is not None and record.bar_y is not None:
                    path_points.append((int(record.bar_x), int(record.bar_y)))
                    path_phases.append(record.phase)

                completion_text = None
                if (
                    artifacts.completion.completion_frame is not None
                    and frame_index >= artifacts.completion.completion_frame
                ):
                    completion_text = "Snatch complete"
                vis = annotate_frame(
                    frame_bgr=frame,
                    pose=pose,
                    record=record,
                    phase_label=record.phase,
                    path_points=path_points[-120:],
                    global_warnings=artifacts.warnings,
                    score_text=score_text,
                    completion_text=completion_text,
                    score_report=artifacts.score,
                    path_phases=path_phases[-120:],
                )
                writer.write(vis)
        finally:
            capture.release()

    def _write_frames_csv(self, records: list[FrameRecord]) -> None:
        output_path = self.output_dir / "frames.csv"
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(asdict(records[0]).keys()) if records else ["frame_index"],
            )
            writer.writeheader()
            for record in records:
                row = asdict(record)
                row["warnings"] = "|".join(record.warnings)
                writer.writerow(_json_ready(row))

    def _build_metrics_summary(self, artifacts: AnalysisArtifacts) -> dict[str, object]:
        catch_record = artifacts.records[artifacts.phases.catch_frame] if artifacts.records else None
        return {
            "schema_version": SCHEMA_VERSION,
            "video_path": str(self.video_path),
            "device": self.device,
            "pose_backend": self.config.pose_backend_name,
            "pose_model": self.config.model_name,
            "frame_count": artifacts.frame_count,
            "processed_frames": artifacts.processed_frames,
            "fps": artifacts.fps,
            "aborted": artifacts.aborted,
            "phase_frames": asdict(artifacts.phases),
            "thresholds": THRESHOLDS,
            "warnings": artifacts.warnings,
            "completion_frame": artifacts.completion.completion_frame,
            "completion_reason": artifacts.completion.completion_reason,
            "reference_profile": artifacts.reference_profile_path,
            "score": artifacts.score["score"],
            "score_breakdown": artifacts.score["score_breakdown"],
            "score_breakdown_100": artifacts.score.get("score_breakdown_100", {}),
            "low_confidence_categories": artifacts.score["low_confidence_categories"],
            "reference_video_count": artifacts.score.get("reference_video_count"),
            "scoring_notes": artifacts.score.get("scoring_notes", []),
            "findings": artifacts.score["findings"],
            "reference_deviations": artifacts.reference_deviations,
            "annotated_video_codec": None,
            "annotated_video_filename": None,
            "annotated_video_width": annotated_video_dimensions(artifacts.width, artifacts.height)[0],
            "annotated_video_height": annotated_video_dimensions(artifacts.width, artifacts.height)[1],
            "catch_metrics": {
                "bar_x": None if catch_record is None else catch_record.bar_x,
                "bar_y": None if catch_record is None else catch_record.bar_y,
                "shoulder_mid_x": None if catch_record is None else catch_record.shoulder_mid_x,
                "shoulder_mid_y": None if catch_record is None else catch_record.shoulder_mid_y,
                "hip_mid_x": None if catch_record is None else catch_record.hip_mid_x,
                "hip_mid_y": None if catch_record is None else catch_record.hip_mid_y,
                "ankle_mid_x": None if catch_record is None else catch_record.ankle_mid_x,
                "ankle_mid_y": None if catch_record is None else catch_record.ankle_mid_y,
                "left_knee_angle_deg": None if catch_record is None else catch_record.left_knee_angle_deg,
                "right_knee_angle_deg": None if catch_record is None else catch_record.right_knee_angle_deg,
                "left_hip_angle_deg": None if catch_record is None else catch_record.left_hip_angle_deg,
                "right_hip_angle_deg": None if catch_record is None else catch_record.right_hip_angle_deg,
                "left_elbow_angle_deg": None if catch_record is None else catch_record.left_elbow_angle_deg,
                "right_elbow_angle_deg": None if catch_record is None else catch_record.right_elbow_angle_deg,
            },
        }

    def _write_metrics_json(self, artifacts: AnalysisArtifacts) -> dict[str, object]:
        summary = self._build_metrics_summary(artifacts)
        output_path = self.output_dir / "metrics.json"
        output_path.write_text(json.dumps(_json_ready(summary), indent=2), encoding="utf-8")
        return summary


def discover_reference_videos(paths: list[Path]) -> list[Path]:
    collected: list[Path] = []
    for path in paths:
        if path.is_dir():
            for pattern in ("*.mp4", "*.mov", "*.avi", "*.mkv"):
                collected.extend(sorted(path.glob(pattern)))
        elif path.exists():
            collected.append(path)
    deduped: list[Path] = []
    seen = set()
    for path in collected:
        resolved = str(path.resolve())
        if resolved not in seen:
            deduped.append(path)
            seen.add(resolved)
    return deduped


def validate_reference_candidate(artifacts: AnalysisArtifacts) -> list[str]:
    issues: list[str] = []
    phase_frames = asdict(artifacts.phases)
    required = [
        "first_pull_start",
        "second_pull_start",
        "catch_frame",
        "recovery_start",
        "finish_frame",
    ]
    for key in required:
        if phase_frames[key] is None:
            issues.append(f"missing {key}")
    if artifacts.completion.completion_frame is None:
        issues.append("recovery stability not confirmed")
    if artifacts.processed_frames < 20:
        issues.append("too few processed frames")
    return issues


def build_reference_profile(
    videos: list[Path],
    backend_name: str,
    model_name: str,
    device: str,
    reference_output_path: Path,
) -> dict[str, Any]:
    reference_output_path.parent.mkdir(parents=True, exist_ok=True)
    successful_videos: list[dict[str, Any]] = []
    excluded_videos: list[dict[str, Any]] = []
    build_output_dir = make_output_dir(
        reference_output_path.with_suffix(""),
        output_root=reference_output_path.parent.parent / "outputs",
    )

    for video_path in videos:
        session = SnatchAnalysisSession(
            AnalyzerConfig(
                video_path=video_path,
                pose_backend_name=backend_name,
                model_name=model_name,
                device=device,
                output_dir=make_output_dir(video_path),
                auto_pause=False,
                interactive=True,
                show_live_window=True,
                persist_outputs=False,
            )
        )
        artifacts = session.run_analysis()
        issues = validate_reference_candidate(artifacts)
        payload = {
            "video_path": str(video_path),
            "processed_frames": artifacts.processed_frames,
            "warnings": artifacts.warnings,
            "issues": issues,
            "normalized_lift": build_normalized_lift(artifacts.records, artifacts.phases),
        }
        if issues:
            excluded_videos.append(payload)
        else:
            successful_videos.append(payload)

    if not successful_videos:
        raise RuntimeError("No reference videos produced valid setup, pull, catch, recovery, and finish phases.")

    reference_payload = aggregate_reference_payload(successful_videos, backend_name, model_name, resolve_device(device))
    reference_payload["excluded_videos"] = [
        {
            "video_path": item["video_path"],
            "issues": item["issues"],
            "warnings": item["warnings"],
        }
        for item in excluded_videos
    ]
    reference_output_path.write_text(json.dumps(_json_ready(reference_payload), indent=2), encoding="utf-8")
    build_report = {
        "schema_version": SCHEMA_VERSION,
        "reference_output": str(reference_output_path),
        "reference_video_count": len(successful_videos),
        "excluded_video_count": len(excluded_videos),
        "videos": [
            {
                "video_path": item["video_path"],
                "issues": item["issues"],
                "warnings": item["warnings"],
            }
            for item in successful_videos + excluded_videos
        ],
    }
    (build_output_dir / "reference_build_report.json").write_text(
        json.dumps(_json_ready(build_report), indent=2),
        encoding="utf-8",
    )
    return build_report
