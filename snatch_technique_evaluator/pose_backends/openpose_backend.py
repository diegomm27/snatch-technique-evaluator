from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .base import PoseBackend, PoseResult

OPENPOSE_COCO_KEYPOINTS = {
    "left_shoulder": 5,
    "right_shoulder": 2,
    "left_elbow": 6,
    "right_elbow": 3,
    "left_wrist": 7,
    "right_wrist": 4,
    "left_hip": 11,
    "right_hip": 8,
    "left_knee": 12,
    "right_knee": 9,
    "left_ankle": 13,
    "right_ankle": 10,
}


class OpenPoseBackend(PoseBackend):
    def __init__(
        self,
        model_name: str = "openpose",
        device: str = "cpu",
        input_size: int = 368,
        confidence_threshold: float = 0.10,
    ) -> None:
        proto_path, weights_path = self._resolve_model_paths(model_name)
        self.net = cv2.dnn.readNetFromCaffe(str(proto_path), str(weights_path))
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold

        if device.lower().startswith("cuda"):
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            except Exception:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def infer(self, frame_bgr) -> PoseResult | None:
        height, width = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame_bgr,
            1.0 / 255.0,
            (self.input_size, self.input_size),
            (0, 0, 0),
            swapRB=False,
            crop=False,
        )
        self.net.setInput(blob)
        output = self.net.forward()
        if output is None or output.ndim != 4:
            return None

        keypoints: dict[str, tuple[float, float, float] | None] = {}
        visible_points: list[tuple[float, float, float]] = []
        heatmap_height = output.shape[2]
        heatmap_width = output.shape[3]

        for name, index in OPENPOSE_COCO_KEYPOINTS.items():
            if index >= output.shape[1]:
                keypoints[name] = None
                continue
            heatmap = output[0, index, :, :]
            _, confidence, _, point = cv2.minMaxLoc(heatmap)
            if confidence < self.confidence_threshold:
                keypoints[name] = None
                continue
            x = float(width * point[0] / max(heatmap_width, 1))
            y = float(height * point[1] / max(heatmap_height, 1))
            keypoint = (x, y, float(confidence))
            keypoints[name] = keypoint
            visible_points.append(keypoint)

        if not visible_points:
            return None

        xs = [point[0] for point in visible_points]
        ys = [point[1] for point in visible_points]
        bbox_xyxy = (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))
        visible_area = float((max(xs) - min(xs)) * (max(ys) - min(ys)))
        pose_confidence = float(np.mean([point[2] for point in visible_points]))
        return PoseResult(
            keypoints=keypoints,
            bbox_xyxy=bbox_xyxy,
            pose_confidence=pose_confidence,
            visible_area=visible_area,
        )

    @staticmethod
    def _resolve_model_paths(model_name: str) -> tuple[Path, Path]:
        raw = model_name.strip()
        if ";" in raw:
            parts = {}
            for item in raw.split(";"):
                if "=" in item:
                    key, value = item.split("=", 1)
                    parts[key.strip().lower()] = Path(value.strip())
            if "proto" in parts and "weights" in parts:
                return OpenPoseBackend._validate_paths(parts["proto"], parts["weights"])

        path = Path(raw)
        if path.is_dir():
            proto_candidates = [
                path / "pose_deploy_linevec.prototxt",
                path / "pose_deploy.prototxt",
            ]
            weight_candidates = [
                path / "pose_iter_440000.caffemodel",
                path / "pose_iter_584000.caffemodel",
            ]
            proto_path = next(
                (candidate for candidate in proto_candidates if candidate.exists()),
                None,
            )
            weights_path = next(
                (candidate for candidate in weight_candidates if candidate.exists()),
                None,
            )
            if proto_path and weights_path:
                return OpenPoseBackend._validate_paths(proto_path, weights_path)

        if path.suffix.lower() == ".prototxt":
            weights_path = path.with_name("pose_iter_440000.caffemodel")
            return OpenPoseBackend._validate_paths(path, weights_path)

        if path.suffix.lower() == ".caffemodel":
            proto_path = path.with_name("pose_deploy_linevec.prototxt")
            return OpenPoseBackend._validate_paths(proto_path, path)

        raise RuntimeError(
            "OpenPose requires Caffe model files. Set --backend openpose and pass "
            "--model as an OpenPose model directory, a .prototxt path, a .caffemodel path, "
            "or 'proto=C:\\path\\pose_deploy_linevec.prototxt;weights=C:\\path\\pose_iter_440000.caffemodel'."
        )

    @staticmethod
    def _validate_paths(proto_path: Path, weights_path: Path) -> tuple[Path, Path]:
        if not proto_path.exists():
            raise RuntimeError(f"OpenPose prototxt not found: {proto_path}")
        if not weights_path.exists():
            raise RuntimeError(f"OpenPose weights not found: {weights_path}")
        return proto_path, weights_path
