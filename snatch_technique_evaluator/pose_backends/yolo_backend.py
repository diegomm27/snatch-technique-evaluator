from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import KEYPOINT_NAMES, PoseBackend, PoseResult


@dataclass(slots=True)
class _Candidate:
    score: float
    visible_area: float
    bbox_xyxy: tuple[float, float, float, float]
    keypoints: dict[str, tuple[float, float, float] | None]


class YoloPoseBackend(PoseBackend):
    def __init__(
        self,
        model_name: str = "yolo11n-pose.pt",
        device: str = "cpu",
        conf: float = 0.25,
        imgsz: int = 960,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not installed. Run "
                r"'python -m pip install ultralytics'."
            ) from exc

        self.model = YOLO(model_name)
        self.device = device
        self.conf = conf
        self.imgsz = imgsz

    def infer(self, frame_bgr) -> PoseResult | None:
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        if not results:
            return None

        result = results[0]
        if result.keypoints is None or result.keypoints.data is None:
            return None

        keypoint_array = result.keypoints.data.detach().cpu().numpy()
        if keypoint_array.size == 0:
            return None

        boxes = None if result.boxes is None else result.boxes.xyxy.detach().cpu().numpy()
        box_scores = None if result.boxes is None else result.boxes.conf.detach().cpu().numpy()

        candidates: list[_Candidate] = []
        for index, points in enumerate(keypoint_array):
            bbox_xyxy = (
                (
                    float(boxes[index][0]),
                    float(boxes[index][1]),
                    float(boxes[index][2]),
                    float(boxes[index][3]),
                )
                if boxes is not None
                else (0.0, 0.0, 0.0, 0.0)
            )
            visible = {
                name: (
                    float(points[kp_index][0]),
                    float(points[kp_index][1]),
                    float(points[kp_index][2]),
                )
                if float(points[kp_index][2]) > 0.05
                else None
                for name, kp_index in KEYPOINT_NAMES.items()
            }
            visible_points = [item for item in visible.values() if item is not None]
            if not visible_points:
                continue

            xs = [item[0] for item in visible_points]
            ys = [item[1] for item in visible_points]
            visible_area = max(xs) - min(xs)
            visible_area *= max(ys) - min(ys)
            pose_confidence = float(np.mean([item[2] for item in visible_points]))
            if box_scores is not None:
                pose_confidence *= float(box_scores[index])

            candidates.append(
                _Candidate(
                    score=pose_confidence,
                    visible_area=float(visible_area),
                    bbox_xyxy=bbox_xyxy,
                    keypoints=visible,
                )
            )

        if not candidates:
            return None

        selected = max(
            candidates,
            key=lambda candidate: (candidate.score, candidate.visible_area),
        )
        return PoseResult(
            keypoints=selected.keypoints,
            bbox_xyxy=selected.bbox_xyxy,
            pose_confidence=selected.score,
            visible_area=selected.visible_area,
        )
