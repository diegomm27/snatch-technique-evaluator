from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

KEYPOINT_NAMES = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}


@dataclass(slots=True)
class PoseResult:
    keypoints: dict[str, tuple[float, float, float] | None]
    bbox_xyxy: tuple[float, float, float, float]
    pose_confidence: float
    visible_area: float


class PoseBackend(ABC):
    @abstractmethod
    def infer(self, frame_bgr) -> PoseResult | None:
        raise NotImplementedError

    def close(self) -> None:
        return None
