from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class BarTrackState:
    point: tuple[float, float] | None
    smoothed_point: tuple[float, float] | None
    confidence: float
    fallback_used: bool
    lost_frames: int


class BarbellTracker:
    def __init__(
        self,
        first_frame_bgr: np.ndarray,
        initial_point: tuple[int, int],
        patch_radius: int = 18,
        smoothing_window: int = 5,
    ) -> None:
        self.patch_radius = patch_radius
        self.smoothing_window = smoothing_window
        self.points = deque(maxlen=smoothing_window)
        self.prev_gray = self._to_gray(first_frame_bgr)
        self.prev_point = np.array([[initial_point]], dtype=np.float32)
        self.template = self._extract_patch(first_frame_bgr, initial_point)
        self.lost_frames = 0
        self.last_confidence = 1.0
        self.last_fallback = False
        self.points.append((float(initial_point[0]), float(initial_point[1])))

    def reinitialize(self, frame_bgr: np.ndarray, point: tuple[int, int]) -> None:
        self.prev_gray = self._to_gray(frame_bgr)
        self.prev_point = np.array([[point]], dtype=np.float32)
        self.template = self._extract_patch(frame_bgr, point)
        self.lost_frames = 0
        self.last_confidence = 1.0
        self.last_fallback = False
        self.points.append((float(point[0]), float(point[1])))

    def current_point(self) -> tuple[float, float]:
        return self.points[-1]

    def update(self, frame_bgr: np.ndarray) -> BarTrackState:
        next_gray = self._to_gray(frame_bgr)
        next_point, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            next_gray,
            self.prev_point,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                20,
                0.03,
            ),
        )

        point: tuple[float, float] | None = None
        confidence = 0.0
        fallback_used = False

        if status is not None and int(status[0][0]) == 1:
            candidate = (float(next_point[0][0][0]), float(next_point[0][0][1]))
            if self._point_in_bounds(frame_bgr, candidate):
                optical_error = float(error[0][0]) if error is not None else 0.0
                confidence = max(0.0, 1.0 - min(optical_error / 25.0, 1.0))
                if confidence >= 0.25:
                    point = candidate

        if point is None:
            matched, match_score = self._fallback_match(frame_bgr)
            if matched is not None:
                point = matched
                confidence = match_score
                fallback_used = True

        if point is None:
            self.lost_frames += 1
            self.prev_gray = next_gray
            return BarTrackState(
                point=None,
                smoothed_point=self.points[-1] if self.points else None,
                confidence=0.0,
                fallback_used=False,
                lost_frames=self.lost_frames,
            )

        self.lost_frames = 0
        self.prev_gray = next_gray
        self.prev_point = np.array([[point]], dtype=np.float32)
        self.points.append(point)
        self.template = self._extract_patch(frame_bgr, (int(point[0]), int(point[1])))
        self.last_confidence = confidence
        self.last_fallback = fallback_used
        smoothed = self._smoothed_point()
        return BarTrackState(
            point=point,
            smoothed_point=smoothed,
            confidence=confidence,
            fallback_used=fallback_used,
            lost_frames=0,
        )

    def _smoothed_point(self) -> tuple[float, float]:
        xs = [point[0] for point in self.points]
        ys = [point[1] for point in self.points]
        return float(sum(xs) / len(xs)), float(sum(ys) / len(ys))

    def _fallback_match(self, frame_bgr: np.ndarray) -> tuple[tuple[float, float] | None, float]:
        if self.template is None or self.template.size == 0:
            return None, 0.0

        prev_x, prev_y = self.current_point()
        search_radius = max(48, self.patch_radius * 4)
        x1 = max(0, int(prev_x - search_radius))
        y1 = max(0, int(prev_y - search_radius))
        x2 = min(frame_bgr.shape[1], int(prev_x + search_radius))
        y2 = min(frame_bgr.shape[0], int(prev_y + search_radius))
        search = frame_bgr[y1:y2, x1:x2]
        if search.shape[0] < self.template.shape[0] or search.shape[1] < self.template.shape[1]:
            return None, 0.0

        result = cv2.matchTemplate(search, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_score, _, max_loc = cv2.minMaxLoc(result)
        if max_score < 0.55:
            return None, float(max_score)

        center_x = x1 + max_loc[0] + self.template.shape[1] / 2.0
        center_y = y1 + max_loc[1] + self.template.shape[0] / 2.0
        return (float(center_x), float(center_y)), float(max_score)

    def _extract_patch(self, frame_bgr: np.ndarray, point: tuple[int, int]) -> np.ndarray:
        x, y = point
        x1 = max(0, x - self.patch_radius)
        y1 = max(0, y - self.patch_radius)
        x2 = min(frame_bgr.shape[1], x + self.patch_radius)
        y2 = min(frame_bgr.shape[0], y + self.patch_radius)
        return frame_bgr[y1:y2, x1:x2].copy()

    @staticmethod
    def _to_gray(frame_bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _point_in_bounds(frame_bgr: np.ndarray, point: tuple[float, float]) -> bool:
        x, y = point
        return 0 <= x < frame_bgr.shape[1] and 0 <= y < frame_bgr.shape[0]
