"""Convert MediaPipe hand landmarks into provisional visualization targets."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


JOINT_COUNT = 16
MAX_TARGET_NORMALIZED = 1.0
CURL_FEATURE_AT_FULL_TARGET = 0.65


@dataclass(frozen=True)
class HandFeatures:
    finger_curl: tuple[float, float, float, float]
    thumb_curl: float
    thumb_opposition: float


def _angle_degrees(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return the angle ABC in degrees, or zero for a degenerate segment."""
    first = a - b
    second = c - b
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator < 1e-8:
        return 0.0
    cosine = float(np.dot(first, second) / denominator)
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def _curl(points: np.ndarray, proximal: int, middle: int, distal: int) -> float:
    # 180 degrees is extended; progressively smaller angles indicate curling.
    angle = _angle_degrees(points[proximal], points[middle], points[distal])
    return float(np.clip((155.0 - angle) / 100.0, 0.0, 1.0))


def extract_features(landmarks: Iterable[Iterable[float]]) -> HandFeatures:
    """Extract scale-independent flexion and thumb opposition from 21 points."""
    points = np.asarray(tuple(tuple(point) for point in landmarks), dtype=float)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ValueError("Expected 21 finite landmarks, each with x, y, and z")

    fingers = (
        (5, 6, 7, 8),     # index
        (9, 10, 11, 12),  # middle
        (13, 14, 15, 16), # ring
        (17, 18, 19, 20), # little
    )
    curls = tuple(
        (_curl(points, mcp, pip, dip) + _curl(points, pip, dip, tip)) / 2.0
        for mcp, pip, dip, tip in fingers
    )

    thumb_curl = _curl(points, 2, 3, 4)
    palm_width = float(np.linalg.norm(points[5, :2] - points[17, :2]))
    if palm_width < 1e-6:
        raise ValueError("Landmarks do not define a usable palm width")
    thumb_gap = float(np.linalg.norm(points[4, :2] - points[5, :2])) / palm_width
    thumb_opposition = float(np.clip((1.25 - thumb_gap) / 0.9, 0.0, 1.0))

    return HandFeatures(curls, thumb_curl, thumb_opposition)


def _scale_curl_to_full_joint_range(curl: float) -> float:
    """Map a human curl feature to normalized Allegro joint travel."""
    return float(np.clip(curl / CURL_FEATURE_AT_FULL_TARGET, 0.0, 1.0))


def retarget_to_visual_joints(features: HandFeatures) -> tuple[float, ...]:
    """Map features to 16 normalized Allegro joint targets in [0, 1].

    The target order is thumb, index, middle, and ring (four values each).
    The little finger has no Allegro digit. This is not a calibrated kinematic
    retargeter and must not drive hardware.
    """
    thumb = (
        features.thumb_opposition,
        _scale_curl_to_full_joint_range(features.thumb_curl),
        _scale_curl_to_full_joint_range(features.thumb_curl),
        _scale_curl_to_full_joint_range(features.thumb_curl),
    )
    values = list(thumb)
    for curl in features.finger_curl[:3]:
        joint_curl = _scale_curl_to_full_joint_range(curl)
        values.extend((0.5, joint_curl, joint_curl, joint_curl))
    return tuple(float(value) for value in values)
