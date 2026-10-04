"""Map normalized visualization targets into a robot profile's joint ranges."""

from __future__ import annotations

import numpy as np


def normalized_targets_to_joint_positions(
    targets: tuple[float, ...],
    joint_limits: np.ndarray,
    target_ranges: np.ndarray | None = None,
) -> np.ndarray:
    """Scale thumb/index/middle/ring normalized targets into bounded angles."""
    values = np.asarray(targets, dtype=float)
    limits = np.asarray(joint_limits, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Expected a finite normalized joint target vector")
    if limits.ndim != 2 or limits.shape[1] != 2 or not np.isfinite(limits).all():
        raise ValueError("joint_limits must have shape (joint_count, 2)")
    if np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("Each joint limit must have a positive range")
    ranges = limits if target_ranges is None else np.asarray(target_ranges, dtype=float)
    if ranges.shape != limits.shape or not np.isfinite(ranges).all():
        raise ValueError("target_ranges must match the joint limit shape and be finite")
    if np.any(np.isclose(ranges[:, 0], ranges[:, 1])):
        raise ValueError("Each target range must have distinct open and closed positions")
    if np.any(ranges[:, 0] < limits[:, 0]) or np.any(ranges[:, 1] > limits[:, 1]):
        raise ValueError("target_ranges must lie within the robot joint limits")

    normalized = np.clip(values, 0.0, 1.0).copy()
    if len(normalized) == 16:
        for base_joint in (4, 8, 12):
            normalized[base_joint] = 0.5 + (normalized[base_joint] - 0.5) * 0.3
    return ranges[:, 0] + normalized * (ranges[:, 1] - ranges[:, 0])


def feature_targets_to_joint_positions(
    targets: dict[str, float],
    joint_limits: np.ndarray,
    joint_features: tuple[str, ...],
    target_ranges: np.ndarray,
) -> np.ndarray:
    """Map named normalized features to per-joint open/closed angle targets."""
    limits = np.asarray(joint_limits, dtype=float)
    ranges = np.asarray(target_ranges, dtype=float)
    if limits.ndim != 2 or limits.shape[1] != 2 or not np.isfinite(limits).all():
        raise ValueError("joint_limits must have shape (joint_count, 2) and be finite")
    if ranges.shape != limits.shape or not np.isfinite(ranges).all():
        raise ValueError("target_ranges must match joint_limits and be finite")
    if len(joint_features) != limits.shape[0]:
        raise ValueError("joint_features length must match joint count")
    if np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("Joint limits must have a positive range")
    if np.any(np.isclose(ranges[:, 0], ranges[:, 1])):
        raise ValueError("Each target range must have distinct open and closed positions")
    if np.any(ranges[:, 0] < limits[:, 0]) or np.any(ranges[:, 1] > limits[:, 1]):
        raise ValueError("target_ranges must lie within joint limits")

    normalized = np.asarray([targets[name] for name in joint_features], dtype=float)
    if not np.isfinite(normalized).all():
        raise ValueError("Feature targets must be finite")
    normalized = np.clip(normalized, 0.0, 1.0)
    return ranges[:, 0] + normalized * (ranges[:, 1] - ranges[:, 0])
