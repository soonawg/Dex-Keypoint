"""Convert MediaPipe hand landmarks into a robot-scale palm coordinate frame."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def human_fingertip_targets(
    landmarks: Iterable[Iterable[float]],
    palm_width_m: float,
    source_to_robot: Iterable[Iterable[float]],
) -> np.ndarray:
    """Return thumb/index/middle/ring fingertip targets in the robot base frame.

    Human landmark scale and orientation are normalized using the wrist,
    middle-finger MCP, and index-to-little MCP span before applying the
    robot-specific scale and rotation.
    """
    points = np.asarray(tuple(tuple(point) for point in landmarks), dtype=float)
    rotation = np.asarray(source_to_robot, dtype=float)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ValueError("Expected 21 finite landmarks, each with x, y, and z")
    if not np.isfinite(palm_width_m) or palm_width_m <= 0:
        raise ValueError("palm_width_m must be a positive finite value")
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("source_to_robot must be a finite 3x3 matrix")

    across = points[17] - points[5]
    across_norm = float(np.linalg.norm(across))
    if across_norm < 1e-6:
        raise ValueError("Landmarks do not define a usable palm width")
    x_axis = across / across_norm

    forward = points[9] - points[0]
    forward -= np.dot(forward, x_axis) * x_axis
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm < 1e-6:
        raise ValueError("Landmarks do not define a usable palm direction")
    y_axis = forward / forward_norm
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)
    basis = np.column_stack((x_axis, y_axis, z_axis))

    tip_ids = (4, 8, 12, 16)
    wrist_relative = points[list(tip_ids)] - points[0]
    normalized = wrist_relative @ basis / across_norm
    return normalized @ rotation.T * palm_width_m


def extract_named_controls(
    landmarks: Iterable[Iterable[float]],
    wrist_reference: tuple[float, float] | None = None,
) -> tuple[dict[str, float], tuple[float, float]]:
    """Extract normalized digit flexion and relative palm-tilt controls.

    Wrist controls are relative to the first valid frame (or an explicitly
    recalibrated frame) and saturate at approximately 30 degrees of tilt.
    """
    from dex_teleop.retargeting import extract_features

    points = np.asarray(tuple(tuple(point) for point in landmarks), dtype=float)
    features = extract_features(points)

    across = points[17] - points[5]
    across_norm = float(np.linalg.norm(across))
    if across_norm < 1e-6:
        raise ValueError("Landmarks do not define a usable palm width")
    across /= across_norm
    forward = points[9] - points[0]
    forward -= np.dot(forward, across) * across
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm < 1e-6:
        raise ValueError("Landmarks do not define a usable palm direction")
    forward /= forward_norm
    normal = np.cross(across, forward)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 1e-6:
        raise ValueError("Landmarks do not define a stable palm orientation")
    normal /= normal_norm

    angles = (
        float(np.arctan2(normal[0], max(abs(normal[2]), 1e-6))),
        float(np.arctan2(normal[1], max(abs(normal[2]), 1e-6))),
    )
    if wrist_reference is None:
        wrist_reference = angles
    if len(wrist_reference) != 2 or not np.isfinite(wrist_reference).all():
        raise ValueError("wrist_reference must contain two finite angles")
    wrist_delta = np.asarray(angles) - np.asarray(wrist_reference)
    wrist_normalized = np.clip(0.5 + wrist_delta / np.deg2rad(60.0), 0.0, 1.0)

    scale_curl = lambda value: float(np.clip(value / 0.65, 0.0, 1.0))
    controls = {
        "thumb_opposition": features.thumb_opposition,
        "thumb_curl": scale_curl(features.thumb_curl),
        "index_base": 0.5,
        "index_curl": scale_curl(features.finger_curl[0]),
        "middle_base": 0.5,
        "middle_curl": scale_curl(features.finger_curl[1]),
        "ring_base": 0.5,
        "ring_curl": scale_curl(features.finger_curl[2]),
        "little_base": 0.5,
        "little_curl": scale_curl(features.finger_curl[3]),
        "wrist_pitch": float(wrist_normalized[0]),
        "wrist_roll": float(wrist_normalized[1]),
    }
    return controls, (float(wrist_reference[0]), float(wrist_reference[1]))
