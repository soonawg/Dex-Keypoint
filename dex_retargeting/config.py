"""Robot-specific settings loaded from small JSON profiles."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class HandConfig:
    model_path: Path
    joint_names: tuple[str, ...]
    fingertip_body_names: tuple[str, ...]
    palm_width_m: float
    source_to_robot: tuple[tuple[float, float, float], ...]
    target_ranges: tuple[tuple[float, float], ...] | None = None
    joint_features: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not self.model_path.is_file():
            raise ValueError(f"Robot model does not exist: {self.model_path}")
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must be a non-empty list of unique names")
        if not self.fingertip_body_names or len(set(self.fingertip_body_names)) != len(self.fingertip_body_names):
            raise ValueError("fingertip_body_names must be a non-empty list of unique names")
        if not np.isfinite(self.palm_width_m) or self.palm_width_m <= 0:
            raise ValueError("palm_width_m must be a positive finite value")
        rotation = np.asarray(self.source_to_robot, dtype=float)
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise ValueError("source_to_robot must be a finite 3x3 matrix")
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-4):
            raise ValueError("source_to_robot must be an orthonormal rotation matrix")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4):
            raise ValueError("source_to_robot must be a right-handed rotation matrix")
        if self.target_ranges is not None:
            target_ranges = np.asarray(self.target_ranges, dtype=float)
            if target_ranges.shape != (len(self.joint_names), 2) or not np.isfinite(target_ranges).all():
                raise ValueError("target_ranges must provide one finite open/closed pair per joint")
            if np.any(np.isclose(target_ranges[:, 0], target_ranges[:, 1])):
                raise ValueError("Each target range must have distinct open and closed positions")
        if self.joint_features is not None:
            if len(self.joint_features) != len(self.joint_names):
                raise ValueError("joint_features must provide one feature name per joint")
            if self.target_ranges is None:
                raise ValueError("joint_features require target_ranges")


def load_hand_config(path: str | Path) -> HandConfig:
    """Load a robot profile; model paths are resolved relative to that profile."""
    config_path = Path(path).resolve()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    required = {
        "model_path",
        "joint_names",
        "fingertip_body_names",
        "palm_width_m",
        "source_to_robot",
    }
    missing = required - data.keys()
    if missing:
        raise ValueError(f"Missing hand configuration fields: {', '.join(sorted(missing))}")
    return HandConfig(
        model_path=(config_path.parent / data["model_path"]).resolve(),
        joint_names=tuple(data["joint_names"]),
        fingertip_body_names=tuple(data["fingertip_body_names"]),
        palm_width_m=float(data["palm_width_m"]),
        source_to_robot=tuple(tuple(row) for row in data["source_to_robot"]),
        target_ranges=(
            None
            if "target_ranges" not in data
            else tuple(tuple(row) for row in data["target_ranges"])
        ),
        joint_features=(
            None
            if "joint_features" not in data
            else tuple(data["joint_features"])
        ),
    )
