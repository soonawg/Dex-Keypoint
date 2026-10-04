"""Stateful landmark-to-joint retargeting session."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .config import HandConfig
from .landmarks import human_fingertip_targets
from .model import MujocoHandModel
from .optimizer import KinematicRetargeter


class RetargetingSession:
    """Keep the last valid joint solution when landmarks are temporarily lost."""

    def __init__(self, config: HandConfig, temporal_weight: float = 0.02) -> None:
        self.config = config
        self.robot = MujocoHandModel(config)
        self.optimizer = KinematicRetargeter(
            self.robot.forward_kinematics,
            self.robot.joint_limits,
            temporal_weight=temporal_weight,
        )
        self.current_positions = self.optimizer.current_positions

    def update(
        self, landmarks: Iterable[Iterable[float]] | None
    ) -> tuple[np.ndarray, bool, float | None]:
        if landmarks is None:
            return self.current_positions.copy(), False, None
        targets = human_fingertip_targets(
            landmarks,
            self.config.palm_width_m,
            self.config.source_to_robot,
        )
        positions = self.optimizer.retarget(targets)
        actual_tips = self.robot.forward_kinematics(positions)
        error = float(np.sqrt(np.mean(np.sum((actual_tips - targets) ** 2, axis=1))))
        self.current_positions = positions.copy()
        return positions, True, error
