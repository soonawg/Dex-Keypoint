"""Bounded, temporally regularized fingertip retargeting."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.optimize import least_squares


class KinematicRetargeter:
    """Fit robot fingertip positions to human targets using bounded least squares."""

    def __init__(
        self,
        forward_kinematics: Callable[[np.ndarray], np.ndarray],
        joint_limits: np.ndarray,
        fingertip_weights: np.ndarray | None = None,
        temporal_weight: float = 0.02,
        max_nfev: int = 40,
    ) -> None:
        limits = np.asarray(joint_limits, dtype=float)
        if limits.ndim != 2 or limits.shape[1] != 2 or limits.shape[0] == 0:
            raise ValueError("joint_limits must have shape (joint_count, 2)")
        if not np.isfinite(limits).all() or np.any(limits[:, 0] >= limits[:, 1]):
            raise ValueError("joint_limits must contain finite increasing bounds")
        if not np.isfinite(temporal_weight) or temporal_weight < 0:
            raise ValueError("temporal_weight must be finite and non-negative")
        if max_nfev < 1:
            raise ValueError("max_nfev must be positive")

        self.forward_kinematics = forward_kinematics
        self.joint_limits = limits
        self.temporal_weight = float(temporal_weight)
        self.max_nfev = int(max_nfev)
        self.fingertip_weights = None if fingertip_weights is None else np.asarray(fingertip_weights, dtype=float)
        if self.fingertip_weights is not None and (
            self.fingertip_weights.ndim != 1
            or not np.isfinite(self.fingertip_weights).all()
            or np.any(self.fingertip_weights <= 0)
        ):
            raise ValueError("fingertip_weights must be a finite positive vector")
        self._previous = limits.mean(axis=1)

    @property
    def current_positions(self) -> np.ndarray:
        return self._previous.copy()

    def reset(self, joint_positions: np.ndarray | None = None) -> None:
        if joint_positions is None:
            self._previous = self.joint_limits.mean(axis=1)
            return
        positions = np.asarray(joint_positions, dtype=float)
        if positions.shape != self._previous.shape or not np.isfinite(positions).all():
            raise ValueError("Reset positions must match the robot joint count and be finite")
        if np.any(positions < self.joint_limits[:, 0]) or np.any(positions > self.joint_limits[:, 1]):
            raise ValueError("Reset positions must lie inside joint limits")
        self._previous = positions.copy()

    def retarget(self, target_points: np.ndarray) -> np.ndarray:
        targets = np.asarray(target_points, dtype=float)
        if targets.ndim != 2 or targets.shape[1] != 3 or targets.shape[0] == 0:
            raise ValueError("target_points must have shape (tip_count, 3)")
        if not np.isfinite(targets).all():
            raise ValueError("target_points must be finite")
        initial_points = np.asarray(self.forward_kinematics(self._previous), dtype=float)
        if initial_points.shape != targets.shape or not np.isfinite(initial_points).all():
            raise ValueError("Forward kinematics output must match target_points shape and be finite")

        weights = (
            np.ones(targets.shape[0], dtype=float)
            if self.fingertip_weights is None
            else self.fingertip_weights
        )
        if weights.shape != (targets.shape[0],):
            raise ValueError("fingertip_weights length must match target tip count")
        limits = self.joint_limits

        def residual(positions: np.ndarray) -> np.ndarray:
            tips = np.asarray(self.forward_kinematics(positions), dtype=float)
            if tips.shape != targets.shape or not np.isfinite(tips).all():
                raise ValueError("Forward kinematics returned invalid fingertip positions")
            tip_error = ((tips - targets) * weights[:, None]).ravel()
            temporal_error = (positions - self._previous) * self.temporal_weight
            return np.concatenate((tip_error, temporal_error))

        result = least_squares(
            residual,
            np.clip(self._previous, limits[:, 0] + 1e-9, limits[:, 1] - 1e-9),
            bounds=(limits[:, 0], limits[:, 1]),
            max_nfev=self.max_nfev,
        )
        if result.status < 0:
            raise RuntimeError(f"Retargeting optimization failed: {result.message}")
        if (
            not np.isfinite(result.x).all()
            or np.any(result.x < limits[:, 0])
            or np.any(result.x > limits[:, 1])
        ):
            raise RuntimeError("Retargeting optimization returned invalid joint positions")
        self._previous = result.x.copy()
        return result.x.copy()
