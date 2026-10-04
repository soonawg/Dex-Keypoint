"""Bound and smooth visualization targets before displaying them."""

from __future__ import annotations

import math
from collections.abc import Sequence

from .retargeting import JOINT_COUNT, MAX_TARGET_NORMALIZED


class TargetSmoother:
    def __init__(
        self,
        max_rate_normalized_s: float = 2.0,
        smoothing_time_constant_s: float = 0.0,
        joint_count: int = JOINT_COUNT,
    ) -> None:
        if not math.isfinite(max_rate_normalized_s) or max_rate_normalized_s <= 0:
            raise ValueError("max_rate_normalized_s must be finite and positive")
        if not math.isfinite(smoothing_time_constant_s) or smoothing_time_constant_s < 0:
            raise ValueError("smoothing_time_constant_s must be finite and non-negative")
        if joint_count < 1:
            raise ValueError("joint_count must be positive")
        self.max_rate_normalized_s = max_rate_normalized_s
        self.smoothing_time_constant_s = smoothing_time_constant_s
        self.joint_count = joint_count
        self._targets: tuple[float, ...] | None = None

    def reset(self) -> None:
        self._targets = None

    def update(self, targets: Sequence[float], dt: float) -> tuple[float, ...]:
        if len(targets) != self.joint_count:
            raise ValueError(f"Expected exactly {self.joint_count} joint targets")
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and non-negative")
        if not all(math.isfinite(value) for value in targets):
            raise ValueError("Joint targets must all be finite")

        requested = tuple(
            min(MAX_TARGET_NORMALIZED, max(0.0, float(value)))
            for value in targets
        )
        if self._targets is None:
            self._targets = requested
            return self._targets

        if self.smoothing_time_constant_s > 0 and dt > 0:
            alpha = 1.0 - math.exp(-dt / self.smoothing_time_constant_s)
            requested = tuple(
                current + alpha * (target - current)
                for current, target in zip(self._targets, requested)
            )
        max_step = self.max_rate_normalized_s * dt
        self._targets = tuple(
            current + min(max_step, max(-max_step, target - current))
            for current, target in zip(self._targets, requested)
        )
        return self._targets
