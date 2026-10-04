"""Bound and smooth visualization targets before displaying them."""

from __future__ import annotations

import math
from collections.abc import Sequence

from .retargeting import JOINT_COUNT, MAX_TARGET_NORMALIZED


class TargetSmoother:
    def __init__(self, max_rate_normalized_s: float = 2.0) -> None:
        if not math.isfinite(max_rate_normalized_s) or max_rate_normalized_s <= 0:
            raise ValueError("max_rate_normalized_s must be finite and positive")
        self.max_rate_normalized_s = max_rate_normalized_s
        self._targets: tuple[float, ...] | None = None

    def reset(self) -> None:
        self._targets = None

    def update(self, targets: Sequence[float], dt: float) -> tuple[float, ...]:
        if len(targets) != JOINT_COUNT:
            raise ValueError(f"Expected exactly {JOINT_COUNT} joint targets")
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

        max_step = self.max_rate_normalized_s * dt
        self._targets = tuple(
            current + min(max_step, max(-max_step, target - current))
            for current, target in zip(self._targets, requested)
        )
        return self._targets
