import unittest
import math

from dex_teleop.safety import TargetSmoother


class SmootherTests(unittest.TestCase):
    def test_clamps_and_limits_target_rate(self):
        smoother = TargetSmoother(max_rate_normalized_s=1.0)
        self.assertEqual(smoother.update((2.0,) * 16, 0.01), (1.0,) * 16)
        next_targets = smoother.update((0.0,) * 16, 0.1)
        self.assertEqual(next_targets, (0.9,) * 16)

    def test_rejects_wrong_size_and_non_finite_values(self):
        smoother = TargetSmoother()
        with self.assertRaises(ValueError):
            smoother.update((0.0,) * 15, 0.1)
        with self.assertRaises(ValueError):
            smoother.update((float("nan"),) + (0.0,) * 15, 0.1)

    def test_low_pass_filter_reduces_small_target_jitter(self):
        smoother = TargetSmoother(
            max_rate_normalized_s=100.0,
            smoothing_time_constant_s=0.1,
        )
        center = smoother.update((0.5,) * 16, 0.02)
        high = smoother.update((0.6,) * 16, 0.02)
        low = smoother.update((0.4,) * 16, 0.02)

        self.assertAlmostEqual(high[0], 0.5 + (1.0 - math.exp(-0.2)) * 0.1)
        self.assertLess(abs(low[0] - high[0]), 0.03)
        self.assertNotEqual(center, high)

    def test_rejects_invalid_smoothing_time_constant(self):
        with self.assertRaises(ValueError):
            TargetSmoother(smoothing_time_constant_s=-0.01)


if __name__ == "__main__":
    unittest.main()
