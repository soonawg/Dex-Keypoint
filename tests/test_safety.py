import unittest

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


if __name__ == "__main__":
    unittest.main()
