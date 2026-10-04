import unittest

import numpy as np

from dex_teleop.retargeting import (
    HandFeatures,
    JOINT_COUNT,
    extract_features,
    retarget_to_visual_joints,
)


def extended_hand():
    points = np.zeros((21, 3), dtype=float)
    points[0] = (0.5, 0.9, 0.0)
    points[1:5] = ((0.43, 0.82, 0), (0.39, 0.74, 0), (0.35, 0.68, 0), (0.31, 0.62, 0))
    for mcp, x in ((5, 0.4), (9, 0.48), (13, 0.56), (17, 0.64)):
        points[mcp] = (x, 0.68, 0)
        points[mcp + 1] = (x, 0.56, 0)
        points[mcp + 2] = (x, 0.43, 0)
        points[mcp + 3] = (x, 0.30, 0)
    return points


class RetargetingTests(unittest.TestCase):
    def test_extended_fingers_produce_bounded_targets(self):
        targets = retarget_to_visual_joints(extract_features(extended_hand()))
        self.assertEqual(len(targets), JOINT_COUNT)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in targets))
        for finger in range(1, 4):
            self.assertAlmostEqual(targets[finger * 4], 0.5)
            self.assertTrue(all(value < 0.1 for value in targets[finger * 4 + 1:finger * 4 + 4]))

    def test_rejects_invalid_landmark_shapes_and_values(self):
        with self.assertRaises(ValueError):
            extract_features([(0, 0, 0)] * 20)
        points = extended_hand()
        points[3, 0] = float("nan")
        with self.assertRaises(ValueError):
            extract_features(points)

    def test_full_human_curl_uses_full_allegro_flexion_range(self):
        targets = retarget_to_visual_joints(
            HandFeatures(
                finger_curl=(0.65, 0.8, 1.0, 0.9),
                thumb_curl=0.8,
                thumb_opposition=1.0,
            )
        )
        self.assertEqual(targets[0:4], (1.0, 1.0, 1.0, 1.0))
        for finger in range(1, 4):
            self.assertEqual(targets[finger * 4 + 1:finger * 4 + 4], (1.0, 1.0, 1.0))

    def test_open_hand_stays_at_open_target(self):
        targets = retarget_to_visual_joints(
            HandFeatures(
                finger_curl=(0.0, 0.0, 0.0, 0.0),
                thumb_curl=0.0,
                thumb_opposition=0.0,
            )
        )
        self.assertEqual(targets[0:4], (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(targets[4:], (0.5, 0.0, 0.0, 0.0) * 3)


if __name__ == "__main__":
    unittest.main()
