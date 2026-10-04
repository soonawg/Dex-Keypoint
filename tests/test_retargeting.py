import unittest
from pathlib import Path

import numpy as np

from dex_retargeting import (
    KinematicRetargeter,
    MujocoHandModel,
    RetargetingSession,
    human_fingertip_targets,
    load_hand_config,
)
from dex_retargeting.mapping import normalized_targets_to_joint_positions
from dex_retargeting.mapping import feature_targets_to_joint_positions
from dex_retargeting.landmarks import extract_named_controls


ROOT = Path(__file__).resolve().parents[1]


class RetargetingTests(unittest.TestCase):
    def test_human_targets_are_palm_normalized_and_scaled(self):
        points = np.zeros((21, 3), dtype=float)
        points[0] = (0.5, 0.9, 0.0)
        points[5] = (0.4, 0.68, 0.0)
        points[9] = (0.5, 0.68, 0.0)
        points[17] = (0.6, 0.68, 0.0)
        points[4] = (0.3, 0.6, 0.0)
        points[8] = (0.4, 0.4, 0.0)
        points[12] = (0.5, 0.3, 0.0)
        points[16] = (0.6, 0.4, 0.0)

        targets = human_fingertip_targets(points, 0.1, np.eye(3))

        self.assertEqual(targets.shape, (4, 3))
        np.testing.assert_allclose(targets[1], (-0.05, 0.25, 0.0))

    def test_optimizer_respects_joint_limits_and_tracks_targets(self):
        def fk(q):
            return np.asarray(((q[0], 0.0, 0.0), (0.0, q[1], 0.0)))

        optimizer = KinematicRetargeter(fk, np.array([[0.0, 1.0], [0.0, 1.0]]))
        result = optimizer.retarget(np.array(((0.7, 0.0, 0.0), (0.0, 0.3, 0.0))))

        np.testing.assert_allclose(result, (0.7, 0.3), atol=1e-3)
        self.assertTrue(np.all(result >= 0.0))
        self.assertTrue(np.all(result <= 1.0))

    def test_optimizer_returns_best_bounded_result_when_evaluation_budget_is_reached(self):
        def fk(q):
            return np.asarray(((q[0] ** 2, 0.0, 0.0),))

        optimizer = KinematicRetargeter(
            fk,
            np.array([[0.0, 1.0]]),
            temporal_weight=0.0,
            max_nfev=1,
        )
        result = optimizer.retarget(np.array(((0.8, 0.0, 0.0),)))

        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(0.0 <= result[0] <= 1.0)

    def test_flexion_mapping_moves_fingers_and_stays_inside_model_limits(self):
        config = load_hand_config(ROOT / "configs" / "allegro_right.json")
        hand = MujocoHandModel(config)
        open_targets = normalized_targets_to_joint_positions(
            (0.0, 0.0, 0.0, 0.0)
            + (0.5, 0.0, 0.0, 0.0) * 3,
            hand.joint_limits,
        )
        closed_targets = normalized_targets_to_joint_positions(
            (1.0,) * 16,
            hand.joint_limits,
        )

        self.assertTrue(np.all(open_targets >= hand.joint_limits[:, 0]))
        self.assertTrue(np.all(open_targets <= hand.joint_limits[:, 1]))
        self.assertTrue(np.all(closed_targets >= hand.joint_limits[:, 0]))
        self.assertTrue(np.all(closed_targets <= hand.joint_limits[:, 1]))
        self.assertGreater(float(np.linalg.norm(closed_targets - open_targets)), 1.0)

    def test_allegro_profile_loads_and_provides_bounded_forward_kinematics(self):
        config = load_hand_config(ROOT / "configs" / "allegro_right.json")
        hand = MujocoHandModel(config)
        angles = hand.joint_limits.mean(axis=1)
        tips = hand.forward_kinematics(angles)

        self.assertEqual(len(hand.joint_ids), 16)
        self.assertEqual(tips.shape, (4, 3))
        self.assertTrue(np.isfinite(tips).all())

    def test_leap_profile_loads_and_provides_bounded_forward_kinematics(self):
        config = load_hand_config(ROOT / "configs" / "leap_hand.json")
        hand = MujocoHandModel(config)
        positions = hand.joint_limits.mean(axis=1)
        tips = hand.forward_kinematics(positions)

        self.assertEqual(len(hand.joint_ids), 16)
        self.assertEqual(tips.shape, (4, 3))
        self.assertTrue(np.isfinite(tips).all())
        self.assertEqual(config.joint_names[:4], ("12", "13", "14", "15"))

    def test_shadow_hand_maps_all_24_joints_including_little_finger_and_wrist(self):
        config = load_hand_config(ROOT / "configs" / "shadow_hand.json")
        hand = MujocoHandModel(config)
        self.assertEqual(len(hand.joint_ids), 24)
        self.assertEqual(len(config.joint_names), 24)

        controls = {
            "thumb_opposition": 0.6,
            "thumb_curl": 0.7,
            "index_base": 0.5,
            "index_curl": 0.4,
            "middle_base": 0.5,
            "middle_curl": 0.5,
            "ring_base": 0.5,
            "ring_curl": 0.6,
            "little_base": 0.5,
            "little_curl": 0.8,
            "wrist_pitch": 0.75,
            "wrist_roll": 0.25,
        }
        positions = feature_targets_to_joint_positions(
            controls,
            hand.joint_limits,
            config.joint_features,
            np.asarray(config.target_ranges),
        )

        self.assertTrue(np.isfinite(positions).all())
        self.assertTrue(np.all(positions >= hand.joint_limits[:, 0]))
        self.assertTrue(np.all(positions <= hand.joint_limits[:, 1]))
        self.assertNotAlmostEqual(positions[-2], hand.joint_limits[-2].mean())
        self.assertNotAlmostEqual(positions[-1], hand.joint_limits[-1].mean())

    def test_named_controls_calibrate_neutral_wrist_and_track_little_finger(self):
        points = np.zeros((21, 3), dtype=float)
        points[0] = (0.5, 0.9, 0.0)
        for mcp, x in ((5, 0.4), (9, 0.48), (13, 0.56), (17, 0.64)):
            points[mcp] = (x, 0.68, 0.0)
            points[mcp + 1] = (x, 0.56, 0.0)
            points[mcp + 2] = (x, 0.43, 0.0)
            points[mcp + 3] = (x, 0.30, 0.0)
        points[1:5] = (
            (0.43, 0.82, 0.0),
            (0.39, 0.74, 0.0),
            (0.35, 0.68, 0.0),
            (0.31, 0.62, 0.0),
        )
        points[18] = (0.64, 0.56, 0.0)
        points[19] = (0.72, 0.56, 0.0)
        points[20] = (0.82, 0.56, 0.0)

        controls, neutral = extract_named_controls(points)
        recalibrated, same_neutral = extract_named_controls(points, neutral)

        self.assertIn("little_curl", controls)
        self.assertGreater(controls["little_curl"], controls["index_curl"])
        self.assertAlmostEqual(controls["wrist_pitch"], 0.5)
        self.assertAlmostEqual(controls["wrist_roll"], 0.5)
        self.assertEqual(recalibrated, controls)
        self.assertEqual(same_neutral, neutral)

    def test_rejects_invalid_targets(self):
        optimizer = KinematicRetargeter(
            lambda q: np.zeros((1, 3)),
            np.array([[0.0, 1.0]]),
        )
        with self.assertRaises(ValueError):
            optimizer.retarget(np.array([[float("nan"), 0.0, 0.0]]))

    def test_tracking_loss_holds_the_last_valid_joint_solution(self):
        config = load_hand_config(ROOT / "configs" / "allegro_right.json")
        session = RetargetingSession(config)
        before = session.current_positions.copy()
        positions, tracked, error = session.update(None)

        np.testing.assert_array_equal(positions, before)
        self.assertFalse(tracked)
        self.assertIsNone(error)

    def test_session_maps_landmarks_to_valid_robot_joint_positions(self):
        points = np.zeros((21, 3), dtype=float)
        points[0] = (0.5, 0.9, 0.0)
        points[5] = (0.4, 0.68, 0.0)
        points[9] = (0.48, 0.68, 0.0)
        points[17] = (0.64, 0.68, 0.0)
        for mcp, x in ((5, 0.4), (9, 0.48), (13, 0.56), (17, 0.64)):
            points[mcp] = (x, 0.68, 0.0)
            points[mcp + 1] = (x, 0.56, 0.0)
            points[mcp + 2] = (x, 0.43, 0.0)
            points[mcp + 3] = (x, 0.30, 0.0)
        points[1:5] = (
            (0.43, 0.82, 0.0),
            (0.39, 0.74, 0.0),
            (0.35, 0.68, 0.0),
            (0.31, 0.62, 0.0),
        )

        config = load_hand_config(ROOT / "configs" / "allegro_right.json")
        session = RetargetingSession(config)
        positions, tracked, error = session.update(points)

        self.assertTrue(tracked)
        self.assertTrue(np.isfinite(error))
        self.assertTrue(np.all(positions >= session.robot.joint_limits[:, 0]))
        self.assertTrue(np.all(positions <= session.robot.joint_limits[:, 1]))


if __name__ == "__main__":
    unittest.main()
