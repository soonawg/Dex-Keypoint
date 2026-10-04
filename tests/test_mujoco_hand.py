import unittest

import mujoco

from dex_teleop.mujoco_hand import (
    ACTUATOR_NAMES,
    count_goal_objects,
    load_hand_model,
    set_model_targets,
)
from dex_teleop.retargeting import JOINT_COUNT


class MuJoCoHandTests(unittest.TestCase):
    def test_model_has_sixteen_ordered_position_actuators(self):
        model, data, actuator_ids = load_hand_model()
        self.assertEqual(len(actuator_ids), JOINT_COUNT)
        self.assertEqual(model.nu, JOINT_COUNT)
        self.assertEqual(
            [model.actuator(i).name for i in actuator_ids],
            list(ACTUATOR_NAMES),
        )
        targets = tuple(index / (JOINT_COUNT - 1) for index in range(JOINT_COUNT))
        set_model_targets(model, data, actuator_ids, targets)
        expected = []
        for index, actuator_id in enumerate(actuator_ids):
            low, high = model.actuator_ctrlrange[actuator_id]
            normalized = targets[index]
            if index in (4, 8, 12):
                normalized = 0.5 + (normalized - 0.5) * 0.3
            expected.append(low + normalized * (high - low))
        self.assertTrue(
            all(
                abs(data.ctrl[actuator_id] - target) < 1e-9
                for actuator_id, target in zip(actuator_ids, expected)
            )
        )
        mujoco.mj_step(model, data)
        self.assertTrue(
            all(
                abs(data.ctrl[actuator_id] - target) < 1e-9
                for actuator_id, target in zip(actuator_ids, expected)
            )
        )

    def test_finger_closure_moves_the_gravity_dropped_object(self):
        model, data, actuator_ids = load_hand_model()
        object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
        positions = []
        for target in (0.0, 1.0):
            mujoco.mj_resetData(model, data)
            set_model_targets(model, data, actuator_ids, (target,) * JOINT_COUNT)
            for _ in range(1000):
                mujoco.mj_step(model, data)
            positions.append(data.xpos[object_id].copy())
            self.assertGreater(data.ncon, 0)

        self.assertGreater(float(((positions[1] - positions[0]) ** 2).sum()) ** 0.5, 0.005)

    def test_scene_has_four_objects_and_goal_zone(self):
        model, data, _ = load_hand_model()
        body_ids = tuple(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ("object", "cube", "ball", "capsule")
        )
        self.assertTrue(all(body_id >= 0 for body_id in body_ids))
        self.assertGreaterEqual(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "goal_zone"), 0
        )
        self.assertEqual(count_goal_objects(data, body_ids), 0)

        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
        qpos = int(model.jnt_qposadr[joint_id])
        data.qpos[qpos:qpos + 3] = (0.09, 0.0, 0.03)
        mujoco.mj_forward(model, data)
        self.assertEqual(count_goal_objects(data, body_ids), 1)

    def test_allegro_base_is_fixed_for_finger_teleoperation_task(self):
        model, data, _ = load_hand_model()
        self.assertEqual(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hand_root"),
            -1,
        )
        self.assertEqual(data.qpos.size, JOINT_COUNT + 4 * 7)


if __name__ == "__main__":
    unittest.main()
