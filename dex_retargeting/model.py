"""MuJoCo forward-kinematics adapter for configurable hand models."""

from __future__ import annotations

import numpy as np
import mujoco

from .config import HandConfig


class MujocoHandModel:
    def __init__(self, config: HandConfig) -> None:
        self.config = config
        self.model = mujoco.MjModel.from_xml_path(str(config.model_path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)
        self.joint_ids = tuple(self._named_id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in config.joint_names)
        self.qpos_addresses = tuple(int(self.model.jnt_qposadr[joint_id]) for joint_id in self.joint_ids)
        self.joint_limits = np.asarray(
            [self.model.jnt_range[joint_id] for joint_id in self.joint_ids],
            dtype=float,
        )
        self.fingertip_body_ids = tuple(
            self._named_id(mujoco.mjtObj.mjOBJ_BODY, name)
            for name in config.fingertip_body_names
        )
        self._validate_model()

    def _named_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"Robot model is missing {object_type.name} {name!r}")
        return object_id

    def _validate_model(self) -> None:
        for joint_id, joint_name in zip(self.joint_ids, self.config.joint_names):
            joint_type = int(self.model.jnt_type[joint_id])
            if joint_type not in (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            ):
                raise ValueError(f"Joint {joint_name!r} must be a scalar hinge or slide joint")
            if not self.model.jnt_limited[joint_id]:
                raise ValueError(f"Joint {joint_name!r} must have finite limits")
        if not np.isfinite(self.joint_limits).all() or np.any(self.joint_limits[:, 0] >= self.joint_limits[:, 1]):
            raise ValueError("Robot model has invalid joint limits")
        if self.config.target_ranges is not None:
            target_ranges = np.asarray(self.config.target_ranges, dtype=float)
            if np.any(target_ranges[:, 0] < self.joint_limits[:, 0]) or np.any(
                target_ranges[:, 1] > self.joint_limits[:, 1]
            ):
                raise ValueError("Configured target ranges exceed robot joint limits")
        mujoco.mj_forward(self.model, self.data)

    def forward_kinematics(self, joint_positions: np.ndarray) -> np.ndarray:
        positions = np.asarray(joint_positions, dtype=float)
        if positions.shape != (len(self.joint_ids),) or not np.isfinite(positions).all():
            raise ValueError(f"Expected {len(self.joint_ids)} finite joint positions")
        self.data.qpos[list(self.qpos_addresses)] = positions
        mujoco.mj_forward(self.model, self.data)
        return np.asarray(self.data.xpos[list(self.fingertip_body_ids)], dtype=float).copy()
