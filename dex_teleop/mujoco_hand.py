"""MuJoCo Menagerie Allegro Hand V3 right-hand model.

The bundled MJCF and meshes are from Google DeepMind MuJoCo Menagerie and are
redistributed under the accompanying BSD-2-Clause license.
"""

from __future__ import annotations

from pathlib import Path
import time

import mujoco
import mujoco.viewer

from .retargeting import JOINT_COUNT


ACTUATOR_NAMES = tuple(
    f"{finger}{joint}"
    for finger in ("tha", "ffa", "mfa", "rfa")
    for joint in range(4)
)


def load_hand_model() -> tuple[mujoco.MjModel, mujoco.MjData, tuple[int, ...]]:
    xml_path = Path(__file__).with_name("assets") / "wonik_allegro" / "scene_right.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    actuator_ids = []
    for name in ACTUATOR_NAMES:
        actuator_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
        )
        if actuator_id < 0:
            raise ValueError(f"MuJoCo model is missing actuator {name!r}")
        actuator_ids.append(actuator_id)
    if len(actuator_ids) != JOINT_COUNT:
        raise ValueError(f"Expected {JOINT_COUNT} actuators")
    return model, data, tuple(actuator_ids)


def set_model_targets(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    actuator_ids: tuple[int, ...],
    targets: tuple[float, ...],
) -> None:
    if len(targets) != JOINT_COUNT or len(actuator_ids) != JOINT_COUNT:
        raise ValueError(f"Expected exactly {JOINT_COUNT} targets and actuators")
    for index, (actuator_id, target) in enumerate(zip(actuator_ids, targets)):
        low, high = model.actuator_ctrlrange[actuator_id]
        normalized = min(1.0, max(0.0, target))
        if index in (4, 8, 12):
            normalized = 0.5 + (normalized - 0.5) * 0.3
        data.ctrl[actuator_id] = low + normalized * (high - low)


def count_goal_objects(
    data: mujoco.MjData,
    body_ids: tuple[int, ...],
    goal_xy: tuple[float, float] = (0.09, 0.0),
    radius: float = 0.055,
) -> int:
    return sum(
        (data.xpos[body_id, 0] - goal_xy[0]) ** 2
        + (data.xpos[body_id, 1] - goal_xy[1]) ** 2 < radius**2
        and data.xpos[body_id, 2] < 0.06
        for body_id in body_ids
    )


class MuJoCoHand:
    def __init__(self) -> None:
        self.model, self.data, self._actuator_ids = load_hand_model()
        self._objects = {}
        for name in ("object", "cube", "ball", "capsule"):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_free")
            if joint_id < 0:
                raise ValueError(f"Scene is missing the {name!r} free joint")
            self._objects[name] = (
                int(self.model.jnt_qposadr[joint_id]),
                int(self.model.jnt_dofadr[joint_id]),
            )
        self._object_body_ids = tuple(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in self._objects
        )
        self._last_step = time.monotonic()
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def set_targets(self, targets: tuple[float, ...]) -> None:
        with self.viewer.lock():
            set_model_targets(self.model, self.data, self._actuator_ids, targets)

    def step(self) -> None:
        now = time.monotonic()
        elapsed = min(now - self._last_step, 0.05)
        self._last_step = now
        timestep = self.model.opt.timestep
        steps = max(1, min(30, int(elapsed / timestep)))
        with self.viewer.lock():
            for _ in range(steps):
                mujoco.mj_step(self.model, self.data)
            self.viewer.sync()

    def reset(self) -> None:
        with self.viewer.lock():
            mujoco.mj_resetData(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            self.viewer.sync()

    @property
    def goal_count(self) -> int:
        with self.viewer.lock():
            return count_goal_objects(self.data, self._object_body_ids)

    def is_running(self) -> bool:
        return self.viewer.is_running()

    def close(self) -> None:
        self.viewer.close()
