"""Camera/video retargeting preview with optional MuJoCo visualization."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from .config import load_hand_config
from .landmarks import extract_named_controls
from .mapping import (
    feature_targets_to_joint_positions,
    normalized_targets_to_joint_positions,
)
from .model import MujocoHandModel
from .session import RetargetingSession
from dex_teleop.frame_transport import CameraFrameServer
from dex_teleop.retargeting import extract_features, retarget_to_visual_joints
from dex_teleop.safety import TargetSmoother


WINDOW = "Dex-Retargeting preview"


def _profile_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "allegro_right.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retarget webcam or video hand landmarks to a MuJoCo hand model."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, help="camera device index (default: 0)")
    source.add_argument("--video", type=Path, help="video file to process")
    source.add_argument(
        "--tcp-camera",
        action="store_true",
        help="receive frames from the existing Windows camera sender",
    )
    parser.add_argument("--config", type=Path, default=_profile_path(), help="robot JSON profile")
    parser.add_argument("--listen-host", default="0.0.0.0", help="TCP listen address")
    parser.add_argument("--port", type=int, default=8765, help="TCP frame port")
    parser.add_argument(
        "--method",
        choices=("features", "ik"),
        default="features",
        help="retargeting method; features is the responsive baseline, ik is experimental",
    )
    parser.add_argument(
        "--handedness",
        choices=("Right", "Left", "any"),
        default="Right",
        help="MediaPipe hand label to retarget (default: Right)",
    )
    parser.add_argument("--no-viewer", action="store_true", help="skip the MuJoCo hand window")
    args = parser.parse_args()

    try:
        import cv2
        import mediapipe as mp
    except ImportError as error:
        raise RuntimeError(
            "Camera/video preview dependencies are missing; install with "
            "`python -m pip install -e '.[camera]'`"
        ) from error

    config = load_hand_config(args.config)
    if args.method == "ik" and config.joint_features is not None:
        raise ValueError("The IK method currently supports only 16-joint profiles without joint_features")
    session = RetargetingSession(config) if args.method == "ik" else None
    robot = session.robot if session is not None else MujocoHandModel(config)
    smoother = TargetSmoother(
        max_rate_normalized_s=5.0,
        smoothing_time_constant_s=0.10,
        joint_count=len(config.joint_names),
    )
    wrist_reference = None
    current_positions = robot.joint_limits.mean(axis=1)
    frame_server = CameraFrameServer(args.listen_host, args.port) if args.tcp_camera else None
    capture = None
    if frame_server is None:
        capture_source = str(args.video) if args.video is not None else (
            0 if args.camera is None else args.camera
        )
        capture = cv2.VideoCapture(capture_source)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Could not open frame source {capture_source!r}")
    else:
        host, port = frame_server.address
        print(f"Waiting for Windows camera sender on {host}:{port}")

    viewer = None
    viewer_data = None
    mujoco = None
    if not args.no_viewer:
        import mujoco.viewer
        import mujoco as mujoco_module

        mujoco = mujoco_module
        viewer_data = mujoco.MjData(robot.model)
        viewer_data.qpos[list(robot.qpos_addresses)] = current_positions
        mujoco.mj_forward(robot.model, viewer_data)
        viewer = mujoco.viewer.launch_passive(robot.model, viewer_data)

    try:
        previous_time = time.monotonic()
        with mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        ) as hands:
            while viewer is None or viewer.is_running():
                if frame_server is not None:
                    frame = frame_server.read()
                else:
                    ok, frame = capture.read()
                    if not ok:
                        if args.video is not None:
                            break
                        raise RuntimeError("Camera frame source stopped unexpectedly")
                now = time.monotonic()
                dt = min(now - previous_time, 0.1)
                previous_time = now

                frame = cv2.flip(frame, 1)
                result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                landmarks = None
                hand_label = "No hand"
                if result.multi_hand_landmarks:
                    candidate_label = "Unknown"
                    if result.multi_handedness:
                        candidate_label = result.multi_handedness[0].classification[0].label
                    hand_label = candidate_label
                    selected = args.handedness == "any" or candidate_label == args.handedness
                    if selected:
                        landmarks = tuple(
                            (point.x, point.y, point.z)
                            for point in result.multi_hand_landmarks[0].landmark
                        )
                        mp.solutions.drawing_utils.draw_landmarks(
                            frame,
                            result.multi_hand_landmarks[0],
                            mp.solutions.hands.HAND_CONNECTIONS,
                        )

                error = None
                if landmarks is not None:
                    if session is not None:
                        current_positions, tracked, error = session.update(landmarks)
                    else:
                        if config.joint_features is not None:
                            controls, wrist_reference = extract_named_controls(
                                landmarks,
                                wrist_reference,
                            )
                            requested = tuple(controls[name] for name in config.joint_features)
                            targets = smoother.update(requested, dt)
                            smoothed_controls = dict(zip(config.joint_features, targets))
                            current_positions = feature_targets_to_joint_positions(
                                smoothed_controls,
                                robot.joint_limits,
                                config.joint_features,
                                np.asarray(config.target_ranges, dtype=float),
                            )
                        else:
                            features = extract_features(landmarks)
                            requested = retarget_to_visual_joints(features)
                            targets = smoother.update(requested, dt)
                            current_positions = normalized_targets_to_joint_positions(
                                targets,
                                robot.joint_limits,
                                config.target_ranges,
                            )
                        tracked = True
                else:
                    tracked = False
                status = f"Tracking {hand_label}" if tracked else f"Holding last target ({hand_label})"
                if args.method == "features":
                    mapping_name = (
                        "finger + wrist mapping"
                        if config.joint_features is not None
                        else "flexion mapping"
                    )
                    status = f"{status} | {mapping_name}"
                color = (60, 220, 100) if tracked else (0, 180, 255)
                if error is not None:
                    status += f" | fingertip RMSE {error * 1000:.1f} mm"
                cv2.putText(
                    frame,
                    status,
                    (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    (
                        "c: calibrate wrist | q/Esc: quit"
                        if config.joint_features is not None
                        else "q/Esc: quit | simulation visualization only"
                    ),
                    (12, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow(WINDOW, frame)

                if viewer is not None:
                    with viewer.lock():
                        viewer_data.qpos[list(robot.qpos_addresses)] = current_positions
                        mujoco.mj_forward(robot.model, viewer_data)
                        viewer.sync()

                key = cv2.waitKey(1) & 0xFF
                if key == ord("c") and config.joint_features is not None:
                    wrist_reference = None
                if key in (ord("q"), 27):
                    break
    finally:
        if capture is not None:
            capture.release()
        if frame_server is not None:
            frame_server.close()
        if viewer is not None:
            viewer.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
