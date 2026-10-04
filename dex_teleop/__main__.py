from __future__ import annotations

import argparse
import time

import cv2
import mediapipe as mp

from .frame_transport import CameraFrameServer
from .mujoco_hand import MuJoCoHand
from .retargeting import extract_features, retarget_to_visual_joints
from .safety import TargetSmoother


WINDOW = "Dex-Keypoint-Teleop (visualization only)"


def _draw_hand(frame, landmarks, width: int, height: int) -> None:
    points = [
        (int(point.x * width), int(point.y * height))
        for point in landmarks
    ]
    for start, end in mp.solutions.hands.HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (70, 210, 100), 2)
    for point in points:
        cv2.circle(frame, point, 4, (40, 80, 255), -1)


def _draw_targets(frame, targets: tuple[float, ...]) -> None:
    height, width = frame.shape[:2]
    panel_width = 280
    cv2.rectangle(frame, (width - panel_width, 0), (width, height), (24, 24, 24), -1)
    cv2.putText(frame, "Joint target (normalized)", (width - 265, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)
    names = ("Thumb", "Index", "Middle", "Ring")
    for finger, name in enumerate(names):
        base_y = 70 + finger * 150
        cv2.putText(frame, name, (width - 265, base_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (180, 220, 255), 1, cv2.LINE_AA)
        for joint in range(4):
            index = finger * 4 + joint
            value = targets[index]
            y = base_y + 16 + joint * 20
            cv2.putText(frame, f"J{joint + 1}", (width - 265, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 210), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (width - 220, y - 5), (width - 105, y + 5),
                          (65, 65, 65), -1)
            bar_end = width - 220 + int(value * 115)
            cv2.rectangle(frame, (width - 220, y - 5), (bar_end, y + 5),
                          (60, 190, 240), -1)
            cv2.putText(frame, f"{value:.2f}", (width - 95, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (240, 240, 240), 1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize webcam hand tracking and provisional Allegro targets.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, help="local camera device index")
    source.add_argument("--tcp-camera", action="store_true", help="receive frames from the Windows camera sender")
    parser.add_argument("--mujoco", action="store_true", help="drive the Allegro MuJoCo manipulation scene")
    parser.add_argument("--listen-host", default="0.0.0.0", help="TCP listen address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="TCP frame port (default: 8765)")
    args = parser.parse_args()

    frame_server = None
    camera = None
    simulator = MuJoCoHand() if args.mujoco else None
    if args.tcp_camera:
        frame_server = CameraFrameServer(args.listen_host, args.port)
        print(f"Waiting for Windows camera sender on {args.listen_host}:{args.port}")
    else:
        camera_index = 0 if args.camera is None else args.camera
        camera = cv2.VideoCapture(camera_index)
        if not camera.isOpened():
            camera.release()
            raise RuntimeError(f"Could not open camera device {camera_index}")

    smoother = TargetSmoother()
    targets = (0.0,) * 16
    previous_time = time.monotonic()
    try:
        with mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        ) as hands:
            while simulator is None or simulator.is_running():
                if frame_server is not None:
                    frame = frame_server.read()
                    ok = frame is not None
                else:
                    ok, frame = camera.read()
                now = time.monotonic()
                dt = min(now - previous_time, 0.1)
                previous_time = now
                if not ok:
                    raise RuntimeError("Camera frame source stopped unexpectedly")

                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                status = "No hand detected"
                color = (0, 180, 255)
                if result.multi_hand_landmarks:
                    hand = result.multi_hand_landmarks[0]
                    _draw_hand(frame, hand.landmark, width, height)
                    points = ((point.x, point.y, point.z) for point in hand.landmark)
                    features = extract_features(points)
                    targets = smoother.update(retarget_to_visual_joints(features), dt)
                    handedness = "Unknown"
                    if result.multi_handedness:
                        handedness = result.multi_handedness[0].classification[0].label
                    status = f"Tracking {handedness} hand"
                    color = (60, 220, 100)
                else:
                    targets = smoother.update((0.0,) * 16, dt)

                if simulator is not None:
                    simulator.set_targets(targets)
                    simulator.step()
                    caption = f"{status} | Goal: {simulator.goal_count}/4 objects"
                else:
                    caption = f"{status} | Visualization only - no robot commands"
                cv2.putText(frame, caption, (16, 32), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, color, 2, cv2.LINE_AA)
                controls = (
                    "Move fingers to guide objects | r reset | q/Esc quit"
                    if simulator is not None else "q/Esc: quit"
                )
                cv2.putText(frame, controls, (16, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                _draw_targets(frame, targets)
                cv2.imshow(WINDOW, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("r") and simulator is not None:
                    smoother.reset()
                    simulator.reset()
                if key in (ord("q"), 27):
                    break
    finally:
        if camera is not None:
            camera.release()
        if frame_server is not None:
            frame_server.close()
        if simulator is not None:
            simulator.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
