"""Capture a Windows webcam and stream JPEG frames to the WSL app."""

from __future__ import annotations

import argparse
import socket
import struct
import time

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Windows camera index")
    parser.add_argument("--host", default="127.0.0.1", help="WSL host (localhost forwarding by default)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--fps", type=float, default=30.0, help="maximum JPEG frames per second")
    parser.add_argument("--quality", type=int, default=80, help="JPEG quality from 1 to 100")
    parser.add_argument("--max-frames", type=int, help="stop after sending this many frames (for connection tests)")
    args = parser.parse_args()
    if (
        args.fps <= 0
        or not 1 <= args.quality <= 100
        or not 0 <= args.port <= 65535
        or (args.max_frames is not None and args.max_frames < 1)
    ):
        parser.error("fps must be positive, quality 1-100, port 0-65535, and max-frames positive")

    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Could not open Windows camera {args.camera}")

    frame_interval = 1.0 / args.fps
    frames_sent = 0
    print(f"Capturing Windows camera {args.camera}; sending JPEG frames to {args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            try:
                connection = socket.create_connection((args.host, args.port), timeout=2.0)
                connection.settimeout(5.0)
            except OSError as error:
                print(f"Waiting for WSL receiver ({error}); retrying in 1.5s")
                time.sleep(1.5)
                continue

            print("Connected to WSL receiver")
            try:
                with connection:
                    while True:
                        started = time.monotonic()
                        ok, frame = camera.read()
                        if not ok:
                            raise RuntimeError(f"Failed to read a frame from Windows camera {args.camera}")
                        encoded_ok, encoded = cv2.imencode(
                            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality]
                        )
                        if not encoded_ok:
                            raise RuntimeError("Failed to encode webcam frame as JPEG")
                        payload = encoded.tobytes()
                        connection.sendall(struct.pack("!I", len(payload)) + payload)
                        frames_sent += 1
                        if args.max_frames is not None and frames_sent >= args.max_frames:
                            print(f"Sent {frames_sent} frames")
                            return
                        delay = frame_interval - (time.monotonic() - started)
                        if delay > 0:
                            time.sleep(delay)
            except (ConnectionError, OSError) as error:
                print(f"WSL receiver disconnected ({error}); reconnecting")
    except KeyboardInterrupt:
        print("\nStopping camera sender")
    finally:
        camera.release()


if __name__ == "__main__":
    main()
