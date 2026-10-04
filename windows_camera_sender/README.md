# Windows Camera Sender

This folder is the only part of the project needed on Windows. It captures
the webcam and sends JPEG frames to the WSL2 receiver; it does not run
MediaPipe or retarget the hand.

Copy these files to a Windows directory, then in PowerShell run:

```powershell
.\setup_camera_sender.ps1
.\run_camera_sender.ps1
```

Start the receiver first in WSL2. Add `--mujoco` to use the software hand
simulation:

```bash
python -m dex_teleop --tcp-camera --mujoco
```

The default connection is `127.0.0.1:8765`, using WSL localhost forwarding.
Use `.\run_camera_sender.ps1 -Camera 1` for another Windows camera index.
Press Ctrl+C to stop the sender. Keep the camera attached to Windows; do not
attach it to WSL with `usbipd`.

To send a short test stream, run
`.\run_camera_sender.ps1 -Port 8765` after starting the WSL receiver; for a
finite smoke test, use
`.venv\Scripts\python.exe .\camera_sender.py --max-frames 5`.

The TCP stream has no authentication or encryption. Keep it on localhost or a
trusted network and do not expose the receiver port publicly.
