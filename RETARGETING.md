# Project 14: Dex-Retargeting

A small, configurable library for mapping human hand landmarks to robot
fingertip targets and solving bounded robot joint positions with kinematic
least squares. It does not access a camera or send commands to hardware.

## Current implementation

- Normalize MediaPipe's 21 landmarks into a wrist/palm coordinate frame.
- Scale and rotate the four supported human fingertip targets into the chosen
  robot model's frame.
- Load a MuJoCo hand model and validate configured joint limits and fingertip
  body names.
- Minimize fingertip position error with bounded least squares and a temporal
  penalty to discourage frame-to-frame jumps.
- Select robot model, joint order, fingertip bodies, scale, and coordinate
  transform through a JSON profile.
- Run the same retargeting path from a webcam or recorded video, with optional
  MuJoCo visualization; tracking loss holds the last valid joint solution.

The `allegro_right.json` profile reuses Project 39's Allegro assets. The
`leap_hand.json` profile selects the LEAP Hand model converted from the
official URDF and uses its own joint order and limits. Choose a profile with
`--config configs/leap_hand.json` or `--config configs/allegro_right.json`.
Additional hands can be added with a MuJoCo-compatible model and a profile
listing joint names, fingertip bodies, palm scale, and source-to-robot
rotation.

The `shadow_hand.json` profile includes all 24 Shadow Hand joints: thumb,
index, middle, ring, little finger, and two wrist axes. Wrist tilt is measured
relative to the first detected pose; press `c` in the preview to recalibrate.

## Install and test

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
dex-retarget --camera 0
dex-retarget --video demo.mp4
dex-retarget --tcp-camera
```

When running from the project root, the existing Project 39 virtual
environment can be used without an editable install:

```bash
./.venv/bin/python -m dex_retargeting.teleop --camera 0
./.venv/bin/python -m dex_retargeting.teleop --tcp-camera
```

For `--tcp-camera`, start `windows_camera_sender/run_camera_sender.ps1` in
Windows PowerShell after the WSL command reports that it is waiting for the
sender. Pass `--no-viewer` to process landmarks without opening MuJoCo.
The frame receiver is unauthenticated; keep it on localhost or a trusted
network and do not expose it publicly.
Use `--config` to choose another robot profile, and `--handedness any` if the
input hand is not reported as Right by MediaPipe.

The default `features` method reuses Project 39's responsive finger-flexion
mapping. `--method ik` selects the experimental 3D fingertip optimizer. The
preview is simulation-only and does not send hardware commands. The Allegro
profile's IK coordinate transform is an initial identity mapping and may need
calibration for a particular camera orientation and hand pose. Check the
displayed fingertip RMSE and joint limits before treating IK as validated.
