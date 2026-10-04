# Shadow Hand E3M5 model assets

The MJCF and meshes are from the Google DeepMind MuJoCo Menagerie
[`shadow_hand`](https://github.com/google-deepmind/mujoco_menagerie/tree/main/shadow_hand)
model and are provided under the included Apache-2.0 license.

`configs/shadow_hand.json` maps all 24 hand joints:

- Thumb: opposition plus four flexion joints.
- Index, middle, and ring: knuckle/base plus three flexion joints each.
- Little finger: metacarpal, knuckle, and three flexion joints.
- Wrist: two axes driven from relative palm tilt.

The two wrist axes follow palm tilt relative to the first detected hand pose.
Press `c` in the preview to recalibrate the neutral wrist orientation. The
mapping is a simulation visualization baseline, not validated hardware
control.
