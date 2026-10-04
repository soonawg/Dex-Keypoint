# LEAP Hand model assets

The URDF and STL meshes in this directory are from
[`leap-hand/LEAP_Hand_Sim`](https://github.com/leap-hand/LEAP_Hand_Sim), under
the included MIT license. `robot.urdf` is the source model;
`leap_hand.xml` is a kinematic MuJoCo conversion of its joint/link tree and
visual mesh origins.

Rebuild the MuJoCo XML from the project root with:

```bash
./.venv/bin/python scripts/convert_leap_urdf.py
```

The conversion supports finite revolute joints and mesh visuals, as used by
this source model. The profile `configs/leap_hand.json` lists joints in
thumb/index/middle/ring order to match the feature retargeter's 16-target
layout. This conversion is intended for visualization and retargeting tests;
it has not been validated for physical hardware control.
