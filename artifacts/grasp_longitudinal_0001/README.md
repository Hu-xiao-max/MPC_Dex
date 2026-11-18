# Longitudinal grasp exploration

This is an exploratory interpretation of source-video alignment, not author calibration. The prototype places object-local y along palm-local z using a relative +90° x rotation. Shared simulator sources, model assets, the original 45 mm grasp, and the verified transverse 40 mm grasp were not modified.

The bounded search attempted 20 initialization geometries: ten each for 40 mm and 45 mm AF rods, keeping the existing 120 mm length. None passed initial geometry checks, so no candidate advanced to dynamic settling or cold verification and no grasp/configuration was exported. The attempted palm-local centers had z=60–80 mm. Joint-target IK errors were 19.8–33.8 mm, and object/finger intersections reached 28.4–36.3 mm. These failures do not establish that longitudinal grasping is impossible: the middle-finger targets were beyond its reach and the uniform long prism intersects multiple finger links.

Exact attempted targets and initial contacts are in [search_report.json](search_report.json). The artifact-local [driver](synthesize.py) refuses to repeat the search after the report exists.

An independent static kinematics review identified the following reachable target sets. These are suggestions for a future bounded pass; they received no physical grasp or contact verification in this task. Coordinates are in meters in the palm frame. Finger order is first, middle, ring, thumb; angles lie in the palm x/y cross-section, and z values are absolute coordinates.

| Set | Center | Site radius | Angles, degrees | Site z values |
| --- | --- | --- | --- | --- |
| A, 40 mm | `[.065, 0, .030]` | `.033` | `[70, 0, -70, 180]` | `[.055, .050, .040, .005]` |
| B, 45 mm | `[.062, 0, .030]` | `.035` | `[70, 0, -70, 175]` | `[.050, .045, .035, 0]` |
| C, thumb joint margin | `[.067, .004, .030]` | `.034` | `[85, 0, -70, 160]` | `[.050, .045, .035, 0]` |

Set A places the thumb base near its upper joint limit. The review's joint seed for C is:

```text
[-.14337, .37803, 1.22714, .82672,
  .03958, .97084, .52641, .36228,
  .23320, .64921, 1.15055, .73901,
 1.24122, .26302, 1.29269, 1.02657]
```

Middle-finger z at or below 50 mm improves reachability. Radial targets should account for the hexagon's different distances to vertices and faces, plus the fingertip surface radius. Even a successful future bootstrap grasp requires a separate MPC cost-frame review before being used by the main pipeline.
