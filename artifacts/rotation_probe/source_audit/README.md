# Source geometry and rotation-axis audit

**Finding:** the current uniform 120 mm long hexagonal rod is not a faithful
reconstruction of the visibly composite source object. The papers do not
provide an axial length or a transform identifying the rotation axis in
Menagerie's palm frame. This audit does not establish a defensible replacement
thickness or a specific 90-degree axis correction.

## Explicit source facts

- The requested paper's Section III-A specifies a hexagonal object with **45 mm
  across flats** and repeated rotation about its primary axis. Its Section III-D
  describes the other task as rotation about a perpendicular axis. Neither
  section gives an extrusion length, marker-arm dimensions, object CAD, or a
  Menagerie-relative coordinate frame. [Requested paper](https://arxiv.org/html/2609.14878v1#S3.SS1)
- The precursor planner's Section III-B1 uses a **40 mm across-flats hexagonal
  object**, controls rotation about its principal axis, and penalizes changes in
  position and the other orientation axes. It likewise does not specify an
  extrusion length or a local x/y/z axis in the robot model.
  [Planner paper](https://arxiv.org/html/2609.14868v1#S3.SS2.SSS1)
- Object pose comes from external OptiTrack motion capture in the requested
  paper. The official implementation table calls one observation-noise field
  "yaw," but this name alone does not establish its frame or its alignment to
  Menagerie's palm. [Requested paper, hardware](https://arxiv.org/html/2609.14878v1#S3.SS1)
  and [project implementation details](https://real-world-rl-with-mpc.github.io/)

## Direct visual evidence and bounded interpretation

1. `artifacts/sources/task_setup.png`, the requested paper's Figure 2, shows a
   **broader central white/green body and a substantially narrower green arm**
   bearing several spherical markers. The long narrow arm is not a uniform
   continuation of the central body's hexagonal cross section.
   [Figure 2](https://arxiv.org/html/2609.14878v1#S3.F2)
2. `rl_153s.jpg` is an unmodified frame extracted at 153 seconds from the
   official supplementary video. It shows the authors' MuJoCo twin alongside
   the physical object. Both visibly have a central faceted/circular body and a
   narrow marker-bearing arm projecting away from it. These perspective frames do not reliably distinguish an axial extension
   from an offset or radial attachment; no attachment-axis claim is made.
   [Official video](https://real-world-rl-with-mpc.github.io/static/videos/Real_world_RL_video_compressed.mp4#t=153)
3. `rl_137s.jpg`, from the geometry-adaptation section, shows the larger circular
   object and smaller hexagonal object. The broad contact bodies differ while
   both carry narrow extensions and spherical markers. This supports treating
   the contact body and marker-bearing extension as distinct geometry.
   [Official video, adaptation](https://real-world-rl-with-mpc.github.io/static/videos/Real_world_RL_video_compressed.mp4#t=137)
4. `planner_setup.png` and `grasp_9s.jpg` provide independent views of the green
   assembly from the predecessor project. The longer green bar bears the
   markers; the fingers grasp the broader central body. Its role as a tracking
   fixture is a visual inference consistent with the stated motion-capture
   setup, not an explicitly named part or supplied CAD specification.
   [Official planner setup](https://primitive-informed-mpc.github.io/static/images/experimental_setup.png)
   and [official grasping video](https://primitive-informed-mpc.github.io/static/videos/grasping_wPrimitive.mp4#t=9)

The justified rotation axis is the primary/symmetry axis of the **central
hexagonal contact body**. Its relationship to the marker-bearing arm is not
unambiguously established by the inspected perspective frames.
Uncalibrated perspective views do not reliably establish whether that axis is
Menagerie palm-local x, y, or z. The source object appears more compact than a
uniform 120 mm extrusion, but assigning a numerical thickness from these views
would be an additional assumption.

## Qualitative hand-axis alignment

In `rl_153s.jpg`, the authors' rendered central body appears to extend beneath
its palm along the wrist-to-fingertip direction, with the visible end face
near the fingertips. This is **more consistent with longitudinal hand alignment
than a prism extending across the row of fingers**. The dedicated planner
rotation clip (`planner_rotation_1_13s.jpg`, one frame per second) provides
additional hand views, but no calibrated axes. This is a visual inference,
not a recoverable exact rigid transform.

In the local Menagerie XML, the finger bases are spread along palm y
(approximately +0.0435, 0, -0.0435 m), and successive straight finger-link
offsets lie along local z. Thus **palm-local z is a plausible axis mapping**
for the apparent longitudinal source arrangement; palm-local y is the
cross-finger direction used by the current rod. The signs, initial angle, exact
alignment, and source-body thickness remain unverified. No assertion is made
that the source explicitly specifies a 90-degree axis correction.

This distinction can materially affect contact topology. In the existing
cross-finger prism, the index, middle, and ring contacts tend to occupy the
same cross-sectional side, leaving the thumb as the sole opposing contact.
A longitudinal arrangement could allow the outer fingers to contact opposite
sides of the cross section. That is a geometric hypothesis motivating a
separate reconstruction check, not evidence that a revised controller will
succeed.

## Consequence for the local reproduction

`mpc_dex.env.make_model` creates one uniform 45 mm AF hexagonal prism extruded
120 mm along object-local y and initializes that axis parallel to palm-local y.
No marker arm or separate central-body profile is modeled. This gives a
substantially different contact geometry and mass distribution than the source
assembly. Its bootstrap results establish behavior on that chosen rod, not on a
verified recreation of the paper object. The existing model should be labeled
an independent surrogate until the central-body dimensions and object-to-hand
axis transform are established. No active model, asset, training source, or
configuration was modified during this audit.

The `*_contact_sheet.jpg` files are frame grids for inspection; named timestamp
frames are direct full-frame video extracts. `source_manifest.json` identifies
the downloaded official resources and their hashes.
