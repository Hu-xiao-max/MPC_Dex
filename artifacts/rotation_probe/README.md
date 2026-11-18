# Unsupported axial rotation feasibility probe

This is an independent geometric feasibility experiment under the unchanged
`mpc_dex.env` contact model and `assets/initial_grasp.npz`, not the paper's
controller, a trained policy, or a full-turn result. Only 16 joint position
targets are applied, held at 10 Hz. The free object receives no applied force,
torque, joint support, or scripted displacement. Fingertip IK runs in a separate
scratch state; the execution state evolves through real MuJoCo dynamics.

## Strictly verified retained progress

`orbit_0.8_s4_p1_f0.npz` advances fingertip orbit targets by 0.8 radians over
4 seconds, followed by a 3-second hold. Measured axial object twist ends at
**0.7432750242 radians (42.59 degrees)**. Independent replay audits all 3,500
physics transitions, finding **at least 3 contacting fingers at every 2 ms
sample**, at most **0.02421745 radians (1.39 degrees) axis tilt**, no drop, no
palm/table contact, no applied external or generalized force, and exact angle
agreement with the original rollout. See the matching `_replay.json`.

The NPZ stores 70 60-dimensional observations and next observations, normalized
16-dimensional actions actually executed, paper rewards, physical qpos/qvel,
object position, measured angle, contacts, tilt, and IK error. It is a potential
partial-motion bootstrap demonstration, not evidence for continuous rotation.

Reproduce:

```sh
.venv/bin/python scripts/probe_rotation.py --target .8
.venv/bin/python scripts/probe_rotation.py --verify artifacts/rotation_probe/orbit_0.8_s4_p1_f0.npz
```

## Slower demonstration compatible with a 0.02 radian target step limit

`orbit_0.8_s8_p1_f0.npz` advances the same 0.8-radian tip orbit over 8 seconds,
then holds for 3 seconds. It retains **0.7210566465 radians (41.31 degrees)**,
with **all four fingers contacting at every 2 ms physics step** and at most
**0.01605463 radians (0.92 degrees) tilt**. Maximum per-joint target change is
**0.01850811 radians per 100 ms**. The NPZ contains 110 transitions with the
same observations/actions/physical-state fields and its independent replay
matches exactly. This is the preferred seed when a 0.02-radian target rate
limit applies. Generate with `--target .8 --seconds 8`.

## Larger motion and limits

Following 80% of the observed center drift and reducing fingertip orbit radius
to 97% retains larger angles: `orbit_1.8_s4_p0.97_f0.8.npz` ends at
**1.718680244 radians (98.47 degrees)** with four final contacts and 0.02528
radians final tilt. It does **not** satisfy the stronger continuous two-finger
criterion: high-resolution replay finds brief single-finger intervals. No
replay has shown a full revolution. A more conservative 1.4 target retains
1.04253 radians with small tilt but likewise includes a single-finger interval.

At baseline orbit phase 0.8, thumb joint `thj0` reaches its 1.396-radian upper
bound. Larger phases increasingly require millimeters to centimeters of IK
error. The thumb is the sole opposing contact in the original grasp, while the
other three tips approach the same side. Straightforward one-finger resets by
one hexagon face angle (pi/3), with outward lift 0–12 mm, fail while resetting
the thumb or the following finger; high attained twist on these dropped trials
is not retained rotation. Orders tested: thumb-index-middle-ring,
index-thumb-middle-ring, index-middle-ring-thumb.

These observations suggest that a bootstrap controller can learn meaningful
initial motion from the strict 0.743-radian demonstration, while full-turn
finger gait requires a different grasp/contact arrangement or a contact-aware
transfer that creates opposing support before the thumb resets. Candidate
solutions should audit physics-step contacts and orientation, since 10 Hz
samples hide simultaneous release of the three adjacent fingers.

## Bounded contact-transfer follow-up: three alternatives

Three additional alternatives preserve the original physics and start at the
same saved grasp. No direct object state changes occur during execution.

1. **Thumb posture reset at a fixed tip site.** After the slow 0.8-radian orbit,
   lower the thumb base target toward 0.95 rad while preserving its commanded
   tip-site position by IK. The object drops during the reset despite tip-site
   IK error below 0.1 micrometers. The tip site is a point on a rotating rigid
   distal link; preserving it does not preserve the surface contact point or
   fingertip orientation. Null-space posture motion changes the loaded contact.
2. **Transfer opposition to the curled index finger.** Move the index tip from
   orbit phase 0.8 to 2.8 radians with 4 mm radial clearance, then reset the thumb
   and continue the orbit. The object stays in the hand for the 29-second trial,
   but final axial progress is only 0.312 rad and axis tilt grows to 84.3 degrees.
   This establishes reachable alternate contact arrangements, but fails the
   intended axial-rotation task. Loading the off-center index introduces a
   large transverse moment; a centered or symmetric transfer merits separate
   investigation, with care about thumb/finger collision.
3. **Delay thumb reset to a more wrapped grasp.** Advance to phase 1.4 slowly,
   then reset the thumb without outward lift and reset the other fingers.
   Four contacts remain at the end of the thumb-reset stage, but object twist
   reverses from 1.084 to -0.024 rad. It eventually drops during ring reset.
   Contact retention alone does not make a positive-rotation gait.

Reproduce using `--transfer-alternative 1`, `2`, or `3`; saved arrays and phase
summaries are `transfer_alternative_N.npz` and `.json`.

`--contact-audit` reproduces `contact_wrench_audit.json`. This linear-program
check asks whether the instantaneous contacts can balance the gravity wrench.
It uses a circumscribed friction polygon, independently allows full torsional
friction, and omits actuator limits. This makes equilibrium easier than the
physical model: infeasibility rules out equilibrium for those contacts, whereas
feasibility does not prove the hand can execute or stabilize it. Removing thumb
contacts makes equilibrium infeasible at the initial, 0.721-rad, and 1.043-rad
states. A more deeply wrapped 1.719-rad state admits equilibrium without thumb
in this relaxed check. Thus a transfer needs to establish a different opposing
contact arrangement before unloading the thumb; current tip-only trajectories
do not maintain the necessary contact wrench through the transfer.

At 1.043 radians, representative finger tip force directions in palm coordinates
are approximately (-0.978, 0.011, -0.207), while medial-link contacts still push
mostly along -z; the thumb distal contact pushes (0.852, -0.018, 0.523). The
additional phalange contacts explain why solving only fingertip positions does
not control all loaded contacts or guarantee force closure.

## Fixed-thumb rolling: three configurations

A further bounded experiment holds the thumb joint targets fixed after an
optional 1.5 mm inward squeeze. The other fingertips follow a bounded orbit,
then reset individually with radial clearance. Actual joint target changes are
limited to 0.02 radians per 100 ms; three cycles are requested. These are
additional policies tested in the original unmodified free-object dynamics.

| Configuration | Drive | Reset lift | Measured behavior |
| --- | --- | --- | --- |
| 1 | All three fingers, 0.6 rad stroke, no extra thumb squeeze | 6 mm | First drive ends at 0.502 rad and four contacts. Index reset ends at 0.668 rad and four contacts. Drops during the beginning of middle reset. |
| 2 | All three fingers, 0.35 rad stroke, 1.5 mm thumb squeeze | 4 mm | First drive ends at 0.292 rad and four contacts. Index reset ends at 0.460 rad and four contacts. Drops as middle reset begins. |
| 3 | Index and ring drive 0.6 rad; middle and thumb hold, 1.5 mm thumb squeeze | 4 mm | Drops during the first drive, before any reset. |

**No complete repeatable positive cycle was obtained.** The reported terminal
angles of dropped trajectories (0.945, 0.940, and 1.788 rad) include motion after
loss and must not be counted as retained rotation. In configurations 1 and 2,
center drift reaches roughly +13 to +17 mm along palm x before the last stable
sample. Thumb normal contact force is then 6–8 N (and reaches 10–12 N earlier),
followed by an abrupt slip and loss of all contacts. This experiment fixes joint
targets, not the physical thumb pose under servo compliance. It also allows the
opposing finger targets to follow 80% of object center drift; that choice may
permit migration of the object out of the static thumb's reachable contact
region. Results therefore reject these schedules, not all fixed-thumb gaits.

Run with `--fixed-thumb 1`, `2`, or `3`. `fixed_thumb_N.npz` contains executed
commands, physical state, observations, and phase labels. Matching JSON files
summarize outcomes; `_contacts.json` files report contact force and center drift
near each failure.
