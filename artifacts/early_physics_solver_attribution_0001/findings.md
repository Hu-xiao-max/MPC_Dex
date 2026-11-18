# Early physical branching: two unchanged three-control traces

The first large divergence follows a **different loaded contact manifold**, not exhaustion of the constraint solver's iteration budget. This is evidence about the current MuJoCo model and settings. It does not establish convergence under tighter settings or distinguish a physically appropriate feature transition from a collision-discretization sensitivity.

Exactly 300 new physics steps ran: the original nominal first three recorded float32 controls and the new 86-pair head policy's first three recorded float32 controls, each from the original cold reset. Every available source control field and all 150 saved physics samples per branch reproduce bit for bit. The head's saved initial/full control-boundary integration states also match; the older nominal source did not contain full integration snapshots. Both newly compiled model binaries are identical. No policy inference, learning, parameter change, extra forward solve, or additional physical branch ran. The only configuration edit redirects the grasp path to its byte-identical archived copy.

## Settings and measurements

The archived model uses MuJoCo 3.14.0, `implicitfast`, timestep `.002 s`, Newton, elliptic friction cones, `impratio=10`, solver iterations 50, tolerance `1e-8`, line-search iterations 50, line-search tolerance `.01`, and no noslip iterations. Native CCD and multiple-contact generation are enabled; CCD iterations are 35 and tolerance `1e-6`. Relevant finger/object contacts have dimension 4, zero margin/gap, `solref=[.02,1]`, and `solimp=[.9,.95,.001,.5,2]`. Their combined sliding friction is `.8`; finger and object geom defaults are `.8` and `.5`. Full options, collision geometry properties, compiled models, initial states, inputs, and library hash are archived.

The environment calls `mj_step` followed by `mj_forward` at every transition. The latter overwrites solver statistics. This study records both calls separately, including every solver iteration, contact point/frame/force, active constraint state, qacc, full integration state, and the stationarity residual `M*qacc - qfrc_smooth - qfrc_constraint`. The read-only hooks add no solver calls and preserve exact replay. `solver_fwdinv` is disabled; its zero values are **not** residual evidence. The separately recorded global-mass-trace scaled residual is also not the solver's island-specific stopping field.

## Sequence of divergence

| Event | Measured result |
| --- | --- |
| First step, `.002 s` | Maximum qpos difference `1.38e-8`; qvel difference `1.10e-5`. Both post-forward stationarity residual norms are below `9e-16`. Tiny different commands already produce tiny different states under a tightly solved contact system. |
| Control 2 ends, `.200 s` | qpos difference `3.52e-7`; qvel difference `7.95e-6`. |
| `.240/.242 s` | First constraint-state labels differ on two `ff_medial/object` rows. All affected forces are exactly zero in both traces. These are not evidence of a loaded slip transition. |
| Post-forward `.264 s`, step 132 | First contact-set and solver-iteration-count difference. Nominal: 11 contacts/44 constraints. Head: 12 contacts/48 constraints. qpos difference remains `5.24e-6`, qvel `2.67e-4`. |
| Next integration step, `.266 s` | qpos difference jumps to `.000305083`; qvel to `.152663794`. |
| Control 3 ends, `.300 s` | qpos difference `.001703813`; qvel `.069241196`, matching the original archived discrepancy. |

At `.264 s`, the `ff_proximal` **box** versus object **mesh** pair has one contact in the nominal trace and two in the head trace. The new second point is **27.60 mm** from the first and carries **5.9735 N** normal force. The nominal single point carries **7.0681 N**; the head's shared point carries **3.6665 N**. The contact depths are approximately `−1.676 mm` and `−1.632 mm`. The resulting generalized constraint-force vector difference has norm **.746992**, while the two stationarity residual norms are only **1.46e-8** and **7.59e-5**. These mixed generalized-force norms compare numerical magnitudes; they are not a bound on future trajectory error.

The loaded contact-set difference is generated before the force solve at that configuration and precedes the next large integrated-state difference. The earlier solver iteration counts match across traces. This supports a manifold-triggered branch; it does not prove that earlier finite solver/CCD errors could not shift which side of the manifold boundary is reached.

## Solver termination and limits of attribution

Across the 600 recorded solves, no constraint solve reaches 50 iterations; integration solves use at most 15 and post-forward solves at most 11. No MuJoCo warnings occur. At the branching post-forward solve, Newton uses 7 iterations for nominal and 8 for head. Final stored improvement/gradient values are respectively `1.202e-7 / 3.232e-8` and `.000467519 / .000168126`. Thus the final gradient need not be below `1e-8` even when the solver terminates early.

The pinned [MuJoCo 3.14.0 primal solver source](https://github.com/google-deepmind/mujoco/blob/3.14.0/src/engine/engine_solver.c) has several stopping paths, including Newton decrement and unsuccessful line search. The exact chosen path/decrement is not exposed in `mjData`; the final statistics cannot uniquely label it. Raising only the iteration cap would not address the observed early-exit paths. The [collision implementation](https://github.com/google-deepmind/mujoco/blob/3.14.0/src/engine/engine_collision_convex.c) obtains multiple witnesses from native CCD for this box/mesh pair. CCD's actual iteration count and termination status are not exposed here, so this study cannot certify CCD convergence.

A tightly bounded convergence comparison would be scientifically justified before claiming solver-independent contact sensitivity, especially for the collision manifold. It has **not** been run or selected here. The present evidence does not justify relaxing task criteria, replacing contact dynamics, or changing training physics automatically. Root has prioritized the separate assisted suffix-support evidence instead.

`analysis.json` contains all events, thresholds, actual solver statistics, and interpretation limits. `nominal/` and `head/` preserve complete diagnostics and exact replay arrays. `reference/` contains the pinned official source files used for interpretation; `frozen_inputs.json` and the final manifest bind the archived inputs and results.
