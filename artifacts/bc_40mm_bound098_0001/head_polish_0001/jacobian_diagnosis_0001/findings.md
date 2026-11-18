The frozen polished actor responds almost linearly to its early off-teacher observations, with joint-velocity deviations dominating the realized local response. The first-order model matches controls 2–7 closely; by control 8, nonlinear effects become material. This diagnosis uses recorded data and the original actor only: no fitting, optimizer updates, or physics runs occurred.

For each time-aligned control, let `delta = visited_observation - teacher_observation` and `J = d policy_action / d observation` at the teacher observation. The predicted action change is `J @ delta`. The actual state-induced change is `policy(visited) - policy(teacher)`. Adding the tiny teacher-state fitting error gives the predicted error against the original recorded teacher action. These are comparisons to a time-aligned recorded action, not claims that this action is optimal at the visited state.

All computations use the archived actor implementation, the original frozen polished checkpoint, and the same singleton CUDA float32 inference route with TF32 disabled. New teacher and visited predictions match the saved teacher predictions and actually executed actor actions bit for bit. Weights, checkpoint files, source files, input hashes and update counters remained unchanged.

| Control | Actual max action error against teacher | Predicted max error: fit + J delta | Relative L2 Taylor residual |
|---|---:|---:|---:|
| 1 | 1.49012e-07 | 1.49012e-07 | — |
| 2 | 9.77516e-06 | 9.6115e-06 | 1.564% |
| 3 | 2.23815e-05 | 2.23901e-05 | 0.485% |
| 4 | 1.97291e-05 | 1.97482e-05 | 0.546% |
| 5 | 0.000104904 | 0.000104977 | 0.201% |
| 6 | 0.00092262 | 0.000913084 | 0.964% |
| 7 | 0.000661999 | 0.000663096 | 0.128% |
| 8 | 0.0113508 | 0.0102164 | 30.567% |

The first observation is identical to the teacher observation, so its `J delta` term is zero: control 1 differs only by the 1.490116e-7 fitting error. For controls 2–7, relative L2 Taylor error ranges from 0.128% to 1.564%. Some early absolute residuals are only a few 1e-7 normalized-action units, where float32 inference rounding matters. At control 8, the local model has 30.567% relative L2 error and cosine alignment 0.95677. It captures 90.656% under the measure `1 - ||actual - linear||² / ||actual||²`; this is a zero-change-relative residual measure, not a causal variance decomposition.

By control 8, the observed principal-angle difference reaches 0.231225 radians and the joint-velocity difference reaches 0.274296 rad/s. A separate central finite-difference check over 1/64 of that displacement agrees with the autograd directional derivative to 0.02194% relative L2 error. The derivative is numerically consistent locally, while the full control-8 displacement leaves the most accurate linear neighborhood.

Group contributions are `J_group @ delta_group`, all expressed in common normalized-action units. The following shares project each contribution onto the total predicted action-change vector; they sum to 100%, may be negative or exceed 100% because of cancellation, and are not independent causal effects.

| Observation group at control 8 | Action contribution L2 norm | Signed share of predicted change |
|---|---:|---:|
| Joint positions (16) | 0.00148087 | 2.762% |
| Object position (3) | 0.000181817 | -0.190% |
| Object quaternion (4) | 0.00657358 | 11.299% |
| Current/previous principal angles (2) | 0.00400064 | 12.090% |
| Joint finite-difference velocities (16) | 0.0136 | 72.711% |
| Object finite-difference velocity (3) | 0.00284825 | 1.343% |
| Previous normalized action (16) | 2.05477e-05 | -0.016% |

Joint velocities supply 92.93% of the predicted-direction contribution pooled over controls 2–4, and 72.71% at control 8. At control 5 their share is 126.45% because other groups partly cancel it. Principal angles and quaternion components become notable later (12.09% and 11.30% at control 8). Direct previous-action input contributes little at that point; effects of earlier actions can already be represented in the changed physical-state and velocity inputs.

This supports investigating feedback-state coverage rather than only reducing error on the recorded observations. The actor maps realized off-teacher velocity and pose changes into action differences even where its teacher-state fit is near float32 precision. Corrective labels at those visited states could therefore address information absent from the original trajectory. The analysis does not identify the correct recovery actions, measure the simulator’s local dynamics Jacobian, or prove closed-loop stability or that any added dataset will succeed. All groups are correlated along one trajectory; raw input norms and Jacobian column magnitudes have different units and were not ranked.

`arrays.npz` preserves all eight 16×60 Jacobians, observations, policy/teacher actions, feature and group contributions, predicted and actual changes, and Taylor residuals. `report.json` contains per-control and per-group metrics, arithmetic checks, runtime details and all input hashes. `diagnose.py` reproduces the calculation; `manifest.json` hashes the generated artifacts. The archived actor remains a diagnostic BC actor with one prior head correction, not a SAC-trained policy or production primitive.

Checkpoint SHA-256: `a3a9473e6388a4a2a870240582d1ca048dbf4a961866d35f2d1c3aa0ecd13374`.
Arrays SHA-256: `bd33b4d0b63107cd2098d6305501fe34d325ca1a88e1e45556e0506368c40325`.
