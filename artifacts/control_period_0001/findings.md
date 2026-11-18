# Findings: control period (route b)

## Unscaled planner at a longer period makes no progress (2026-10-08, `diag_P0.1.log`, `diag_P0.3.log`)

With the archived CEM proposal spread (exploration floor 1.0 rad per knot on the residual, PCA-coefficient spread 1-2.8) and a planner step equal to the period, the 0.1 s reference finds 45-369 feasible candidates of 1,050 per plan and completes the turn in 13 controls (1.3 s), whereas at 0.3 s only 4-11 candidates per plan are feasible and the only one that survives is holding the current target: the action never changes (step 0.000 at 16 of 17 controls) and the object rotates 0.1 rad in 5 s. A 1 rad target change executed to convergence over a 0.3 s hold, or compounded over a 1.5 s horizon, releases the object in nearly every candidate; at 0.1 s the same target change is only partly tracked before the next command. The archived recipe's proposal spread is therefore tied to the 0.1 s period and must scale down with it. (At 0.5 s with the unscaled spread one attempt in nine reached the environment's success after 20 controls, 10 s, but failed the strict force audit.) Trace grid under way: floor scale 0.5 / 0.25 at 0.2 s, 0.33 / 0.15 at 0.3 s, 0.2 / 0.1 at 0.5 s.

## Review of the probe (2026-10-08, adversarial workflow, 17 confirmed findings, 0 refuted)

Settled in favour of the period change: the strict force-qualified success is accumulated per 2 ms physics step (observer, environment, evaluator and the native rollout all count contact loss in seconds and angle increments per step), so a success at a longer period is judged by the same rule and, because success and termination are sampled only at control boundaries, is at least as demanding as at 0.1 s. Fixed in the probe: the archived horizon is 100 controls (10 s), not 90 s, so the earlier scaling truncated 0.5 s attempts at 20 controls and reported the truncation as a boundary (the horizon is now kept as 100 control decisions and truncation is reported as such); the tolerance hold is now 2 s in every period (previously 20 controls = 20 P s); the perturbation ray is fixed per (period, control, direction) so tolerance is monotone in the radius, with r = 0 as the exact-replay check and radii down to 0.0003; the control budget is the archived 40 decisions (the per-control motion is period-independent because the planner's floors are per knot, so a turn needs a similar number of decisions at any period, and the earlier 12-24 budgets were binding); attempts are logged per seed; a probe-local copy of the native cache keeps the archived artifact byte-stable. Noted for the learning stage, not the audit: the per-control time penalty of the reward (-0.3) becomes -3.0 / -1.5 / -1.0 / -0.6 per simulated second at 0.1 / 0.2 / 0.3 / 0.5 s, and the observation's finite-difference velocities average over the period. The planner itself is the open problem: its proposal floors (1.0 rad per knot on the residual) are tied to the 0.1 s period (trace above).

## Proposal spread scaled with the period restores the planner (2026-10-08, trace grid `diag_P*_fs*.log`, seed 42, 24 controls)

| period | floor scale | feasible candidates per plan | outcome |
| --- | --- | --- | --- |
| 0.2 s | 0.5 | 38-139 | strict success, 14 controls (2.8 s) |
| 0.2 s | 0.25 | 111-271 | environment success at 10 controls, strict audit failed (force-qualified angle 5.3 rad at a raw 7.0) |
| 0.3 s | 0.33 | 34-215 | strict success, 16 controls (4.8 s) |
| 0.3 s | 0.15 | 254-483 | no success in 24 controls (7.2 s; 4.5 rad reached) |
| 0.5 s | 0.2 | 171-324 | environment success at 13 controls, strict audit failed (force-qualified 6.0 at a raw 8.7) |
| 0.5 s | 0.1 | 74-384 | environment success at 13 controls, strict audit failed (force-qualified 5.6 at a raw 6.3) |

A proposal spread of about 0.1 / P times the archived one (0.5 at 0.2 s, 0.33 at 0.3 s) gives the planner a feasible set of the archived size and strict successes in a similar number of control decisions; a smaller spread slows the turn, a larger one starves the feasible set (unscaled trace above). At 0.5 s both scales reach the environment's success while the strict audit fails because the object rotates part of the turn without force-bearing contact (raw angle well ahead of the force-qualified angle): half-second holds let the object spin free between contacts. Success-rate and open-loop-tolerance probes now run at (0.2 s, 0.5), (0.3 s, 0.33) and (0.5 s, 0.2) with the corrected probe.

## Corrected open-loop tolerance of the first successes (2026-10-08, `tolerance_P0100ms_seed42.json`, `tolerance_P0200ms_seed43.json`; fixed rays, 3 per control, 2 s hold, successes of 3)

| control j | 0.1 s (13 controls): r = .0003 / .001 / .003 / .01 | 0.2 s, unscaled proposal spread (17 controls): r = .0003 / .001 / .003 / .01 |
| --- | --- | --- |
| 0-4 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| 5 | 1 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| 6 | 2 / 1 / 1 / 1 | 0 / 0 / 0 / 0 |
| 7 | 3 / 2 / 2 / 1 | 0 / 0 / 0 / 0 |
| 8 | 3 / 3 / 2 / 3 | 0 / 0 / 0 / 0 |
| 9 | 3 / 3 / 3 / 2 | 1 / 0 / 0 / 0 |
| 10 | 3 / 3 / 3 / 2 | 2 / 0 / 0 / 0 |
| 11 | 3 / 3 / 3 / 3 | 3 / 1 / 0 / 0 |
| 12 | 3 / 3 / 3 / 3 | 2 / 3 / 2 / 1 |
| 13-16 | – | 2-3 / 2-3 / 3 / 2-3 |

The r = 0 column is 3/3 everywhere (exact replay). Doubling the period with the archived proposal spread does not widen the tolerance: the 0.2 s demonstration is intolerant to 0.0003 at its first nine controls, as the 0.1 s one is at its first five, and its tolerant tail is the same final second of the turn. The chaos is in the grasp dynamics of large per-control moves, not in the control rate alone. Whether the scaled proposal spread (smaller moves per control, 0.5 at 0.2 s and 0.33 at 0.3 s) changes this is what the running probes measure.

## Scaled spread at 0.2 s: the tolerant tail is a count of controls, not a time (2026-10-08, `probe_P0.2_fs0.5.log`, seed 42, 15 controls)

| control j | r = .0003 / .001 / .003 / .01 (successes of 3) |
| --- | --- |
| 0-8 | 0 / 0 / 0 / 0 |
| 9 | 1 / 0 / 1 / 0 |
| 10 | 1 / 0 / 0 / 0 |
| 11 | 3 / 3 / 3 / 2 |
| 12 | 3 / 3 / 2 / 1 |
| 13-14 | 3 / 3 / 3 / 3 |

With the proposal spread scaled to 0.5 the 0.2 s demonstration takes 15 controls (3 s) and is intolerant to 0.0003 at its first nine; the tolerant tail is again the last four to five control decisions, as at 0.1 s (last five of 13) and at 0.2 s unscaled (last five of 17). A perturbation needs about five control decisions to grow beyond the basin whatever the period, so the sensitivity is per control decision (per contact transition), not per second, and a longer period with the same number of decisions per turn does not widen it.

## Where the sensitivity lives: contracting dynamics between discrete bifurcation events (2026-10-08, `divergence_j0_r001.log`)

Observation distance from the unperturbed replay after a 0.001 perturbation of control 0 (three rays):

| demonstration | controls 1-7 after the perturbation | first control with distance > 1.0 |
| --- | --- | --- |
| 0.1 s, seed 42 (13 controls) | 0.02 → 1.3 (rays 0, 1); 0.007 → 0.09 → 0.57 (ray 2) | 2, 2, 5 |
| 0.2 s unscaled, seed 43 (17 controls) | 0.004 → 0.003 → 0.000 → 0.000 → 0.000 → 0.000 → 0.008 | 9, 9, 9 (0.88 at control 8, 2.4 at 9) |
| 0.2 s scaled 0.5, seed 42 (15 controls) | 0.01-0.04 → 0.003-0.07 → 0.05-0.2 → 0.1-0.3 | 9, 10, 9 |

At 0.2 s the perturbation is damped to zero (to three decimals) for six controls, i.e. the servo-driven dynamics contract between contact transitions, and the trajectory still diverges at one later control, where a residual below 0.001 flips the outcome: the divergence is a bifurcation at a discrete contact event (an edge of the hexagonal prism rolling over a fingertip), not a continuous exponential growth that a slower control rate could tame. This is why the tolerant tail is a count of decisions (the last few controls contain no further event) and why closed-loop MPC succeeds: it re-plans from the exact state at each event. A longer control period therefore cannot widen the open-loop tolerance; the lever is the geometry that creates the events (a round object has no edge transitions) or, failing that, a controller that re-plans through them.

## 0.3 s with scaled spread (2026-10-08, `probe_P0.3_fs0.33.log`, seed 42, 16 controls): worse than 0.1 s

| control j | r = .0003 / .001 / .003 / .01 (successes of 3) |
| --- | --- |
| 0-8 | 0 / 0 / 0 / 0 |
| 9-14 | 0-2 / 0-2 / 0-1 / 0-1 (sporadic) |
| 15 | 3 / 3 / 3 / 0 |

## Conclusion on the control period

A longer control period does not make the task learnable. With the proposal spread scaled to the period the archived MPC completes strict turns at 0.2 s and 0.3 s in 15-16 control decisions (0.5 s fails the strict audit in 3 of 3 attempts: the object spins part of the turn without force-bearing contact), but the open-loop tolerance of the resulting demonstrations is zero down to a perturbation of 0.0003 over the first 9 of 15 controls at 0.2 s and the first 9 of 16 at 0.3 s, against the first 5-7 of 13 at 0.1 s, and the tolerant tail is the last four to five control decisions at every period. The divergence measurement shows why: between contact events the servo dynamics contract a perturbation to zero, and at a later contact event (an edge of the hexagon rolling over a fingertip) a residual below 0.001 flips the outcome. The sensitivity is a sequence of discrete bifurcations tied to the geometry of the grasp, not a rate that a slower controller could damp; a longer period only passes more such events per control. The period probes were stopped after one success each at 0.2 and 0.3 s (the 0.1 s reference has two); the collector and the stage template keep their period options for the record. The object-shape diagnostic (`object_shape_0001`) tests the geometry directly.
