# Five prescribed noisy-state MPC recoveries

**All five prescribed states reach a strict force-qualified full turn after 1–7 actual MPC corrections.** These are fixed-mean noisy prefixes followed by MPC feedback, not learned-policy or SAC successes.

| Source seed | Prefix force-qualified angle | MPC corrections | Total controls / seconds | Final force-qualified angle | Loaded fingers | Final tilt |
|---:|---:|---:|---:|---:|---:|---:|
| 17021 | 3.683945 rad | 7 | 16 / 1.6 | 6.636885 rad | 3 | 0.129022 rad |
| 17022 | 5.646855 rad | 1 | 10 / 1.0 | 6.429024 rad | 3 | 0.088191 rad |
| 17023 | 4.559690 rad | 6 | 15 / 1.5 | 6.472274 rad | 4 | 0.193668 rad |
| 17024 | 5.413047 rad | 3 | 12 / 1.2 | 6.480114 rad | 3 | 0.074671 rad |
| 17025 | 4.524553 rad | 4 | 13 / 1.3 | 6.620339 rad | 3 | 0.117556 rad |

Every branch begins with a cold-reset replay of its nine previously recorded controls. Actions, observations, rewards, terminal flags, qpos/qvel, actuator targets, complete integration states and all 450 prefix physics rows match the original source bit-for-bit. No noise is regenerated. The original latent means, standardized Gaussian draws, actual sampled commands, RNG snapshots and reports are copied into inputs/. The prefixes are the earlier fixed-mean experiment, not the failing closed-loop neural trajectories.

The planner is the previously approved corrected native recipe: eight workers, 350 candidates, ten elites, three iterations, six knots over .5 seconds, .1-second controls, retained progress, perpendicular weight 100, position weight 1000, principal weight 1, and the same six-component measured 40 mm prior. Each branch starts CEM with seed 42 and at most twenty correction controls. The .98 normalized-action search limit has the same float32-safe effective bound .9799998998641968. Warm nominal and current-target reference candidates are retained. Where the true live target lies outside search bounds, only the reference target is explicitly projected; live physics, accumulated rotation, velocities and targets are never reset or clipped at the switch. Every planning call leaves live integration state unchanged. All executed correction commands remain inside the bounds and are bitwise unchanged by BC epsilon .02 clipping.

Each branch stopped at its first environment success and independently passed the stronger force-supported criterion. No branch drops or uses palm/table/other/applied support. A separate fresh physical replay of all 66 executed controls and 3,300 physics transitions matches bit-for-bit. The task used unchanged gravity/contact physics and the strict 40 mm horizon-100 environment. No alternative prefix, planner parameter, hold branch, BC update or SAC update was tried.

`actual_mpc_correction_pairs.npz` contains exactly 21 actually executed MPC rows; all 45 noisy prefix rows are excluded as expert labels. The sidecar maps every row to its original branch, control and hash-bound full transition archive. Observations, actions, actual next observations, rewards, terminal flags and physical states are preserved. No successor actions are invented. These are supervised diagnostic pairs from mid-episode states, not a new cold-reset replay dataset or main-stage demonstration export.

This establishes that the fixed feedback planner can recover a turn from each of these five specifically tested partial states. It supplies actual corrective coverage that the original nominal BC dataset lacked. It does not establish that an actor can fit or use that coverage, that stochastic SAC will succeed, or that the recoveries generalize to other states. No post-success hold or continued-policy test was performed; sustained retention is unmeasured for these branches. Five prescribed examples are not a population robustness estimate. The diagnostic prior remains MPC-derived, so none of this advances the SAC-prerequisite or twenty main-demonstration counts.

One startup attempt failed on a missing archived helper import before constructing an environment. The error log is retained; the byte-identical helper was added, inputs were refrozen, and the same unchanged protocol and execution script then completed with exit 0. All copied and original immutable inputs pass the final hash check. Source snapshots, native build provenance, command, runtime, complete failed-start record, branch states, controls, independent audits and row identities remain in this artifact.
