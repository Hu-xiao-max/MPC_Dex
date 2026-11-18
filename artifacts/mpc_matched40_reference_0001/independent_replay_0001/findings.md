The 60-control, 6-second diagnostic MPC trajectory independently reproduces **three force-qualified full turns without palm contact or dropping**. Its entire recorded execution is bit-exact: all observations, actions, rewards, next observations, qpos, qvel, and all 3,000 × 40 per-physics measurements have maximum replay difference zero. The copied environment/assets/grasp contract matches the original report.

| Replay | Executed duration | Peak force-qualified angle | Final force-qualified angle | Outcome |
|---|---:|---:|---:|---|
| Original 60 controls | 6.0 s | 23.225873 rad | 23.121914 rad | Three qualified turns; no palm/drop |
| Original plus 20 held targets | 8.0 s | 23.225873 rad | 23.145990 rad | Full two-second hold passes; four loaded fingers |
| Teacher clipped by .02 | 1.9 s | 3.394268 rad | 2.833417 rad | Palm failure; no qualified full turn |
| Teacher clipped by 1e-6 | 2.9 s | 8.305565 rad | 8.305565 rad | First full turn qualifies, followed by drop |

The exact trajectory first crosses one, two, and three force-qualified turns at **1.852, 3.250, and 4.806 seconds**. Their first qualified 10 Hz control boundaries are 1.9, 3.3, and 4.9 seconds. At 6.0 seconds it retains three loaded fingers, .106929 rad axis tilt, and zero palm/table/other applied support. After the separate hold through 8.0 seconds, it retains four loaded fingers with .053827 rad tilt and zero external support. The force-balance residual never exceeds 3.01e-14 N.

**476/960 scalar teacher actions are exactly saturated (49.5833%).** The first 19 controls contain 104/304 saturated values. The .02 clipping trial therefore still changes many commands substantially and fails without reaching a full turn; its peak palm normal force is 7.538 N.

The 1e-6 clipping trial is materially better than the earlier short demonstrations. It first reaches a strict force-qualified full turn at 1.852 seconds with three loaded fingers, .129075 rad tilt, raw angle 6.564833 rad, and force-qualified angle 6.286826 rad. At the 1.9-second control boundary its raw/force-qualified angles are **7.026046/6.748039 rad**, with three loaded fingers, .160754 rad tilt, and no prior palm, table, workspace, contact-loss, or external-force violation. It later loses the object and terminates at 2.9 seconds. This is a valid first-turn prefix, not a successful full 60-control clipped episode.

Continuous configuration suppresses success termination, so the environment's success field is not used as the event detector. `full_turn_events` independently checks signed raw and force-qualified rotation, current force-bearing retention, and absence of all prior failures/support violations at every 2 ms sample. The original and held episodes also pass the independent final finger-supported success criterion; the clipped episodes do not pass that final criterion.

This experiment performs **no BC or SAC training**. The original controller uses the measured 40 mm PCA prior and optional reference candidates and remains diagnostic, not a SAC-derived prior or production primitive. The results establish successful physical replay and improved tolerance of tiny action clipping for the first turn; they do not establish closed-loop neural imitation or robustness of the remaining multi-turn sequence.

`report.json` contains all four independent audits, clipping counts, and full-turn events. Each replay directory preserves actual executed observations/actions/next observations/rewards/qpos/qvel, termination flags, initial joint frame, command records, and per-physics traces. `source/` contains the frozen numerical code and assets; `manifest.json` binds the inputs and results by SHA-256. Shared sources were not edited.
