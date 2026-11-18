# Prespecified 0.1 radian target-slew diagnostic

One 60-control MPC probe tested the existing `--max-target-delta .1` option. All compared settings matched `mpc_matched40_bound098_0001`: the same measured 40 mm prior and initial grasp, six-component basis, seed 42, native eight-worker evaluator, 350 candidates × three iterations, retained-progress scoring, tilt weight 100, normalized action bound 0.98, and nominal/hold reference candidates. The environment, actuator limits, and success criterion were unchanged. No sweep or training was run.

The probe completed 60 controls (6 simulated seconds) in 472.53 wall-clock seconds. It crossed full-turn control boundaries at controls 17 and 41. Independent replay reproduced every recorded observation, action, reward, next observation, qpos, qvel, and physics sample exactly, with maximum error zero. Holding the final recorded action for another 20 controls completed the requested two seconds without a drop or palm/table/other support.

| Measure | Bound 0.98 baseline | Added 0.1 radian slew |
|---|---:|---:|
| Final retained rotation, rad | 7.777342 | 14.938098 |
| Final force-qualified rotation, rad | 7.727621 | 14.871723 |
| Peak retained rotation, rad | 7.796234 | 15.225915 |
| Peak force-qualified rotation, rad | 7.746513 | 15.159540 |
| Largest executed target change, rad/control | 1.769880 | 0.100000078 |
| Peak object angular speed, rad/s | 92.118282 | 49.397558 |
| Peak joint speed, rad/s | 25.092758 | 14.046400 |
| Force-qualified rotation after two-second hold, rad | 7.637516 | 14.768606 |
| Loaded fingers after hold | 3 | 2 |
| Axis tilt after hold, rad | 0.351458 | 0.403757 |

Both holds passed the 30-degree axis gate with zero external support. The executed slew exceeded 0.1 radians by at most `7.772445695386487e-08` radians due to the existing float32 action conversion; the planner's target slew limit was 0.1 radians. This one comparison supports lower command and speed peaks with more rotation in this recording. It does not establish improved policy imitation or robustness. The post-hold grasp has two loaded fingers and greater tilt than the baseline.

This is a **historical probe using the evaluator before action canonicalization**: the planner scored raw float64 targets before execution converted them through the float32 action interface. Its archived predictions therefore do not carry the later evaluator's action-mapping fidelity guarantee. The recorded physical execution and independent float32-action replays remain valid. All numerical planning/physics sources matched the baseline at launch and stayed unchanged throughout this run. `source/` preserves the planner sources, configuration, and prior; `imported_dependencies/` also preserves the local modules imported at launch, including the CLI version that was subsequently edited independently.

The launch command and dependency checks are in `../mpc_matched40_slew01_0001.launch.py`. `audit_and_compare.py` produced:

- `independent_exact_replay/`: exact 60-control replay, physics and force audit.
- `independent_after_hold_2s/`: replay followed by 20 hold controls.
- `comparison.json`: numerical comparison and unchanged-setting checks against the baseline's existing independent audit.

Only after those audits passed, strict first-success preparation produced `../bootstrap_fullturn_seed_0003/replay.npz`: one 17-transition MPC diagnostic demonstration, independently force-qualified at `6.39237056055268` radians. It is not a SAC policy demonstration or production primitive.

Recorded transitions SHA-256: `ef0111dc8f985a6bf9fe48eeb821736a4aecf7a97ac48c93ad90ed2178a487f4`.
