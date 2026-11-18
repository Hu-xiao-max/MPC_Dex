The prespecified slew-limited teacher robustness gate **fails: 0/3 noise cases complete a qualified turn**. As required, this branch stops without BC, head refinement, SAC, or further sensitivity settings.

The source generation was not used while running. `planner_constraints` first confirmed completed 60-control generation plus independent exact/hold audits and supplied immutable hashes. The source transition hash is `ef0111dc8f985a6bf9fe48eeb821736a4aecf7a97ac48c93ad90ed2178a487f4`. This artifact then copied those inputs and archived numerical dependencies. Later live CLI and rollout canonicalization changes are unused.

Independent baseline verification passes:

- All original observations, actions, rewards, next observations, qpos, qvel, and every physics sample replay with maximum difference zero.
- The strict first-success prefix has 17 controls, ending at **6.392371 force-qualified radians** with four loaded fingers and no palm/drop.
- The full 60 controls end at **14.871723 force-qualified radians**.
- Both two-second fixed-target holds pass: after all 60 controls, final **14.768606 rad**; after the strict prefix, final **6.419906 rad**.
- The actual Torch epsilon=.02 action clamp leaves every original recorded action bitwise unchanged.

The three noise cases exactly match the preceding sensitivity protocol: independent scalar uniform noise in [−1e-4,+1e-4], NumPy default_rng seeds 17011/17012/17013, float64 addition followed by float32 conversion. No extra action clipping, slew projection, state perturbation, or object force is applied. All 60 recorded commands are available, so a perturbed run may use later recorded commands until strict success or failure.

| Noise seed | Executed controls | Peak force-qualified angle | Final force-qualified angle | Result |
|---|---:|---:|---:|---|
| 17011 | 7 | 3.873991 rad | 3.873991 rad | Drop at .7 s |
| 17012 | 7 | 3.806762 rad | 3.806762 rad | Drop at .7 s |
| 17013 | 53 | 4.373697 rad | 2.917494 rad | Drop at 5.3 s |

The planner-side comparison confirms the sole controlled planning change was a .1 rad/control target slew limit, aside from float32 conversion excess of 7.77e-8 rad. Maximum object angular speed decreases from the preceding bound-.98 teacher's 92.1183 to 49.3976 rad/s. These improvements and the successful exact trajectory do not establish tolerance to the tested command perturbations. The noise cases are a small prespecified gate, not a population robustness estimate.

`audit_report.json` contains independent exact/strict/hold evidence. `noise_gate_report.json` contains the completed three-case gate; each case preserves its supplied controls, actual executed transitions, commands, and every 2 ms physics sample. The source completion evidence, archived generation imports, initial grasp, inputs, and hashes are retained. The prepared training/refinement scripts were never executed; no `bc_paper` output or refined checkpoint exists. This remains diagnostic research, not a learned SAC policy or production primitive.
