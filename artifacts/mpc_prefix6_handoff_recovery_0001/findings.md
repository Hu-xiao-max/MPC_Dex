# Prefix-6 handoff MPC recovery: strict turn after five corrections

**The bounded corrected native MPC completes a strict force-qualified full turn from the prefix-6 handoff after 5 actual corrections** (branch status `environment_boundary`, exit 0, 25.91 s measured loop, 25.54 s of it planning). The complete episode is 11 controls and 1.1 s: six physically replayed recorded controls, then five MPC controls. Final independent force-qualified rotation is 6.370676601 rad with 4 loaded fingers, 0.156097 rad tilt, no palm support, no external support and no drop; first physics qualification occurs at 1.068 s. This is assisted feedback recovery from one prescribed state, not a learned policy, SAC success or primitive-eligible demonstration.

## Actual execution

- Cold reset with physical seed 42 and re-execution of the six recorded controls. Observations, actions, rewards, flags, qpos/qvel, targets, full integration states and all 300 prefix physics samples matched the source `prefix_transitions.npz` (SHA `7b75bef1…`) bit-for-bit. Handoff is 0.6 s at 5.258153506 force-qualified rad, the same state in which the trained SAC0001 checkpoint dropped in all 262 attempts of the descending screen and the 256-episode warm-up.
- Planner: unchanged diagnostic recipe with the same measured 40 mm PCA prior (verified equal to the reference basis), eight workers, 350 candidates, ten elites, three iterations, six knots over .5 s, .1 s controls, retained progress, perpendicular weight 100, position weight 1000, principal weight 1, CEM seed 42, .98 search bound. Three of the 16 live target dimensions lay outside the search bounds at handoff; only the planning reference was projected, by at most 2.75e−7 rad, in three batches. Live physics, targets and accumulated rotation were never reset or clipped, and every planning call left the live integration state unchanged. Native build key `f5c108c9…`, binary SHA `1bb5515e…`.
- Every executed MPC action satisfied the effective bound .9799998998641968 and is bitwise unchanged by BC epsilon .02 clipping.
- A fresh cold-reset replay of all 11 controls, with a separately recomputed force audit, matched every trajectory field, all integration states and all 550 physics samples bit-for-bit. The frozen execution manifest verified before and after the loop.

## Correction trace

| Control | Origin | Reward | Force-qualified angle (rad) | Contacts | Tilt (rad) | Planning (s) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 7 | MPC 1 | −0.525 | 5.246891 | 3 | 0.142 | 4.26 |
| 8 | MPC 2 | −5.874 | 4.968172 | 4 | 0.103 | 4.92 |
| 9 | MPC 3 | 10.818 | 5.524086 | 4 | 0.127 | 5.88 |
| 10 | MPC 4 | 6.845 | 5.881334 | 2 | 0.116 | 5.54 |
| 11 | MPC 5 | 209.487 | 6.370677 | 4 | 0.156 | 4.94 |

The first two corrections regrasp and briefly lose retained progress (down to 4.968 rad) before the turn completes; control 10 momentarily holds with only two contacts. Maximum target change per control was 1.058 rad.

## Relation to the recorded continuation

The recorded controls 7–9 of the same source trajectory already complete the turn from this handoff at 0.9 s. The MPC corrections are a different solution: per-dimension maximum differences from the recorded control at the same index are 0.813, 1.190 and 1.266 (L2 1.005, 2.613, 2.470) for controls 7–9, and MPC needs two more controls. This run therefore adds an independent feedback-derived label set at the handoff and at four MPC-visited successor states; it does not show that a positive action exists where the trained policy acts, because the policy's own successor states were not visited here.

## Output

`actual_mpc_correction_pairs.npz` holds exactly the 5 executed MPC rows (60-dimensional observations, executed actions, actual next observations, rewards, flags, qpos/qvel, targets, integration states, control times) with exact row provenance in its metadata sidecar. The six prefix rows are excluded as labels and no successor action is fabricated. SHA `911561b9…`; trajectory SHA `072c8633…`.

## Scope

No policy inference, learning update, post-success hold, alternate prefix or parameter sweep occurred. One prescribed example is not a robustness estimate. Whether the SAC learner can use these five off-policy rows at prefix 6, alongside its own on-policy successors, is a separate protocol decision; until a reviewed stage states how they enter replay or supervision, they remain diagnostic evidence. Assisted records remain ineligible for primitives and add no main demonstration.

Inputs and protocol: `protocol.json`, `execution_manifest.json`, `input_provenance.json`; results: `summary.json`, `final_report.json`, `recovery/` (report, transitions, physics, branch state, independent audit), `actual_mpc_correction_pairs.*`; runtime: `runtime.json`, `run.log`; final hashes: `manifest.json`.
