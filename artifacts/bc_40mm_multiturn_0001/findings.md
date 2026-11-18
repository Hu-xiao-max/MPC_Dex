The bounded first-turn versus multi-turn BC comparison produced **zero independently qualified full turns in six prespecified evaluations**. All actors fail after two or three controls. Exact and 1e-6-clipped teacher replay both pass the strict full-turn audit before training, so this failure is specific to the learned approximations under the tested settings.

Both conditions use the same initialization (seed 17001), `SACAgent` 60/16 architecture with two 256-unit LayerNorm/tanh layers, two unused critics, epsilon 1e-6, Adam BC learning rate 3e-4, CUDA, one Torch CPU thread, and 6,000 mean-only updates. The only training comparison is 19 versus 60 actual recorded transitions. Each update uses the complete selected dataset. Evaluations at 1,000/3,000/6,000 updates use the strict 40 mm environment, success termination enabled, horizon 100, and the independent per-physics auditor. No critic, target critic, alpha, or online update occurs; hashes verify those components are unchanged.

| Training transitions | Updates | Training action MSE | Maximum action error | Peak force-qualified angle | Outcome |
|---|---:|---:|---:|---:|---|
| First 19 | 1,000 | .0319133 | 1.25386 | .174189 rad | Palm at .2 s |
| First 19 | 3,000 | 7.104e-5 | .0655672 | .184610 rad | Drop at .3 s |
| First 19 | 6,000 | 2.078e-5 | .0466768 | .184178 rad | Palm at .3 s |
| All 60 | 1,000 | .0121827 | 1.15542 | .192495 rad | Palm at .2 s |
| All 60 | 3,000 | .00813634 | 1.28626 | .171101 rad | Palm at .3 s |
| All 60 | 6,000 | .0277560 | 1.30217 | .176285 rad | Palm at .2 s |

Unlike the earlier short-trace experiments, teacher-state fitting itself remains inadequate. First-19 latent MSE decreases from .308736 to .0001320 to .00004095. All-60 latent MSE is .119852, .0423368, and .101263 at the same checkpoints, showing nonmonotonic fitting under the fixed learning rate. The all-60 actor's action MSE specifically on the first 19 teacher states is .0378272, .0251097, and .0860645. Thus the failure cannot be attributed solely to encountering states outside the demonstration.

The first-19 actor at 6,000 updates begins with a normalized action error .00131437 even at the exact initial teacher observation. After that control, maximum qpos and qvel errors are .0132982 and 3.56771. On its next two actual observations, action errors grow to .117786 and 1.30803. The all-60 actor at 6,000 updates already has .492715 maximum action error on the exact initial observation. All 60 recorded observations are distinct; no exact duplicate-observation contradiction was found.

The first-19 input contains 104 saturated scalar actions out of 304; the full input contains 476 out of 960. Both use the explicit diagnostic 1e-6 clipping setting. The paper's .02 setting was not retested because its direct teacher clipping already failed the preceding replay audit.

These results do not establish that broader teacher coverage cannot help: the larger condition did not fit its recorded targets accurately enough to isolate that question. They establish that this fixed training recipe has not produced a usable closed-loop clone within the authorized budget. No additional settings or updates were run. No actor qualified for the conditional two-second continuous-policy aftermath test, and no constant-command substitute was used.

`report.json` preserves complete fit/evaluation results, including training-set, first-19, and full-60 fit metrics at each checkpoint. `fit_diagnosis.json` records per-observation fitting errors and actual early rollout divergence. Each evaluation preserves actual observations/actions/next observations/rewards, qpos/qvel, initial joint frame, command records, and every 2 ms independent physics sample. Frozen code/assets, exact inputs, all six diagnostic checkpoints, and checksums are retained. This is a BC-only diagnostic, not a SAC-trained policy or production primitive; shared sources remain unchanged.
