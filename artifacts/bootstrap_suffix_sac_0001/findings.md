# Fixed prefix-7 SAC results

The bounded run completed normally at 2,000 online policy interactions and 2,000 SAC updates (exit 0, 298.90 s measured loop). Its final frozen checkpoint completes the turn after seven physically replayed controls in one deterministic and all five prescribed stochastic evaluations. It also passes all six prefix-8 retention checks. All six final cold-reset evaluations fail at the first control. This is learned assisted suffix control, not a successful cold-start SAC policy or a primitive dataset.

## Actual execution

- Initial checkpoint: `66681b0f9651b21edb39ef94cb9827e8dcb1ab08249ac7e82010f81c114cf937`; inherited BC 12,000, critic pretraining 0, online SAC 0.
- Warm-up: 256 complete frozen-actor episodes, 808 actual policy suffix transitions; 165 independently force-qualified successes, 47 drop-labeled and 44 palm-labeled failures.
- Critic pretraining: exactly 500 actual SoftSARSA updates with the actor and temperature unchanged.
- Online: exactly 2,000 policy interactions and updates; 753 completed assisted episodes, 532 qualified successes, 116 drop-labeled and 105 palm-labeled failures. These episodes use changing policies and are not frozen-policy evaluations.
- Last episode remains open after its first suffix action (eight total controls), with neither termination nor truncation. Replay slot 2,807 retains its pending successor; no terminal or next action is invented.
- Replay contains exactly 2,808 actual suffix rows. The 7,070 physically executed prefix controls and all evaluation transitions are excluded. Raw rewards remain unscaled in replay; update batches alone divide rewards by 200.
- Final counters are BC 12,000 / critic pretraining 500 / online SAC 2,000. No new BC, head solve, MPC query, stage advance or trajectory export occurred.

The original 60-input / 16-absolute-action interface, 40 mm scene, physics, reward, force audit, no-palm gate and horizon of 100 total controls remain unchanged. Every recorded prefix is physically replayed from reset and checked bit-for-bit against the source, including integration states and 2 ms physics samples. The [frozen protocol](protocol.json), [execution manifest](execution_manifest.json) and [summary](summary.json) preserve the settings and counts. Reward and entropy scales changed together with the reset curriculum; this is not an isolated causal comparison of either choice.

## Frozen evaluations

The five prefix-7 stochastic seeds are 17101–17105; prefix-8 seeds are 17121–17125; final cold seeds are 17111–17115. Deterministic seed is 42. Evaluations preserve training RNG and learner state and never enter replay. These fixed samples establish their recorded outcomes, not a general reliability estimate.

| Checkpoint | Prefix-7 deterministic | Prefix-7 stochastic | Prefix-8 deterministic / stochastic | Cold deterministic | Cold stochastic |
| --- | --- | --- | --- | --- | --- |
| Initial actor, before critic pretraining | Pass | 4/5 | Pass / 5/5 | Pass (BC-origin) | Not scheduled |
| Online 500 | Fail | 1/5 | Not scheduled | Fail, control 2 | Not scheduled |
| Online 1,000 | Fail | 1/5 | Not scheduled | Fail, control 1 | Not scheduled |
| Online 2,000 | Pass | 5/5 | Pass / 5/5 | Fail, control 1 | 0/5; all control 1 |

All final prefix-7 successes take two policy suffix actions and terminate at nine total controls. Final stochastic force-qualified angles span 6.557732–6.958011 rad; deterministic angle is 6.660177 rad. Prefix-8 successes take one policy action. Cold evaluation failures remain decisive for deployment eligibility.

The early regression later reverses. Success counts in consecutive 100-completed-online-episode bins are 56, 11, 31, 85, 97, 100 and 100; the final partial bin is 52/53. The last unfinished episode is excluded. These changing-policy bins explain why an early-stop conclusion at 500 or 1,000 updates would have missed the later suffix recovery.

## Critic calibration and scope

At the SoftSARSA temperature .02/200, normalized empirical-return RMSE across the 808 warm-up rows falls from .9071/.8660 to .5656/.5676 for the two critics. On the same 26 held-out baseline rows it falls from .8675/.7598 to .5329/.5135. These targets use actual sampled log densities, exclude initial-action entropy, and end at true terminals. The unchanged learner reconstructs recorded-action density through a clamped inverse tanh; that distinction is disclosed in the protocol.

The held-out row aggregate is duration-weighted: 18 of its 26 rows belong to the one failing episode, despite four of five episodes succeeding. Pooled critic error is not an episode success rate, and reduced calibration error does not by itself establish a useful action gradient. This run does not compare normalization alone against an otherwise identical control.

Final learned alpha is .000305859925 (raw-reward equivalent .061171985). Its checkpoint is [online_2000/agent.pt](checkpoints/online_2000/agent.pt), SHA-256 `e4a73d890963a70afc9337db7d29d345c38e192218a7b81dbd51cd6589232892`. Full learner/optimizer/replay/RNG and physical episode state are preserved at all six checkpoint generations. They are exact evidence snapshots; automatic continuation from this artifact is not implemented.

All assisted records remain ineligible for primitives. Positive SAC update counts permit a later independent cold-policy collection only if that collection meets every existing physical and provenance requirement. Current cold failures provide no basis for exporting trajectories or starting the main 20-demonstration/BC/critic/online pipeline.

## Next bounded test

The final prefix-7 result supports a separate frozen-checkpoint descending screen at prefixes 6 through 1, stopping at the first stage with fewer than four of five qualified stochastic successes. Use new seeds 17141–17145 plus deterministic seed 42, the complete learned distribution, and the original physically executed prefix. This operational gate is not a statistical robustness claim. There is no automatic training extension or stage advancement in this completed run.

Four artifact-local synthetic protocol tests passed before execution; they test math/schema/replay boundaries rather than task success. Numerical source files were unchanged. The separately recorded production suite remains 247 passed, with two existing Gym bounds warnings; it was not rerun for this artifact-only experiment.
