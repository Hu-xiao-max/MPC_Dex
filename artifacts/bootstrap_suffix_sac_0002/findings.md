# Gated prefix-6 warm-up: no positive support, no learning

**The 256-episode frozen-policy warm-up at prefix 6 produced zero force-qualified successes, so the learning gate failed and the stage finished data-only** (status `warmup_only_no_positive_support`, exit 0, 64.56 s measured loop). No critic pretraining, SAC update, MPC query, stage advance or trajectory export occurred. The inherited learner is unchanged: the `warmup_complete` checkpoint's agent payload is identical to `initial.pt` (SHA `e4a73d890963a70afc9337db7d29d345c38e192218a7b81dbd51cd6589232892`), counters remain BC 12,000 / critic pretraining 500 / online SAC 2,000, and optimizer steps remain critic 2,500 / actor 2,000 / alpha 2,000.

## Actual execution

- Stage frozen at prefix 6 from the reviewed descending screen binding (`stage_binding.json`); `execution_manifest.json` was verified before the loop and again before the summary was written.
- Every one of the 256 episodes cold-reset with physical seed 42 and physically re-executed the six original recorded controls; observations, actions, rewards, flags, integration states and all 300 prefix physics samples were checked bit-for-bit against the source at every reset. 1,536 physical prefix controls were executed; none entered replay.
- Stochastic sampling used new global seed 17091 with the frozen inherited actor, singleton (1,60) CUDA float32 inference, one thread and TF32 off. All 815 actual policy suffix transitions are in the fresh stage replay with raw rewards and sampled log probabilities; the prior 2,808-row replay was copied as lineage only and its open final row was neither imported nor closed.
- All 256 episodes terminated by drop. No episode was truncated, so every empirical soft-return target is complete. Independent force audit found no palm-support violation and zero peak palm normal force in every episode. Sixteen episodes register table or other contact, in each case only after finger support was already lost, i.e. the dropped object landing.

## Warm-up outcomes

| Policy suffix controls before drop | Episodes |
| --- | ---: |
| 2 | 13 |
| 3 | 184 |
| 4 | 58 |
| 5 | 1 |

Handoff force-qualified angle after the six-control prefix is 5.258153506 rad. Peak force-qualified progress beyond handoff across the 256 episodes is .0175 rad minimum, .0719 rad mean and .1605 rad maximum; only two episodes exceed 5.4 rad, and none approaches the 2π requirement. Final force-qualified angles span −4.2214 to 4.6041 rad. Mean raw suffix return is −94.73; total warm-up raw reward is −24,250.89. Every episode ended with zero force-bearing fingers.

The first policy action at the prefix-6 handoff differs from the originally recorded seventh control by a mean absolute .0556 per dimension and maximum .5166, with mean L2 distance .4215; its per-dimension standard deviation across episodes is .0164. The frozen policy therefore acts consistently at this state and consistently loses the grasp within 200–500 ms. This is a fixed 256-sample observation at one physical handoff state; it does not prove that a supporting action does not exist at prefix 6.

## Baseline evaluations (frozen inherited checkpoint)

| Trial | Result |
| --- | --- |
| Prefix-6 deterministic (seed 42) | Fail, drop after 3 controls; final .491119 rad, peak 5.337721 rad, identical to the earlier screen |
| Prefix-6 stochastic 17201–17205 | 0/5, drops after 4/3/3/3/3 controls; peaks 5.306–5.337 rad |
| Cold deterministic | Fail, drop at control 1 |
| Prefix-7 retention deterministic | Pass, 2 policy controls, final 6.660177 rad (matches SAC0001 final evaluation) |
| Prefix-7 retention stochastic 17221–17225 | 5/5, two policy controls each; final 6.592798–7.011872 rad |

Final cold stochastic trials and later checkpoints were not scheduled because the gate failed. Evaluation transitions never entered replay; training RNG and learner tensors were restored and verified unchanged after evaluation.

## Critic calibration before pretraining

On the 815 complete warm-up rows, normalized empirical-return RMSE at the inherited learned alpha (.000305859925, raw-equivalent .061171985) is .3143/.3323 for the two critics with bias +.1376/+.0530; target mean is −.4026. The 16 held-out prefix-6 baseline rows give RMSE .3303/.3501. The `after` calibration files were not produced because no pretraining ran. Lower RMSE than SAC0001's warm-up partly reflects that all of these episodes are short failures with similar returns; it is not evidence of a useful action gradient.

## Interpretation and scope

Prefix 6 is one control earlier than the supported phase. At that state the trained checkpoint holds the object for about three further controls and then drops it, in all 262 physical attempts recorded here (256 warm-up plus six baseline). Under the protocol, the absence of any positive example forbids new optimization from this data, and no automatic fallback (exploration change, noise sweep, MPC correction, reward change) was executed. Prefix-7 competence is retained exactly.

All outcomes are prefix-assisted and ineligible for primitives or main demonstrations. This stage produced no new cold-start evidence beyond the repeated control-1 drop. The next decision is which of the following separately prepared routes to take; none was run here:

1. Supply verified positive examples at the prefix-6 state from a source other than the frozen policy, e.g. a bounded MPC recovery from the recorded handoff, with corrections kept as explicit non-policy labels, before any new critic/SAC budget.
2. A modified-stage protocol that allows training on failures only, which departs from the current success-gated curriculum and would need its own review.

Inputs and protocol: `protocol.json`, `execution_manifest.json`, `stage_binding.json`; results: `summary.json`, `warmup_gate.json`, `metrics.jsonl`, `episodes/warmup/`, `evaluations/pretraining_baseline/`, `checkpoints/{initialized,warmup_complete}/`, `warmup_calibration_before.*`, `heldout_calibration_before.*`; final file hashes: `manifest.json`.
