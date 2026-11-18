# Interpretation of the fixed prefix-7 SAC run

The final checkpoint supports an **earlier-phase support screen**, not a claim of cold-policy competence. After an early regression, its frozen prefix-7 policy completed the deterministic trial and all five stochastic trials; prefix-8 retention also passed all six checks. All six final cold-policy trials failed. No main-stage or primitive promotion follows from these assisted successes.

| Frozen evaluation | Prefix-7 deterministic | Prefix-7 stochastic | Prefix-8 retention | Cold policy |
|---|---|---|---|---|
| Before critic pretraining | pass | 4/5 | deterministic + 5/5 pass | deterministic pass |
| Online 500 | fail | 1/5 | not scheduled | deterministic fail |
| Online 1,000 | fail | 1/5 | not scheduled | deterministic fail |
| Online 2,000 | pass | 5/5 | deterministic + 5/5 pass | deterministic + 5/5 fail |

The unchanged actor's 256 warmup episodes produced 165 assisted successes and 808 actual suffix transitions. Online completed-episode success counts, in consecutive groups of 100, were **56, 11, 31, 85, 97, 100, 100**, followed by **52/53**. Those groups involve a changing policy and are not frozen-policy success estimates, but their recovery agrees with the final scheduled evaluation. The five held-out seeds were fixed across checkpoints; this is a small paired diagnostic, not a general robustness estimate. All final stochastic cold trials failed on their first action.

## Distribution change, not uniform policy collapse

At the exact same prefix-7 handoff observation, the largest deterministic normalized-action change from initialization was .2896 after 500 updates and .2418 at 2,000. The latent Gaussian KL(new || initial) increased from 43.83 to 337.62 across 16 dimensions. Final latent std spans .00612–.21002, compared with the initial .05 in every dimension. Thus the final successful suffix is a substantially changed action distribution; it is not simply preservation of the BC commands or a uniform collapse of exploration variance.

At the identical cold-reset observation, final maximum action change is .9548 and latent KL is 1,074.86. The initial nominal cold success was lost. This is clear evidence of lost startup behavior under suffix-only training, without proving a particular gradient mechanism. At the exact prefix-8 state, final action change is .3583 and KL is 623.14, yet retention passes. Large change alone therefore does not imply failure. Prefix-8 retention demonstrates competence at one previously supported physical state; it does not establish cold-start generalization, sustained rotation, or coverage of every intermediate state.

## Critic evidence and limits

Read-only CUDA singleton inference used each frozen checkpoint's actual alpha and the saved stochastic evaluation actions. Targets are fully observed terminal soft returns: first-action entropy excluded, later sampled-action entropy included, rewards divided by 200. No new physics or learning occurred. All-row statistics and episode-initial statistics are separate because failures are often longer and otherwise dominate pooled averages.

| Checkpoint, sampled initial state/action | Mean normalized soft return | Mean Q1 / Q2 | Initial-action RMSE Q1 / Q2 |
|---|---:|---:|---:|
| Initial, prefix 7 | .7635 | .1640 / −.2562 | .9045 / 1.2212 |
| Critic pretraining complete, same data | .7635 | .5579 / .5917 | .7021 / .6886 |
| Online 500, prefix 7 | −.1090 | .8953 / .9490 | 1.1874 / 1.2355 |
| Online 1,000, prefix 7 | −.1299 | 1.0521 / .9994 | 1.3450 / 1.3072 |
| Online 2,000, prefix 7 | 1.1099 | 1.2256 / 1.2219 | .1172 / .1142 |
| Online 2,000, prefix 8 | 1.1504 | 1.1102 / 1.3614 | .0414 / .2112 |
| Online 2,000, cold | −.2500 | −.4228 / −.6957 | .1731 / .4460 |

Multiply normalized values by 200 for raw-reward-equivalent units. Final raw-equivalent alpha is .061172, versus initial .05. The fixed-actor pretraining comparison supports improved empirical value fitting, with remaining error; its SoftSARSA alpha was separately .02/200. During early online regression, sampled initial-action Q remained optimistic despite failed returns. At the final successful phase, predictions agree much better with those measured returns. This does **not** establish globally calibrated critics or useful gradients everywhere: later checkpoints visit different states/actions, final phase-7 samples are all successes, and each action has only one sampled continuation. Retention and cold critic errors remain uneven. The baseline held-out pool has 26 transitions, 18 from its single failed episode, so its near-zero pooled target mean cannot replace the 4/5 episode outcome or the episode-initial target above.

The protocol jointly changed curriculum, warmup/pretraining and reward/temperature scaling. It is not a pure curriculum-versus-cold causal comparison. There is nevertheless direct evidence of actual SAC learning in the prescribed assisted task: the actor was unchanged during warmup and critic pretraining, then changed during 2,000 genuine online updates, with late and frozen assisted successes. There were no new BC updates or head refinements.

## One bounded next experiment

Freeze the final 60D checkpoint and screen the **original physically recorded prefixes 6→1**, in order, stopping immediately after the first stage with fewer than four strict force-qualified completions among five stochastic suffixes. At each stage use one deterministic trial with seed 42 and five new sampling seeds **17141–17145**. Keep the original 100-control total horizon and all physics/termination rules. Reset cold, physically replay the selected original prefix, verify every initial/prefix integration and 2 ms force sample, then switch to the final policy without rebasing history or consuming sampling RNG during the prefix.

Preserve the learned state-dependent std and all checkpoint parameters; do not reinitialize std to .05. Use the actual checkpoint alpha for soft-return reporting, and separately state its raw-equivalent scale. The trained deterministic suffix is free to differ from the old nominal nine-control suffix: only the prescribed physical prefix must be bit-exact. Save actual actions, all states, force audits and RNG; reconstruct inference independently. Every completion remains prefix-assisted, despite the source checkpoint now having genuine online SAC updates. Do not repeat the already completed cold stage, train during the screen, change its thresholds, or add branches after seeing results.

A passing support screen would justify considering an earlier fixed training phase; it is not automatic permission to advance through all phases or proof that learning there will work. If the first below-gate stage has mixed outcomes, it is a candidate with measured positive/negative support. If it has zero successes, that finite result does not prove zero support; use the preceding supported stage as the known reference. Later learning evaluations need new held-out seeds beyond these phase-selection seeds.

A behavior-reference trust region remains a possible fallback if a later bounded run loses useful behavior without recovering. The present final evidence does not justify choosing it before measuring earlier-phase support: the final successful policy moved substantially, and an overly tight tether could block that useful change. No additional training or mechanism sweep was run in this review.
