# Recorded-only demonstrations: 64 of 441 training successes, one frozen prefix-6 seed, prefix-7 retention lost then partly regained, cold still fails

**Removing the five MPC rows from the SAC0004 recipe changes the outcome in kind but does not solve the handoff:** online training successes rose from 2/94 (SAC0004) to 64/441, the deterministic actor mean at the prefix-6 handoff converged to within L2 .043 of recorded control 7, yet the frozen prefix-6 deterministic trial failed at every checkpoint and only stochastic seed 17505 passed (at online 500 and 2,000, not 1,000). Prefix-7 retention fell to 0/6 at online 500 as the hypothesis predicted, then partly recovered (deterministic pass plus 2/5 at 1,000 and 2,000). All cold trials failed. The stage completed its full budget (500 mixed critic updates, 2,000 mixed SAC updates, exit 0, 494.0 s).

## Actual execution

- Inherited SAC0001 final checkpoint (SHA `e4a73d89…`); final counters BC 12,000 / pretraining 1,000 / online 4,000, optimizer steps critic 5,000 and actor/alpha 4,000, as prescribed. The training protocol differs from SAC0004 only in `demonstration_include_mpc`; seeds match, and the warm-up raw reward sum (−24,784.28) is bit-identical to SAC0004's. The evaluation schedule additionally covers prefix 8 (19 trials per point versus SAC0004's 13), which does not touch the learner.
- Warm-up: 256 frozen-actor prefix-6 episodes, 812 suffix rows, all drops after 2–5 controls, zero successes (informational).
- Demonstration buffer: 3 recorded rows (source controls 7–9, one terminal episode), 77 demonstration and 179 online rows per batch. Pretraining bootstrapped 40–64 demonstration rows entropy-free per update; their target mean moved 1.074 → 1.135 (non-monotonic, peak 1.363); critic loss .313 → .0035.
- Online: 2,000 interactions, 441 completed assisted episodes (changing policy), 64 force-qualified successes after 3–7 suffix controls (58 at 3, 1 at 4, 4 at 5, 1 at 7; mean 3.2): 4/108 in updates 1–500, 9/103 in 501–1,000, 51/230 in 1,001–2,000; first at episode 14 (cumulative update 2,084). 282 drops, 95 palm-support failures, 0 horizon truncations. The last episode remains open at 8 controls with three loaded fingers; replay holds 2,812 rows. All 4,188 physical prefix controls replayed bit-exactly and never entered replay.
- Learned alpha .000306 → .000377 (raw-equivalent .0612 → .0753); online critic loss rose from a first-500-update mean of .015 to a last-500 mean of .110 (per-500 means .015, .047, .097, .110; final update .211, the series maximum).

## Value calibration

| Rows | Before | After pretraining | Online 500 | 1,000 | 2,000 |
| --- | --- | --- | --- | --- | --- |
| 3 recorded demonstration rows, bias | −.68 / −.62 | −.0003 / +.008 | −.31 / −.31 | −.11 / −.09 | −.42 / −.17 |
| same, RMSE | .88 / .93 | .012 / .012 | .37 / .37 | .24 / .24 | .75 / .44 |
| 812 warm-up rows, RMSE | .319 / .336 | .078 / .076 | — | — | — |
| 15 held-out rows, RMSE | .329 / .339 | .026 / .027 | — | — | — |

Deterministic L2 distance to recorded control 7 at the handoff: .413 (unchanged by pretraining) → .107 → .131 → .043; handoff latent std .067 → .098 → .134 → .176. In SAC0004 the same distance went .413 → 1.78 → .63 → 1.24 while the distance to the MPC label fell to .33.

## Frozen evaluations

| Checkpoint | Prefix-6 det. | Prefix-6 stochastic 17501–17505 | Prefix-7 det. | Prefix-7 stochastic 17521–17525 | Prefix-8 det. | Prefix-8 stochastic 17531–17535 | Noisy trials | Cold det. | Cold stochastic 17511–17515 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline (pretrained) | Drop, 3 controls | 0/5, all drops (3 controls each) | Pass, 2 | 5/5 | Pass, 1 | 5/5 | none scheduled | Drop, control 1 | not scheduled |
| Online 500 | Drop, 4 | 1/5 (17505 pass at 3 controls, 4 fingers; 3 drops, 1 palm) | Drop, 4 | 0/5 (4 drops, 1 palm) | Pass, 1 | 5/5 | none | Drop, 2 | not scheduled |
| Online 1,000 | Drop, 4 | 0/5 (4 drops, 1 palm) | Pass, 2 | 2/5 (17524, 17525 pass; 2 palm, 1 drop) | Pass, 1 | 5/5 | none | Palm, 2 | not scheduled |
| Online 2,000 | Palm after 5 (4 fingers loaded, 5.45 rad, peak 5.70) | 1/5 (17505 pass at 3 controls; 3 drops, 1 palm) | Pass, 3 | 2/5 (17524, 17525 pass; 2 drops, 1 palm) | Pass, 1 | 5/5 | none | Drop, 2 | 0/5 (2 drops, 3 palm; 2–3 controls) |

All 81 evaluation trials (19 at each of the first three points, 24 at online 2,000) terminated; none truncated, all prefix_physics_bitexact true, perturbed_last_prefix_control false, handoff_terminated_before_policy false; every evaluation summary records training_rng_restored true, learner_tensors_and_counters_unchanged true, eligible_for_primitive false. No noisy trials were scheduled.

## Interpretation and scope

The hypothesis held on its first clause: without the MPC rows the actor mean no longer drifts toward an unfinishable continuation but toward recorded control 7 (L2 .043 here versus 1.24 in SAC0004, whose mean instead closed to .33 on the MPC label), and the changing-policy stream produced 64 successes with the same warm-up. It held on retention loss at online 500, but the later partial recovery of prefix 7 and the unbroken prefix-8 6/6 were not predicted (SAC0004 did not evaluate prefix 8). The frozen prefix-6 handoff is still not solved: the deterministic trial ends in a drop or a four-finger palm brace at every checkpoint even though its first action lies within L2 .043 of the recorded control at 2,000, so matching the first label is not sufficient, as SAC0004 also concluded; the one passing stochastic seed is the same seed twice, not five. A mean that sits on the recorded control while latent std grows to .176 is consistent with the narrow success basin around control 7; the rising critic loss and the negative demonstration-row bias at 2,000 indicate the values were not stable either.

This is a single seed with no replicate; the comparison with SAC0004 is seed-matched only through warm-up, after which the batches differ. Every training episode is assisted and demonstration-mixed and is not primitive evidence; training successes are changing-policy, not frozen results. All cold trials failed, so no checkpoint here is deployable, and none qualifies as an initialization for a later stage without a separate decision.

Inputs and protocol: `protocol.json`, `execution_manifest.json`, `input_bindings.json`; results: `summary.json`, `warmup_record.json`, `metrics.jsonl`, `episodes/`, `evaluations/`, `checkpoints/`, `demonstration_buffer.json`, `demonstration_calibration_*.json`, `warmup_calibration_*`, `heldout_calibration_*`, `unfinished_suffix/`; final hashes: `manifest.json`.


_Audit note: drafted by an auditing agent from the artifact files and independently refuted-checked; 3 claim(s) corrected by the verifier before writing._
