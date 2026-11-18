# Variant 80 `ms_dagger_from79`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_set": [8], "demonstration_include_mpc": false, "actor_bc_lambda": 100.0, "online_policy_interactions": 4000, "checkpoints": [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], "initial_from": ["bootstrap_variant_0079_ms_suffix1to8_weighted_from74", "online_4000"], "actor_lr_override": 0.0001, "eval_stochastic_trials": 3, "actor_bc_std_target": 0.01, "prefix_sources": ["mpc_cold_demonstrations_0001", 120], "suffix_set": [1, 2, 3, 4, 5, 6, 7, 8], "demonstration_suffix_rows": 8, "suffix_weights": [2, 2, 2, 2, 1, 1, 1, 1], "extra_sources": [["mpc_dagger_labels_0001", 200]], "holdout_source_names": ["seed_194", "seed_195", "seed_196", "seed_197"], "eval_source_names": ["seed_100", "seed_101"]}`. Inherited checkpoint: `bootstrap_variant_0079_ms_suffix1to8_weighted_from74/checkpoints/online_4000`. Hypothesis: DAgger-style aggregation: the 0079 policy fails on every held-out demonstration (seeds 194-196, 0/90 at every suffix length) because it emits controls that release the object at states it has never seen. The verified cold-reset MPC was run from the states the frozen 0079 policy actually visits on those demonstrations (mpc_dagger_labels_0001: demonstration prefix, 0-3 policy controls, then MPC to a strict force-qualified success, independently replayed bit-for-bit). Those episodes join the training sources; their demonstration rows are the MPC segment after the last policy control only. The three labelled demonstrations and the never-seen seed_197 are evaluated as held-out sources. Continues 0079 online_4000 with the same weighted suffix-1-8 recipe.

Stage status `complete_sac_stage` (exit=0 (inferred from summary.json and all eight checkpoints being written; the launcher did not capture the exit status), 1724 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 5500, 'online_updates': 27750}; 500 critic-pretraining and 4000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: overridden to 0.0001 for this stage (inherited 0.0001; ramp fraction unchanged). Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over all demonstration rows, added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.01)) added to the same actor step. Handoff noise: off.

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'s1': 48, 's2': 42, 's3': 45, 's4': 38, 's5': 13, 's6': 24, 's7': 20, 's8': 26}, force-qualified successes {'s1': 33, 's2': 26, 's3': 17, 's4': 10, 's5': 1, 's6': 1, 's7': 1, 's8': 0} (informational; the gate is the 681-row demonstration buffer).
- Online: 4000 interactions; completed episodes by prefix {'s1': 162, 's2': 137, 's3': 141, 's4': 146, 's5': 78, 's6': 74, 's7': 86, 's8': 93}, successes {'s1': 149, 's2': 77, 's3': 41, 's4': 25, 's5': 10, 's6': 5, 's7': 3, 's8': 3}; outcomes {'palm_support': 191, 'drop': 388, 'success': 313, 'truncated': 25}; handoffs terminated before the policy 0; unfinished last suffix True. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 16229; none entered replay.
- Actor BC loss first/last: 1.362 / 0.03282; spread-anchor loss first/last: 0.0005816 / 3.029e-05; alpha raw-equivalent last 0.8616.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 0.0036, std 0.0100, bias -1.34/-1.26; after_pretraining: L2 0.0036, std 0.0100, bias -0.96/-0.97; online_500: L2 0.1496, std 0.0100, bias -1.92/-1.97; online_1000: L2 0.0644, std 0.0102, bias -1.82/-1.89; online_1500: L2 0.0248, std 0.0101, bias -1.73/-1.69; online_2000: L2 0.0163, std 0.0101, bias -1.53/-1.54; online_2500: L2 0.0074, std 0.0101, bias -1.85/-1.88; online_3000: L2 0.0057, std 0.0101, bias -1.99/-1.99; online_3500: L2 0.0069, std 0.0101, bias -2.06/-2.14; online_4000: L2 0.0063, std 0.0100, bias -2.06/-2.08.

## Frozen evaluations (in-run; 3 stochastic seeds per prefix)

| Checkpoint | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch | Cold det / stoch |
| --- | --- | --- | --- | --- |
| pretraining_baseline | fail / 0/3 | fail / 1/3 | pass / 3/3 | fail |
| online_500 | fail / 0/3 | fail / 0/3 | fail / 1/3 | fail |
| online_1000 | fail / 0/3 | fail / 0/3 | fail / 0/3 | fail |
| online_1500 | fail / 0/3 | fail / 0/3 | pass / 1/3 | fail |
| online_2000 | fail / 0/3 | fail / 0/3 | fail / 0/3 | fail |
| online_2500 | fail / 0/3 | fail / 0/3 | fail / 1/3 | fail |
| online_3000 | fail / 0/3 | fail / 0/3 | fail / 1/3 | fail |
| online_3500 | fail / 0/3 | fail / 0/3 | fail / 0/3 | fail |
| online_4000 | fail / 0/3 | fail / 0/3 | pass / 0/3 | fail / 0/5 |

The cold column counts only the cold-reset trials (an earlier rendering of this table had added the multi-source cells to it; corrected 2026-10-03).

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

No large-sample result for this stage at the time of writing.

## Multi-source evaluation: held-out demonstrations

Cells: `P3` = deterministic pass and 3/3 stochastic trials, `F0` = deterministic fail and 0/3; the eight entries per source are suffix lengths 1–8. 30-trial tables: stochastic successes out of 30 fresh seeds per cell, `D` = the deterministic trial passed. Seeds 194–196 are the demonstrations the inherited `0079` online_4000 policy failed at every suffix length (0/90); seed_197 (32 controls) was never loaded by `0079`, `0080` or `0081` as a training source.

In-run (3 stochastic seeds per cell; seeds 100 and 101 are training sources, 194–197 are held out):

| Checkpoint | seed_100 | seed_101 | seed_194 | seed_195 | seed_196 | seed_197 |
| --- | --- | --- | --- | --- | --- | --- |
| pretraining_baseline | P3 P3 F1 P2 P2 F2 F0 F0 | P3 P3 F0 F0 F0 F0 F0 F0 | F0 F0 F0 F0 F0 F0 F0 F0 | F0 F0 F0 F0 F0 F0 F0 F0 | F0 F0 F0 F0 F0 F0 F0 F0 | P3 F0 F0 F0 F0 F0 F0 F0 |
| online_500 | P3 P3 P1 P3 P2 F0 F0 F0 | P2 F0 F0 F0 F0 F0 F0 F0 | F0 F0 F0 P0 F0 F0 F0 F0 | F3 P3 P0 F0 F0 F0 F0 F0 | P3 P0 F0 P3 F0 F0 F0 F0 | F0 F3 F0 F0 F0 F0 F0 F0 |
| online_1000 | P3 F1 P3 P0 F3 F1 F0 F1 | P2 F0 F0 F0 F0 F0 F0 F0 | F0 F0 F0 F0 F0 F0 F0 F0 | F2 P3 F0 F0 F0 F0 F0 F0 | P3 F0 F0 F0 F0 F0 F0 F0 | P3 F0 F0 F0 F0 F0 F0 F0 |
| online_1500 | P3 P3 P0 F0 F0 P2 F0 F0 | P3 F0 F0 F0 F0 F0 F0 F0 | F2 F0 F0 F0 F0 F0 F0 F0 | P3 P3 F0 F0 F0 F0 F0 F0 | P3 F0 F0 F0 F0 F0 F0 F0 | P3 F0 F0 F0 F0 F0 F0 F0 |
| online_2000 | P3 P3 F0 F0 P0 P1 F0 P0 | P3 F0 F1 F0 F0 F0 F0 F0 | F2 F0 F0 F1 F0 F0 F0 F0 | P3 P2 F1 F0 F0 F0 F0 F0 | P3 F0 F0 F0 F0 F0 F0 F0 | P3 F0 F0 F0 F0 F0 F0 F0 |
| online_2500 | P3 P3 F0 P0 F1 F1 F0 F0 | P3 F0 F1 F0 F0 F0 F0 F0 | P2 F0 F0 F1 F0 F0 F0 F0 | P2 P3 F2 F1 F0 F0 F0 F0 | P3 F0 F0 F0 F0 F0 F0 F0 | P3 F0 F0 F0 F0 F0 F0 F0 |
| online_3000 | P3 P3 F0 F0 F1 F1 F2 F1 | P3 P2 F1 F0 F0 F0 F0 F0 | P1 F0 F0 F0 F0 F0 F0 F0 | P2 P1 F0 F0 F0 F0 F0 F0 | P3 F2 F0 P0 F0 F0 F0 F0 | P3 F0 F0 F0 P0 F0 F0 F0 |
| online_3500 | P3 P3 F0 F0 F2 P2 F0 F0 | P3 F3 F0 F0 F0 F0 F0 F0 | P3 F0 F0 F0 F0 F0 F0 F0 | P1 F2 F2 F0 P0 F0 F0 F0 | P3 P0 F0 F0 F0 F0 F0 F0 | P3 F0 F0 F0 F0 F0 F0 F0 |
| online_4000 | P3 P3 F2 F0 P2 F1 F0 F0 | P3 F0 F0 F0 F0 F0 F0 F0 | P1 F0 F0 F1 F0 F0 F0 F0 | F1 F0 F0 F0 F0 F0 F0 F0 | P3 F0 F0 F0 F0 F0 F0 F0 | P3 F0 F0 F0 F0 F0 F0 F0 |

30-trial frozen evaluation of online_4000 (`frozen_checkpoint_evaluation_0001/results_multisource/bootstrap_variant_0080_ms_dagger_from79/online_4000.json`):

| source | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed_194 | 17/30 D | 0/30 | 0/30 | 7/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_195 | 17/30 | 15/30 | 2/30 | 0/30 | 1/30 | 0/30 | 0/30 | 0/30 |
| seed_196 | 30/30 D | 2/30 | 0/30 | 1/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_197 | 30/30 D | 0/30 | 0/30 | 1/30 | 0/30 | 0/30 | 0/30 | 0/30 |

Reference checkpoints on the same demonstrations (`0079` online_4000, the inherited policy; `0081` online_4000, the control stage that added the three original demonstrations as sources instead of the labelled episodes):

| source | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed_194 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_195 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_196 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_197 | 30/30 D | 0/30 | 1/30 | 2/30 | 3/30 | 0/30 | 0/30 | 0/30 |

| source | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed_194 | 30/30 D | 19/30 D | 7/30 | 5/30 | 3/30 | 0/30 | 0/30 | 0/30 |
| seed_195 | 30/30 D | 15/30 D | 2/30 | 1/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_196 | 30/30 D | 18/30 D | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_197 | 27/30 D | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |

Failure modes of online_4000 at the two shortest suffixes (reason, policy controls before the boundary, count): seed_194 s1 palm support after 4–6 controls (10 of 13 failures; the recorded suffix is one control, the stage's labelled episode for that state takes 3 MPC controls); seed_194 s2 palm support after 3 controls (22 of 30); seed_195 s1 drop after 2–3 controls (9 of 13); seed_196 s2 drop after 5–10 controls (21 of 28); seed_197 s2 drop after 5–8 controls or palm support after 4 (26 of 30).

**Reading.** Against the inherited policy the labelled episodes raise the three labelled demonstrations from 0/90 to 64/90 at suffix length 1 and from 0/90 to 17/90 at length 2 (lengths 3–8: at most 8/90). The control stage, which added the three original demonstrations without any label at a policy-visited state, does better at every length up to 3 (90/90 with 3/3 deterministic passes, 52/90 with 3/3, 9/90), so the eleven labels at policy-visited states added nothing beyond what the demonstrations' own suffixes supply, and the MPC-replanned suffixes of the seventeen demonstration-state episodes (1–10 controls where the recorded suffix has s) are a worse target for the s-control evaluation than the recorded suffix. On the never-trained seed_197 the three checkpoints are indistinguishable: 30/30, 30/30 and 27/30 at length 1 (a state the inherited policy already handled) and 0/30 at length 2 in every case. DAgger-style labelling therefore does not change the generalization picture: competence remains per-trajectory, and the failure on an unseen trajectory begins at its second-to-last control. Nothing here is primitive-eligible; no cold-reset trial succeeded.

### Training-set comparison (fixed eight demonstrations, 30 trials per cell)

Seeds 100, 101, 110, 120, 136, 139, 157 and 158, all training sources of `0079`, `0080` and `0081` (`frozen_checkpoint_evaluation_0001/results_multisource/bootstrap_variant_0080_ms_dagger_from79/online_4000__seed_100_seed_101_seed_110_seed_120_seed_136_seed_139_seed_157_seed_158.json`; the `0079` row is the record in `docs/WORK_STATE.md`, its result file having been overwritten by a later held-out run):

| checkpoint | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `0079` online_4000 | 240/240 | 182/240 | 101/240 | 59/240 | 69/240 | 39/240 | 19/240 | 15/240 |
| `0080` online_4000 | 240/240 det 8/8 | 174/240 det 5/8 | 98/240 det 4/8 | 46/240 det 0/8 | 57/240 det 3/8 | 34/240 det 1/8 | 8/240 det 0/8 | 10/240 det 1/8 |
| `0081` online_4000 | 219/240 det 7/8 | 168/240 det 6/8 | 113/240 det 4/8 | 70/240 det 2/8 | 53/240 det 1/8 | 21/240 det 1/8 | 14/240 det 0/8 | 7/240 det 0/8 |

The DAgger stage does not improve on the inherited checkpoint on its own training trajectories (below it at every length from 2 to 8 by 4–11 of 240) and the control stage trades 21 of 240 at length 1 and 14 at length 2 for gains at lengths 3–4; neither supersedes `0079` online_4000 on this set.

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
