# Variant 81 `ms_add_heldout_demos_from79`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_set": [8], "demonstration_include_mpc": false, "actor_bc_lambda": 100.0, "online_policy_interactions": 4000, "checkpoints": [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], "initial_from": ["bootstrap_variant_0079_ms_suffix1to8_weighted_from74", "online_4000"], "actor_lr_override": 0.0001, "eval_stochastic_trials": 3, "actor_bc_std_target": 0.01, "prefix_sources": ["mpc_cold_demonstrations_0001", 120], "suffix_set": [1, 2, 3, 4, 5, 6, 7, 8], "demonstration_suffix_rows": 8, "suffix_weights": [2, 2, 2, 2, 1, 1, 1, 1], "holdout_source_names": ["seed_197"], "eval_source_names": ["seed_100", "seed_101", "seed_194", "seed_195", "seed_196"]}`. Inherited checkpoint: `bootstrap_variant_0079_ms_suffix1to8_weighted_from74/checkpoints/online_4000`. Hypothesis: Control for the DAgger stage: the same continuation of 0079 online_4000 with the previously held-out demonstrations (seeds 194-196) simply added as ordinary training sources, without labels at policy-visited states; only seed_197 stays held out. Separates the effect of labelling policy-visited states from the effect of adding those three trajectories.

Stage status `complete_sac_stage` (exit=0 (inferred from summary.json and all eight checkpoints being written; the launcher did not capture the exit status), 1896 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 5500, 'online_updates': 27750}; 500 critic-pretraining and 4000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: overridden to 0.0001 for this stage (inherited 0.0001; ramp fraction unchanged). Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over all demonstration rows, added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.01)) added to the same actor step. Handoff noise: off.

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'s1': 48, 's2': 42, 's3': 45, 's4': 38, 's5': 13, 's6': 24, 's7': 20, 's8': 26}, force-qualified successes {'s1': 37, 's2': 29, 's3': 9, 's4': 10, 's5': 3, 's6': 0, 's7': 0, 's8': 1} (informational; the gate is the 504-row demonstration buffer).
- Online: 4000 interactions; completed episodes by prefix {'s1': 163, 's2': 139, 's3': 145, 's4': 149, 's5': 78, 's6': 74, 's7': 87, 's8': 94}, successes {'s1': 158, 's2': 94, 's3': 61, 's4': 45, 's5': 9, 's6': 10, 's7': 3, 's8': 3}; outcomes {'palm_support': 183, 'drop': 347, 'success': 383, 'truncated': 16}; handoffs terminated before the policy 0; unfinished last suffix True. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 15844; none entered replay.
- Actor BC loss first/last: 0.5966 / 0.00184; spread-anchor loss first/last: 0.0002429 / 1.08e-05; alpha raw-equivalent last 0.8628.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 0.0036, std 0.0100, bias -1.34/-1.26; after_pretraining: L2 0.0036, std 0.0100, bias -0.74/-0.74; online_500: L2 0.0359, std 0.0100, bias -1.62/-1.55; online_1000: L2 0.0182, std 0.0101, bias -1.44/-1.40; online_1500: L2 0.0113, std 0.0101, bias -1.45/-1.51; online_2000: L2 0.0058, std 0.0101, bias -1.68/-1.65; online_2500: L2 0.0127, std 0.0100, bias -1.67/-1.60; online_3000: L2 0.0089, std 0.0100, bias -1.86/-1.86; online_3500: L2 0.0055, std 0.0100, bias -1.87/-1.91; online_4000: L2 0.0076, std 0.0100, bias -1.90/-1.96.

## Frozen evaluations (in-run; 3 stochastic seeds per prefix)

| Checkpoint | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch | Cold det / stoch |
| --- | --- | --- | --- | --- |
| pretraining_baseline | fail / 0/3 | fail / 1/3 | pass / 3/3 | fail |
| online_500 | fail / 0/3 | fail / 0/3 | pass / 3/3 | fail |
| online_1000 | fail / 0/3 | fail / 0/3 | pass / 3/3 | fail |
| online_1500 | fail / 0/3 | fail / 0/3 | pass / 3/3 | fail |
| online_2000 | fail / 0/3 | fail / 0/3 | pass / 3/3 | fail |
| online_2500 | fail / 0/3 | fail / 0/3 | pass / 3/3 | fail |
| online_3000 | fail / 0/3 | fail / 0/3 | pass / 3/3 | fail |
| online_3500 | fail / 0/3 | fail / 0/3 | pass / 3/3 | fail |
| online_4000 | fail / 0/3 | fail / 0/3 | pass / 3/3 | fail / 0/5 |

The cold column counts only the cold-reset trials (an earlier rendering of this table had added the multi-source cells to it; corrected 2026-10-03).

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

No large-sample result for this stage at the time of writing.

## Multi-source evaluation: the three added demonstrations and the held-out seed_197

Cells: `P3` = deterministic pass and 3/3 stochastic trials, `F0` = deterministic fail and 0/3; the eight entries per source are suffix lengths 1–8. 30-trial tables: stochastic successes out of 30 fresh seeds per cell, `D` = the deterministic trial passed. Seeds 194–196 are the demonstrations the inherited `0079` online_4000 policy failed at every suffix length (0/90); seed_197 (32 controls) was never loaded by `0079`, `0080` or `0081` as a training source. In this stage seeds 194–196 are ordinary training sources (63 training sources in all) and only seed_197 is held out.

In-run (3 stochastic seeds per cell):

| Checkpoint | seed_100 | seed_101 | seed_194 | seed_195 | seed_196 | seed_197 |
| --- | --- | --- | --- | --- | --- | --- |
| pretraining_baseline | P3 P3 F1 P3 P2 F2 F1 F1 | P3 P3 F0 F0 F0 F0 F0 F0 | F0 F0 F0 F0 F0 F0 F0 F0 | F0 F0 F0 F0 F0 F0 F0 F0 | F0 F0 F0 F0 F0 F0 F0 F0 | P3 F0 F0 F0 F1 F0 F0 F0 |
| online_500 | P3 F0 F3 F3 F2 F1 F2 F0 | F3 F0 F0 F0 F0 F0 F0 F0 | P3 F0 F0 F0 F0 F0 F0 F0 | F0 P3 F1 F0 F0 F0 F0 F0 | P3 F0 F0 F0 F0 F0 F0 F0 | P3 F0 F0 F0 F0 F0 F0 F0 |
| online_1000 | P3 P3 F1 F0 F0 F0 F0 F0 | P3 F0 F1 F0 F0 F0 F0 F0 | P3 F0 F0 F2 F0 F0 F0 F0 | P2 P3 F0 F0 F0 F0 F0 F0 | P2 F0 F0 F0 F0 F0 F0 F0 | P3 F0 F0 F0 F0 F0 F0 F0 |
| online_1500 | P3 P3 F0 F0 P0 F1 F0 F0 | P3 F0 P0 F0 F0 F0 F0 F0 | P3 F0 F0 F0 F0 F0 F0 F0 | F1 P2 F0 F0 F1 F0 F0 F0 | F0 P1 F0 F0 F0 F0 F0 F0 | P3 F0 F0 F0 F0 F0 F0 F0 |
| online_2000 | P3 P3 F0 F0 F0 F0 F0 F0 | P3 F0 F1 F0 F0 F0 F0 F0 | P3 P1 P1 F0 F0 F0 F0 F0 | P3 F0 F0 F0 F0 F0 F0 F0 | F0 F1 F1 F0 F0 F0 F0 F0 | P3 F0 F0 F0 F0 F0 F0 F0 |
| online_2500 | P3 P3 P0 F0 F0 P1 F0 P0 | P3 F0 F1 F0 F0 F0 F0 F0 | P3 F2 P2 P1 F0 F0 F0 F0 | P3 F3 F0 F0 F0 F0 F1 F0 | F0 P2 F0 F0 F0 F0 F0 F0 | P3 F0 F0 F0 F0 F0 F0 F0 |
| online_3000 | P3 P3 F1 P2 F0 P1 P0 F2 | P3 F0 P3 F1 F0 F0 F0 F0 | P3 F1 P2 F0 F0 F0 F0 F0 | P3 F1 F0 F0 F0 F0 F0 F0 | P1 P2 F0 F0 F0 F0 F0 F0 | P2 F0 F0 F0 F0 F0 F0 F0 |
| online_3500 | P3 P3 F1 F0 F1 F0 F0 F0 | P3 F0 P1 F0 F0 F0 F0 F0 | P3 P1 P0 F2 F0 F0 F0 F0 | P3 F2 F0 F0 P0 F0 F0 F0 | P3 F1 F0 F0 F0 F0 F0 F0 | F0 F0 F0 F0 F0 F0 F0 F0 |
| online_4000 | P3 P3 F1 F0 F1 F0 F0 F0 | P3 F0 F1 F1 F0 F0 F0 F0 | P3 P2 F0 F2 F0 F0 F0 F0 | P3 P1 F0 F0 F1 F0 F0 F0 | P3 P1 F0 F0 F0 F0 F0 F0 | P2 F0 F0 F0 F0 F0 F0 F0 |

30-trial frozen evaluation of online_4000 (`frozen_checkpoint_evaluation_0001/results_multisource/bootstrap_variant_0081_ms_add_heldout_demos_from79/online_4000.json`):

| source | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed_194 | 30/30 D | 19/30 D | 7/30 | 5/30 | 3/30 | 0/30 | 0/30 | 0/30 |
| seed_195 | 30/30 D | 15/30 D | 2/30 | 1/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_196 | 30/30 D | 18/30 D | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_197 | 27/30 D | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |

Reference checkpoints on the same demonstrations (`0079` online_4000, the inherited policy, for which 194–196 were held out; `0080` online_4000, the DAgger stage that trained on MPC-labelled episodes branching from these demonstrations instead):

| source | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed_194 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_195 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_196 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_197 | 30/30 D | 0/30 | 1/30 | 2/30 | 3/30 | 0/30 | 0/30 | 0/30 |

| source | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed_194 | 17/30 D | 0/30 | 0/30 | 7/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_195 | 17/30 | 15/30 | 2/30 | 0/30 | 1/30 | 0/30 | 0/30 | 0/30 |
| seed_196 | 30/30 D | 2/30 | 0/30 | 1/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_197 | 30/30 D | 0/30 | 0/30 | 1/30 | 0/30 | 0/30 | 0/30 | 0/30 |

Failure modes of online_4000 at suffix length 2 (reason, policy controls, count): seed_194 drop after 3 (6 of 11); seed_195 drop after 3–6 (15 of 15); seed_196 drop after 3–4 (8 of 12); seed_197 drop after 2 controls in all 30 stochastic failures and the deterministic trial. At length 1 seed_197 fails by palm support after 2 controls in 3 of 30.

**Reading.** Adding the three demonstrations as sources makes their last control deterministic and 90/90 stochastic within 4,000 updates, their last two controls 52/90 with 3/3 deterministic passes, and their last three 9/90; the inherited policy had 0/90 at every length. This is more than the DAgger stage achieved on the same demonstrations (64/90, 17/90, 2/90), so the demonstrations' own recorded suffixes are the better supervision. The never-trained seed_197 stays at the inherited level: 27/30 at length 1 (30/30 before this stage) and 0/30 at length 2, with every failure a drop at the second control. The per-trajectory pattern is unchanged: a trajectory is completed once it is in the training set and not otherwise, beyond its final control. Nothing here is primitive-eligible; no cold-reset trial succeeded.

### Training-set comparison (fixed eight demonstrations, 30 trials per cell)

Seeds 100, 101, 110, 120, 136, 139, 157 and 158, all training sources of `0079`, `0080` and `0081` (`frozen_checkpoint_evaluation_0001/results_multisource/bootstrap_variant_0081_ms_add_heldout_demos_from79/online_4000__seed_100_seed_101_seed_110_seed_120_seed_136_seed_139_seed_157_seed_158.json`; the `0079` row is the record in `docs/WORK_STATE.md`, its result file having been overwritten by a later held-out run):

| checkpoint | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `0079` online_4000 | 240/240 | 182/240 | 101/240 | 59/240 | 69/240 | 39/240 | 19/240 | 15/240 |
| `0081` online_4000 | 219/240 det 7/8 | 168/240 det 6/8 | 113/240 det 4/8 | 70/240 det 2/8 | 53/240 det 1/8 | 21/240 det 1/8 | 14/240 det 0/8 | 7/240 det 0/8 |
| `0080` online_4000 | 240/240 det 8/8 | 174/240 det 5/8 | 98/240 det 4/8 | 46/240 det 0/8 | 57/240 det 3/8 | 34/240 det 1/8 | 8/240 det 0/8 | 10/240 det 1/8 |

On the fixed set this stage is below the inherited checkpoint at lengths 1, 2 and 5–8 (219 against 240 at length 1; 168 against 182 at length 2) and above it at lengths 3–4 (113 against 101; 70 against 59): the three added trajectories were learned at a small cost to the others, and no stage supersedes `0079` online_4000 on this set.

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
