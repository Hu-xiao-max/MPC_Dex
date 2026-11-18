# Variant 93 `cyl_ms_s1to6_from89`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_set": [8], "demonstration_include_mpc": false, "actor_bc_lambda": 100.0, "online_policy_interactions": 4000, "checkpoints": [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], "initial_from": ["bootstrap_variant_0089_cyl_ms_from82", "online_2500"], "actor_lr_override": 0.0001, "eval_stochastic_trials": 3, "actor_bc_std_target": 0.01, "prefix_sources": ["cylinder_demonstrations_0001", 24], "suffix_set": [1, 2, 3, 4, 5, 6], "demonstration_suffix_rows": 6, "suffix_weights": [2, 2, 2, 1, 1, 1], "holdout_source_names": ["seed_153", "seed_154", "seed_155"], "eval_source_names": ["seed_101", "seed_102"], "env_overrides": {"object_shape": "circular"}, "eval_recorded_prefixes": false, "eval_cold_every_point": true}`. Inherited checkpoint: `bootstrap_variant_0089_cyl_ms_from82/checkpoints/online_2500`. Hypothesis: Extension of the cylinder curriculum to suffix lengths 1-6 (weights 2:2:2:1:1:1, so the short suffixes keep two thirds of the episodes) from the reference cylinder checkpoint 0089 online_2500 (90/150, 48/150, 35/150, 15/150 at lengths 1-4; held-out 60/90 at the last control; cold 0/30), on the same 24-source snapshot, for 4,000 updates as the hexagon extension 0074 needed. The staged start from the hexagon checkpoint (0091) did not beat the direct jump on the cylinder; this stage tests whether lengths 4-6 become reliable with the extra updates, with the cold reset evaluated at every checkpoint.

Stage status `complete_sac_stage` (exit=0 (inferred from summary.json and all checkpoints being written; the launcher did not capture the exit status), 1100 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 7000, 'online_updates': 38250}; 500 critic-pretraining and 4000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: overridden to 0.0001 for this stage (inherited 0.0001; ramp fraction unchanged). Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over all demonstration rows, added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.01)) added to the same actor step. Handoff noise: off.

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'s1': 64, 's2': 58, 's3': 51, 's4': 15, 's5': 36, 's6': 32}, force-qualified successes {'s1': 62, 's2': 37, 's3': 21, 's4': 3, 's5': 1, 's6': 3} (informational; the gate is the 126-row demonstration buffer).
- Online: 4000 interactions; completed episodes by prefix {'s1': 252, 's2': 228, 's3': 222, 's4': 110, 's5': 136, 's6': 147}, successes {'s1': 245, 's2': 169, 's3': 105, 's4': 32, 's5': 15, 's6': 10}; outcomes {'success': 576, 'drop': 345, 'palm_support': 170, 'truncated': 4}; handoffs terminated before the policy 0; unfinished last suffix False. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 15790; none entered replay.
- Actor BC loss first/last: 0.0026 / 0.0004506; spread-anchor loss first/last: 2.997e-06 / 2.158e-06; alpha raw-equivalent last 2.4620.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 0.0293, std 0.0100, bias -1.95/-1.97; after_pretraining: L2 0.0293, std 0.0100, bias -1.22/-1.28; online_500: L2 0.0545, std 0.0100, bias -2.70/-2.66; online_1000: L2 0.0657, std 0.0100, bias -2.93/-2.96; online_1500: L2 0.0551, std 0.0100, bias -2.72/-2.77; online_2000: L2 0.0516, std 0.0100, bias -2.96/-2.98; online_2500: L2 0.0569, std 0.0100, bias -3.12/-3.05; online_3000: L2 0.0526, std 0.0100, bias -3.07/-3.16; online_3500: L2 0.0382, std 0.0100, bias -2.75/-2.96; online_4000: L2 0.0480, std 0.0100, bias -3.19/-3.16.

## Frozen evaluations (in-run; no recorded-prefix evaluations; stochastic seeds per prefix)

| Checkpoint |  | Cold det / stoch |
| --- | --- |
| pretraining_baseline | fail / 0/5 |
| online_500 | fail / 0/5 |
| online_1000 | fail / 0/5 |
| online_1500 | fail / 0/5 |
| online_2000 | fail / 0/5 |
| online_2500 | fail / 0/5 |
| online_3000 | fail / 0/5 |
| online_3500 | fail / 0/5 |
| online_4000 | fail / 0/5 |

## Multi-source evaluation (in-run; a cell is the deterministic trial P/F followed by stochastic successes/trials; per source the cells run over the suffix lengths; shared k = the state after k shared cold-start controls)

| Checkpoint | Cold | seed_101 s1..s6 | seed_102 s1..s6 | seed_153 s1..s6 | seed_154 s1..s6 | seed_155 s1..s6 |
| --- | --- | --- | --- | --- | --- | --- |
| pretraining_baseline | F0/5 | F0/3 P2/3 P1/3 F0/3 F0/3 F0/3 | P3/3 P2/3 P3/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_500 | F0/5 | P3/3 F2/3 F0/3 F1/3 F0/3 F0/3 | P3/3 P3/3 F1/3 F1/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_1000 | F0/5 | P3/3 P1/3 P3/3 F0/3 F0/3 F0/3 | P3/3 P3/3 F3/3 P1/3 F1/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 P2/3 F1/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_1500 | F0/5 | P3/3 F2/3 P3/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P3/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 P1/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_2000 | F0/5 | P3/3 P2/3 P3/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P3/3 P2/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 P0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_2500 | F0/5 | F1/3 F2/3 P1/3 F0/3 F0/3 F0/3 | P3/3 P2/3 P3/3 P0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_3000 | F0/5 | F2/3 P3/3 P1/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P2/3 F1/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_3500 | F0/5 | F1/3 P2/3 P2/3 F0/3 F0/3 F0/3 | P3/3 P2/3 P3/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_4000 | F0/5 | P3/3 P3/3 F2/3 F0/3 F0/3 F0/3 | P3/3 P2/3 P3/3 F3/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

The 30-trial results are in the campaign section below.

## Cylinder campaign, lengths 1-6 on the 24-source snapshot (`0093`, from `0089` online_2500, 4,000 updates): 30-trial frozen evaluation

Environment: the archived task with `object_shape: circular` (40 mm; `geometry_transfer.json` records the override under which the unchanged inherited checkpoint is trained and evaluated). Evaluated sources: seeds 101 and 102 (training) and 153, 154, 155 (held out; 13, 15 and 10 controls). 30 fresh-seed stochastic trials plus one deterministic trial per cell (`frozen_checkpoint_evaluation_0001/results_multisource/<stage>/<checkpoint>.json`; cold reset in `results_shared/`).

**online_2000** (cold reset: no 30-trial evaluation at this checkpoint; the in-run cold trials were deterministic fail and 0/5)

| source | s1 | s2 | s3 | s4 | s5 | s6 |
| --- | --- | --- | --- | --- | --- | --- |
| seed_101 | 30/30 D | 19/30 D | 27/30 D | 1/30 | 0/30 | 0/30 |
| seed_102 | 30/30 D | 29/30 D | 29/30 D | 17/30 D | 1/30 | 0/30 |
| seed_153 (held out) | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_154 (held out) | 0/30 | 0/30 | 4/30 D | 0/30 | 0/30 | 3/30 |
| seed_155 (held out) | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| training sources | 60/60 (2 D) | 48/60 (2 D) | 56/60 (2 D) | 18/60 (1 D) | 1/60 (0 D) | 0/60 (0 D) |
| held-out sources | 0/90 (0 D) | 0/90 (0 D) | 4/90 (1 D) | 0/90 (0 D) | 0/90 (0 D) | 3/90 (0 D) |
| all five | 60/150 | 48/150 | 60/150 | 18/150 | 1/150 | 3/150 |

**online_4000** (cold reset 0/30 (deterministic fail))

| source | s1 | s2 | s3 | s4 | s5 | s6 |
| --- | --- | --- | --- | --- | --- | --- |
| seed_101 | 29/30 D | 17/30 D | 6/30 | 0/30 | 0/30 | 0/30 |
| seed_102 | 30/30 D | 27/30 D | 29/30 D | 6/30 | 0/30 | 3/30 |
| seed_153 (held out) | 30/30 D | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_154 (held out) | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_155 (held out) | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| training sources | 59/60 (2 D) | 44/60 (2 D) | 35/60 (1 D) | 6/60 (0 D) | 0/60 (0 D) | 3/60 (0 D) |
| held-out sources | 30/90 (1 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) |
| all five | 89/150 | 44/150 | 35/150 | 6/150 | 0/150 | 3/150 |

**Reading.** Against the start (`0089` online_2500: 90, 48, 35, 15, 2, 0 of 150 at lengths 1-6; training 30/60, 47/60, 32/60, 15/60; held-out 60/90, 1/90, 3/90, 0/90): at online_2000 the five-source rates are 60/150, 48/150, 60/150, 18/150, 1/150, 3/150 (length 3 above the start, length 4 at 18 with seed_102 17/30 D, held-out last control 0/90); at online_4000 they are 89/150, 44/150, 35/150, 6/150, 0/150, 3/150. The length-1 total returns to the start's level with the composition inverted: training 59/60 (from 30/60) and held-out 30/90 (from 60/90; seed_153 alone, seed_154 0/30 at both checkpoints). Length 4 fell from 15 to 6 of 150 and held-out lengths 2-5 from 1, 3, 0, 1 to 0 of 90. The online stream was 245/252, 169/228, 105/222 at lengths 1-3 and 32/110, 15/136, 10/147 at 4-6 (changing-policy data). The profile differs from the hexagon's: `0082` online_4000 tapered over lengths 1-8 on its eight training sources (238, 199, 111, 69, 46, 27, 15, 7 of 240; its held-out source is not in that file), whereas here at online_4000 the training-source rows are 59, 44, 35, 6, 0, 3 of 60 at lengths 1-6 (the five-source totals above include 90 held-out trials per length): single digits at length 4 (18/150 over five sources at online_2000, seed_102 17/30 D) and 0-3 beyond. Open-loop tolerance tables (`control_period_0001/tolerance_seed_*_circular.json`, three replays per radius) exist for three of the five evaluated sources: seed_101 (10 controls) is fully tolerant at every radius up to 0.03 only over the last two controls, seed_102 (13 controls) over the last three, seed_153 (13 controls) over the last two, with earlier controls intolerant down to radius 0.0003; seeds 154 and 155 have no table. No cold-reset trial succeeded; nothing here is primitive-eligible; the circular object is a disclosed deviation from the paper's hexagonal object, and the training-stream counts are changing-policy, assisted, demonstration-mixed episodes, not frozen-policy evidence.

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
