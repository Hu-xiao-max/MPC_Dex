# Variant 92 `cyl_ms_s123_from89`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_set": [8], "demonstration_include_mpc": false, "actor_bc_lambda": 100.0, "checkpoints": [500, 1000, 1500, 2000], "initial_from": ["bootstrap_variant_0089_cyl_ms_from82", "online_2500"], "actor_lr_override": 0.0001, "eval_stochastic_trials": 3, "actor_bc_std_target": 0.01, "prefix_sources": ["cylinder_demonstrations_0001", 24], "suffix_set": [1, 2, 3], "demonstration_suffix_rows": 3, "holdout_source_names": ["seed_153", "seed_154", "seed_155"], "eval_source_names": ["seed_101", "seed_102"], "env_overrides": {"object_shape": "circular"}, "eval_recorded_prefixes": false, "eval_cold_every_point": true}`. Inherited checkpoint: `bootstrap_variant_0089_cyl_ms_from82/checkpoints/online_2500`. Hypothesis: Consolidation candidate: the best cylinder checkpoint so far (0089 online_2500: 90/150 at the last control over five evaluated sources, 60/90 on the three held-out ones, 48/150 at the last two) continued with the narrow suffix set 1-3 on the same 24-source snapshot, against 0091 which starts the same staged curriculum from the hexagon checkpoint. The better of the two at 30 trials seeds the extension to lengths 1-6.

Stage status `complete_sac_stage` (exit=0 (inferred from summary.json and all checkpoints being written; the launcher did not capture the exit status), 655 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 7000, 'online_updates': 36250}; 500 critic-pretraining and 2000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: overridden to 0.0001 for this stage (inherited 0.0001; ramp fraction unchanged). Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over all demonstration rows, added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.01)) added to the same actor step. Handoff noise: off.

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'s1': 99, 's2': 72, 's3': 85}, force-qualified successes {'s1': 90, 's2': 44, 's3': 29} (informational; the gate is the 63-row demonstration buffer).
- Online: 2000 interactions; completed episodes by prefix {'s1': 265, 's2': 249, 's3': 263}, successes {'s1': 248, 's2': 167, 's3': 139}; outcomes {'success': 554, 'palm_support': 76, 'drop': 142, 'truncated': 5}; handoffs terminated before the policy 0; unfinished last suffix False. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 13090; none entered replay.
- Actor BC loss first/last: 0.0004007 / 0.0003691; spread-anchor loss first/last: 3.63e-06 / 5.637e-06; alpha raw-equivalent last 2.0200.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 0.0686, std 0.0100, bias -0.83/-0.86; after_pretraining: L2 0.0686, std 0.0100, bias -0.40/-0.51; online_500: L2 0.0548, std 0.0100, bias -0.90/-0.97; online_1000: L2 0.0522, std 0.0100, bias -1.23/-1.27; online_1500: L2 0.0740, std 0.0100, bias -1.13/-1.37; online_2000: L2 0.0601, std 0.0100, bias -1.21/-1.24.

## Frozen evaluations (in-run; no recorded-prefix evaluations; stochastic seeds per prefix)

| Checkpoint |  | Cold det / stoch |
| --- | --- |
| pretraining_baseline | fail / 0/5 |
| online_500 | fail / 0/5 |
| online_1000 | fail / 0/5 |
| online_1500 | fail / 0/5 |
| online_2000 | fail / 0/5 |

## Multi-source evaluation (in-run; a cell is the deterministic trial P/F followed by stochastic successes/trials; per source the cells run over the suffix lengths; shared k = the state after k shared cold-start controls)

| Checkpoint | Cold | seed_101 s1..s3 | seed_102 s1..s3 | seed_153 s1..s3 | seed_154 s1..s3 | seed_155 s1..s3 |
| --- | --- | --- | --- | --- | --- | --- |
| pretraining_baseline | F0/5 | F0/3 P3/3 P0/3 | P3/3 P3/3 P3/3 | P3/3 F0/3 F0/3 | P3/3 F0/3 F0/3 | F0/3 F0/3 F0/3 |
| online_500 | F0/5 | F0/3 P1/3 F0/3 | P3/3 P2/3 P2/3 | P3/3 F0/3 F0/3 | P3/3 F0/3 F0/3 | F0/3 F0/3 F0/3 |
| online_1000 | F0/5 | F0/3 F2/3 P3/3 | P3/3 F2/3 P3/3 | P3/3 F0/3 F0/3 | F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 |
| online_1500 | F0/5 | F0/3 P3/3 P2/3 | P3/3 P1/3 P2/3 | P3/3 F0/3 F0/3 | F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 |
| online_2000 | F0/5 | F0/3 F2/3 P2/3 | P3/3 F2/3 P3/3 | F1/3 F0/3 F0/3 | F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 |

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

The 30-trial results are in the campaign section below.

## Cylinder campaign, narrow continuation (`0092`, lengths 1-3 from `0089` online_2500, 2,000 updates): 30-trial frozen evaluation

Environment: the archived task with `object_shape: circular` (40 mm; `geometry_transfer.json` records the override under which the unchanged inherited checkpoint is trained and evaluated). Evaluated sources: seeds 101 and 102 (training) and 153, 154, 155 (held out; 13, 15 and 10 controls). 30 fresh-seed stochastic trials plus one deterministic trial per cell (`frozen_checkpoint_evaluation_0001/results_multisource/<stage>/<checkpoint>.json`; cold reset in `results_shared/`).

**online_1000** (cold reset: no 30-trial evaluation at this checkpoint; the in-run cold trials were deterministic fail and 0/5)

| source | s1 | s2 | s3 |
| --- | --- | --- | --- |
| seed_101 | 1/30 | 14/30 | 26/30 D |
| seed_102 | 30/30 D | 13/30 | 29/30 D |
| seed_153 (held out) | 30/30 D | 0/30 | 0/30 |
| seed_154 (held out) | 0/30 | 0/30 | 0/30 |
| seed_155 (held out) | 0/30 | 0/30 | 0/30 |
| training sources | 31/60 (1 D) | 27/60 (0 D) | 55/60 (2 D) |
| held-out sources | 30/90 (1 D) | 0/90 (0 D) | 0/90 (0 D) |
| all five | 61/150 | 27/150 | 55/150 |

**online_2000** (cold reset 0/30 (deterministic fail))

| source | s1 | s2 | s3 |
| --- | --- | --- | --- |
| seed_101 | 12/30 | 15/30 | 25/30 D |
| seed_102 | 30/30 D | 25/30 | 20/30 D |
| seed_153 (held out) | 9/30 | 0/30 | 0/30 |
| seed_154 (held out) | 0/30 | 0/30 | 0/30 |
| seed_155 (held out) | 0/30 | 0/30 | 0/30 |
| training sources | 42/60 (1 D) | 40/60 (0 D) | 45/60 (2 D) |
| held-out sources | 9/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) |
| all five | 51/150 | 40/150 | 45/150 |

**Reading.** Relative to its start (`0089` online_2500: training 30/60, 47/60, 32/60 and held-out 60/90, 1/90, 3/90 at lengths 1-3) the training sources' last control rose (31/60 at online_1000, a one-success difference, 42/60 at online_2000), length 3 rose (55/60, 45/60) and length 2 fell to 27/60 and rose to 40/60, still below the start's 47/60, while the held-out last control fell from 60/90 to 30/90 and 9/90 (seed_153 alone at both checkpoints, 30/30 D then 9/30; seed_154 fell from the parent's 30/30 D to 0/30 at both) and the held-out lengths 2-3 to 0/90: the parent's own continuation with its wide suffix set moved the same way (`0089` online_2500 to online_4000: training last control 30/60 to 41/60, length 3 32/60 to 39/60, length 2 47/60 to 17/60, held-out last control 60/90 to 30/90 with seed_154 30/30 D to 0/30), so the change is not attributable to the narrow suffix set. Its online stream (248/265, 167/249, 139/263) is the campaign's highest by count at lengths 1 and 3 and below `0093`'s rate at lengths 1-2 (changing-policy data). Departure from the pre-registered rule: the hypothesis said the better of `0091` and `0092` at 30 trials would seed the extension; neither matched the parent `0089` online_2500 at the held-out last control (their better checkpoints, `0091` online_2000 and `0092` online_1000: 41/90 and 30/90 against 60/90) or over the five sources at length 1 (80/150 and 61/150 against 90/150), so the parent was kept as the seed of `0093` and `0094`. No cold-reset trial succeeded; nothing here is primitive-eligible; the circular object is a disclosed deviation from the paper's hexagonal object, and the training-stream counts are changing-policy, assisted, demonstration-mixed episodes, not frozen-policy evidence.

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
