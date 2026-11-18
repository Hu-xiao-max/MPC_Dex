# Variant 90 `cyl_ms_from_sac0001`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_set": [8], "demonstration_include_mpc": false, "actor_bc_lambda": 100.0, "online_policy_interactions": 4000, "checkpoints": [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], "actor_lr_override": 0.0001, "eval_stochastic_trials": 3, "actor_bc_std_target": 0.01, "prefix_sources": ["cylinder_demonstrations_0001", 120], "suffix_set": [1, 2, 3, 4, 5, 6, 7, 8], "demonstration_suffix_rows": 8, "eval_source_count": 2, "holdout_source_count": 3, "suffix_weights": [2, 2, 2, 2, 1, 1, 1, 1], "env_overrides": {"object_shape": "circular"}, "eval_recorded_prefixes": false, "eval_cold_every_point": true}`. Inherited checkpoint: `bootstrap_suffix_sac_0001/checkpoints/online_2000`. Hypothesis: Control for index 89: the same cylinder curriculum from the archived SAC0001 prefix-7 checkpoint (the default initial checkpoint of this family) instead of the hexagon multi-source checkpoint, to separate what the inherited hexagon competence contributes on the new object. Caveat: the two inherited checkpoints also differ in their learner counters and temperature (SAC0001 starts the actor-step ramp at 0.4 of 1e-4 and alpha at 3e-4 raw-equivalent, 0082 at the full step and 6e-3), so the pair attributes the joint effect of weights, ramp and temperature.

Stage status `complete_sac_stage` (exit=0 (inferred from summary.json and all eight checkpoints being written; the launcher did not capture the exit status), 2774 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 1000, 'online_updates': 6000}; 500 critic-pretraining and 4000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: overridden to 0.0001 for this stage (inherited 0.0003; ramp fraction unchanged). Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over all demonstration rows, added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.01)) added to the same actor step. Handoff noise: off.

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'s1': 48, 's2': 42, 's3': 45, 's4': 38, 's5': 13, 's6': 24, 's7': 20, 's8': 26}, force-qualified successes {'s1': 14, 's2': 2, 's3': 0, 's4': 0, 's5': 0, 's6': 0, 's7': 0, 's8': 0} (informational; the gate is the 168-row demonstration buffer).
- Online: 4000 interactions; completed episodes by prefix {'s1': 68, 's2': 59, 's3': 59, 's4': 57, 's5': 29, 's6': 34, 's7': 41, 's8': 29}, successes {'s1': 47, 's2': 21, 's3': 8, 's4': 5, 's5': 1, 's6': 1, 's7': 0, 's8': 0}; outcomes {'palm_support': 93, 'drop': 188, 'success': 83, 'truncated': 12}; handoffs terminated before the policy 0; unfinished last suffix True. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 6941; none entered replay.
- Actor BC loss first/last: 3.887 / 0.03647; spread-anchor loss first/last: 3.512 / 0.001034; alpha raw-equivalent last 0.0972.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 3.0713, std 0.0619, bias -1.54/-1.62; after_pretraining: L2 3.0713, std 0.0619, bias -0.94/-0.98; online_500: L2 1.8116, std 0.0094, bias -0.73/-0.74; online_1000: L2 1.5616, std 0.0101, bias -0.81/-0.79; online_1500: L2 1.3360, std 0.0106, bias -1.18/-1.16; online_2000: L2 1.4160, std 0.0104, bias -1.38/-1.38; online_2500: L2 1.5089, std 0.0105, bias -1.34/-1.35; online_3000: L2 1.3944, std 0.0103, bias -0.92/-0.99; online_3500: L2 1.3389, std 0.0101, bias -0.69/-0.68; online_4000: L2 1.3502, std 0.0101, bias -0.60/-0.61.

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

| Checkpoint | Cold | seed_101 s1..s8 | seed_102 s1..s8 | seed_153 s1..s8 | seed_154 s1..s8 | seed_155 s1..s8 |
| --- | --- | --- | --- | --- | --- | --- |
| pretraining_baseline | F0/5 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F1/3 P2/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_500 | F0/5 | F2/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P2/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_1000 | F0/5 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_1500 | F0/5 | P2/3 F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_2000 | F0/5 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_2500 | F0/5 | F0/3 F1/3 F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F2/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_3000 | F0/5 | F0/3 F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P0/3 F2/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_3500 | F0/5 | F0/3 P2/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F1/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_4000 | F0/5 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

The 30-trial results are in the campaign section below.

## Cylinder campaign control (`0090`, the same recipe from the archived SAC0001 checkpoint): 30-trial frozen evaluation

Environment: the archived task with `object_shape: circular` (40 mm; `geometry_transfer.json` records the override under which the unchanged inherited checkpoint is trained and evaluated). Evaluated sources: seeds 101 and 102 (training) and 153, 154, 155 (held out; 13, 15 and 10 controls). 30 fresh-seed stochastic trials plus one deterministic trial per cell (`frozen_checkpoint_evaluation_0001/results_multisource/<stage>/<checkpoint>.json`; cold reset in `results_shared/`). The only spec difference from `0089` is the inherited checkpoint; the two also differ in learner counters and temperature (SAC0001 starts the actor-step ramp at 0.4 of 1e-4 and its learned alpha is 3.1e-4 normalized, 0.061 raw-equivalent; `0082` starts at the full step with 6.4e-3 normalized, 1.27 raw-equivalent), so the comparison attributes their joint effect.

**online_4000** (cold reset 0/30 (deterministic fail))

| source | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed_101 | 3/30 | 9/30 | 1/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_102 | 30/30 D | 30/30 D | 8/30 | 1/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_153 (held out) | 12/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_154 (held out) | 30/30 D | 1/30 | 0/30 | 1/30 | 0/30 | 0/30 | 0/30 | 4/30 |
| seed_155 (held out) | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| training sources | 33/60 (1 D) | 39/60 (1 D) | 9/60 (0 D) | 1/60 (0 D) | 0/60 (0 D) | 0/60 (0 D) | 0/60 (0 D) | 0/60 (0 D) |
| held-out sources | 42/90 (1 D) | 1/90 (0 D) | 0/90 (0 D) | 1/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 4/90 (0 D) |
| all five | 75/150 | 40/150 | 9/150 | 2/150 | 0/150 | 0/150 | 0/150 | 4/150 |

**Reading.** At the matched final checkpoint the control is at or above `0089` online_4000 at lengths 1-2 (75/150 vs 71/150; 40/150 vs 21/150) and below it at lengths 3-4 (9/150 vs 39/150; 2/150 vs 8/150); against the better of `0089`'s two frozen-evaluated checkpoints (online_2500: 90, 48, 35, 15, 2 of 150 at lengths 1-5) it is below at lengths 1-5 (75, 40, 9, 2, 0 of 150), 0/150 for both at lengths 6-7, and above at length 8 (4/150 against 0/150; the four are seed_154). Its held-out last control is 42/90 (seed_154 30/30 D, seed_153 12/30, seed_155 0/30), beyond which the held-out rows are 1, 0, 1, 0, 0, 0, 4 of 90. The in-run last-control cells peaked at online_2000 (5 of 5 deterministic passes, 13/15 stochastic) and were not frozen-evaluated there. The online stream was far thinner than `0089`'s (47/68, 21/59, 8/59 successes at lengths 1-3 against 143/162, 93/138, 59/145; changing-policy data). The `0082` start (its weights, full actor step and higher temperature jointly) therefore carries over the competence at lengths 3-4 and the training signal, and the later cylinder stages inherit that lineage; at the last control the two starts are similar. No cold-reset trial succeeded; nothing here is primitive-eligible; the circular object is a disclosed deviation from the paper's hexagonal object, and the training-stream counts are changing-policy, assisted, demonstration-mixed episodes, not frozen-policy evidence.

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
