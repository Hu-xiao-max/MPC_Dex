# Variant 94 `cyl_ms_s1to6_all_from89`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_set": [8], "demonstration_include_mpc": false, "actor_bc_lambda": 100.0, "online_policy_interactions": 4000, "checkpoints": [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], "initial_from": ["bootstrap_variant_0089_cyl_ms_from82", "online_2500"], "actor_lr_override": 0.0001, "eval_stochastic_trials": 3, "actor_bc_std_target": 0.01, "prefix_sources": ["cylinder_demonstrations_0001", 120], "suffix_set": [1, 2, 3, 4, 5, 6], "demonstration_suffix_rows": 6, "suffix_weights": [2, 2, 2, 1, 1, 1], "holdout_source_names": ["seed_153", "seed_154", "seed_155"], "eval_source_names": ["seed_101", "seed_102"], "env_overrides": {"object_shape": "circular"}, "eval_recorded_prefixes": false, "eval_cold_every_point": true}`. Inherited checkpoint: `bootstrap_variant_0089_cyl_ms_from82/checkpoints/online_2500`. Hypothesis: Same as index 93 (lengths 1-6 weighted from 0089 online_2500, 4,000 updates) over every verified cylinder demonstration (54 at stamp time; 51 training) instead of the 24-source snapshot, with the same three held-out sources, to measure what source count contributes on the cylinder, as 0074 did on the hexagon.

Stage status `complete_sac_stage` (exit=0 (inferred from summary.json and all checkpoints being written; the launcher did not capture the exit status), 1106 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 7000, 'online_updates': 38250}; 500 critic-pretraining and 4000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: overridden to 0.0001 for this stage (inherited 0.0001; ramp fraction unchanged). Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over all demonstration rows, added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.01)) added to the same actor step. Handoff noise: off.

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'s1': 64, 's2': 58, 's3': 51, 's4': 15, 's5': 36, 's6': 32}, force-qualified successes {'s1': 42, 's2': 24, 's3': 15, 's4': 3, 's5': 0, 's6': 0} (informational; the gate is the 306-row demonstration buffer).
- Online: 4000 interactions; completed episodes by prefix {'s1': 223, 's2': 212, 's3': 208, 's4': 104, 's5': 119, 's6': 138}, successes {'s1': 195, 's2': 127, 's3': 93, 's4': 30, 's5': 18, 's6': 14}; outcomes {'drop': 331, 'success': 477, 'palm_support': 185, 'truncated': 11}; handoffs terminated before the policy 0; unfinished last suffix False. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 15727; none entered replay.
- Actor BC loss first/last: 2.703 / 0.01365; spread-anchor loss first/last: 0.002927 / 6.103e-06; alpha raw-equivalent last 2.4622.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 0.0293, std 0.0100, bias -1.95/-1.97; after_pretraining: L2 0.0293, std 0.0100, bias -1.58/-1.55; online_500: L2 0.4927, std 0.0101, bias -3.29/-3.43; online_1000: L2 0.2253, std 0.0099, bias -3.45/-3.38; online_1500: L2 0.0762, std 0.0099, bias -2.76/-2.81; online_2000: L2 0.0651, std 0.0100, bias -2.81/-2.90; online_2500: L2 0.0585, std 0.0100, bias -3.40/-3.50; online_3000: L2 0.1150, std 0.0100, bias -3.72/-3.80; online_3500: L2 0.1253, std 0.0100, bias -3.98/-4.12; online_4000: L2 0.0772, std 0.0100, bias -3.88/-3.85.

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
| pretraining_baseline | F0/5 | F0/3 P3/3 P0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P3/3 F2/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_500 | F0/5 | F0/3 P3/3 P1/3 F0/3 F0/3 F0/3 | P3/3 P2/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F1/3 F1/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_1000 | F0/5 | P3/3 P3/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P3/3 F1/3 P0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_1500 | F0/5 | F0/3 P1/3 P2/3 F0/3 F0/3 F0/3 | P3/3 F3/3 P3/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_2000 | F0/5 | P3/3 F0/3 P3/3 F1/3 F0/3 F0/3 | P3/3 P1/3 F0/3 P3/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F1/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_2500 | F0/5 | F0/3 P2/3 P3/3 F1/3 F0/3 F0/3 | P3/3 P3/3 F0/3 F1/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F1/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_3000 | F0/5 | F0/3 P3/3 P3/3 F0/3 F0/3 F0/3 | P3/3 F2/3 F2/3 F1/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_3500 | F0/5 | P3/3 P3/3 P3/3 F0/3 F0/3 F0/3 | P3/3 P2/3 F2/3 F1/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P1/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_4000 | F0/5 | P3/3 F1/3 P2/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P2/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

The 30-trial results are in the campaign section below.

## Cylinder campaign, lengths 1-6 over all 54 demonstrations (`0094`, 51 training sources, from `0089` online_2500, 4,000 updates): 30-trial frozen evaluation

Environment: the archived task with `object_shape: circular` (40 mm; `geometry_transfer.json` records the override under which the unchanged inherited checkpoint is trained and evaluated). Evaluated sources: seeds 101 and 102 (training) and 153, 154, 155 (held out; 13, 15 and 10 controls). 30 fresh-seed stochastic trials plus one deterministic trial per cell (`frozen_checkpoint_evaluation_0001/results_multisource/<stage>/<checkpoint>.json`; cold reset in `results_shared/`).

**online_2000** (cold reset: no 30-trial evaluation at this checkpoint; the in-run cold trials were deterministic fail and 0/5)

| source | s1 | s2 | s3 | s4 | s5 | s6 |
| --- | --- | --- | --- | --- | --- | --- |
| seed_101 | 16/30 D | 10/30 | 28/30 D | 0/30 | 0/30 | 0/30 |
| seed_102 | 30/30 D | 18/30 D | 2/30 | 12/30 D | 0/30 | 0/30 |
| seed_153 (held out) | 30/30 D | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_154 (held out) | 0/30 | 0/30 | 0/30 | 0/30 | 3/30 | 0/30 |
| seed_155 (held out) | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| training sources | 46/60 (2 D) | 28/60 (1 D) | 30/60 (1 D) | 12/60 (1 D) | 0/60 (0 D) | 0/60 (0 D) |
| held-out sources | 30/90 (1 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 3/90 (0 D) | 0/90 (0 D) |
| all five | 76/150 | 28/150 | 30/150 | 12/150 | 3/150 | 0/150 |

**online_4000** (cold reset 0/30 (deterministic fail))

| source | s1 | s2 | s3 | s4 | s5 | s6 |
| --- | --- | --- | --- | --- | --- | --- |
| seed_101 | 30/30 D | 4/30 | 14/30 D | 0/30 | 1/30 | 0/30 |
| seed_102 | 30/30 D | 27/30 D | 13/30 D | 5/30 | 0/30 | 0/30 |
| seed_153 (held out) | 30/30 D | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_154 (held out) | 30/30 D | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_155 (held out) | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| training sources | 60/60 (2 D) | 31/60 (1 D) | 27/60 (2 D) | 5/60 (0 D) | 1/60 (0 D) | 0/60 (0 D) |
| held-out sources | 60/90 (2 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) |
| all five | 120/150 | 31/150 | 27/150 | 5/150 | 1/150 | 0/150 |

**Reading.** At online_4000 the last control reaches 120/150 over the five sources with 4 of 5 deterministic passes (training 60/60; held-out 60/90, seeds 153 and 154 at 30/30 D, seed_155 0/30), the campaign's highest five-source count; the held-out rate equals the parent `0089` online_2500's 60/90 and had been 30/90 at online_2000, so it is a return to the parent's level after a swing, as `0093` swung 0/90 to 30/90 over the same interval, and no source-count effect on the last control can be separated from that swing. Lengths 2-4 are 31/150, 27/150, 5/150, below both `0093` online_4000 (44, 35, 6) and the parent (48, 35, 15): with 51 sources the stage is below both compared checkpoints at lengths 2-4 (at lengths 5-6 `0093` online_4000 is 0/150 and 3/150, the parent 2/150 and 0/150). Length 4 carried a deterministic pass at online_2000 (seed_102 12/30 D), as in `0093` (17/30 D) and `0089` online_4000 (seed_101 3/30 D); lengths 5-6 are 1/150 and 0/150. Online stream: 195/223, 127/212, 93/208, 30/104, 18/119, 14/138 (changing-policy data). Across the six 30-trial checkpoints of `0089`, `0093` and this stage the frozen competence on the training sources is 17-48 of 60 at length 2 and 27-56 of 60 at length 3 (over all five sources 21-48 and 27-60 of 150), a deterministic pass at length 4 on one source at some checkpoints, and 0-3 of 150 at lengths 5-6; no checkpoint moved it further in 4,000 updates. Continuations: `0095` (lengths 1-8 weighted, 4,000 more updates) and `0096` (lengths 1-9 weighted toward long suffixes, evaluated on the 8-9-control sources, for which a suffix of 8-9 is the cold reset). No cold-reset trial succeeded; nothing here is primitive-eligible; the circular object is a disclosed deviation from the paper's hexagonal object, and the training-stream counts are changing-policy, assisted, demonstration-mixed episodes, not frozen-policy evidence.

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
