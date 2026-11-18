# Variant 95 `cyl_ms_s1to8_cont_from94`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_set": [8], "demonstration_include_mpc": false, "actor_bc_lambda": 100.0, "online_policy_interactions": 4000, "checkpoints": [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], "initial_from": ["bootstrap_variant_0094_cyl_ms_s1to6_all_from89", "online_4000"], "actor_lr_override": 0.0001, "eval_stochastic_trials": 3, "actor_bc_std_target": 0.01, "prefix_sources": ["cylinder_demonstrations_0001", 120], "suffix_set": [1, 2, 3, 4, 5, 6, 7, 8], "demonstration_suffix_rows": 8, "suffix_weights": [2, 2, 2, 2, 1, 1, 1, 1], "holdout_source_names": ["seed_153", "seed_154", "seed_155"], "eval_source_names": ["seed_101", "seed_172"], "env_overrides": {"object_shape": "circular"}, "eval_recorded_prefixes": false, "eval_cold_every_point": true}`. Inherited checkpoint: `bootstrap_variant_0094_cyl_ms_s1to6_all_from89/checkpoints/online_4000`. Hypothesis: Continuation of the all-source cylinder stage (0094 online_4000: 120/150 at the last control, held-out 60/90; 31, 27, 5 of 150 at lengths 2-4) with lengths 1-8 weighted 2:2:2:2:1:1:1:1 for 4,000 more updates, as the hexagon continuation 0074 gained a length; evaluated on a 10-control source (seed_101) and the shortest 8-control source (seed_172) plus the held-out trio.

Stage status `complete_sac_stage` (exit=0 (inferred from summary.json and all checkpoints being written; the launcher did not capture the exit status), 1106 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 7500, 'online_updates': 42250}; 500 critic-pretraining and 4000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: overridden to 0.0001 for this stage (inherited 0.0001; ramp fraction unchanged). Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over all demonstration rows, added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.01)) added to the same actor step. Handoff noise: off.

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'s1': 48, 's2': 42, 's3': 45, 's4': 38, 's5': 13, 's6': 24, 's7': 20, 's8': 26}, force-qualified successes {'s1': 48, 's2': 34, 's3': 24, 's4': 8, 's5': 3, 's6': 1, 's7': 0, 's8': 0} (informational; the gate is the 408-row demonstration buffer).
- Online: 4000 interactions; completed episodes by prefix {'s1': 170, 's2': 151, 's3': 151, 's4': 155, 's5': 82, 's6': 79, 's7': 95, 's8': 98}, successes {'s1': 156, 's2': 95, 's3': 63, 's4': 35, 's5': 8, 's6': 9, 's7': 8, 's8': 4}; outcomes {'drop': 381, 'palm_support': 203, 'success': 378, 'truncated': 19}; handoffs terminated before the policy 0; unfinished last suffix False. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 14396; none entered replay.
- Actor BC loss first/last: 0.7398 / 0.02209; spread-anchor loss first/last: 0.0005138 / 1.458e-05; alpha raw-equivalent last 3.6696.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 2.1926, std 0.0099, bias -4.20/-4.16; after_pretraining: L2 2.1926, std 0.0099, bias -1.39/-1.45; online_500: L2 1.0316, std 0.0100, bias -4.40/-4.35; online_1000: L2 1.0539, std 0.0101, bias -3.77/-3.87; online_1500: L2 1.0262, std 0.0100, bias -3.48/-3.57; online_2000: L2 1.0414, std 0.0100, bias -4.07/-4.21; online_2500: L2 1.0246, std 0.0100, bias -4.08/-4.04; online_3000: L2 1.0462, std 0.0100, bias -4.17/-4.20; online_3500: L2 1.0581, std 0.0100, bias -4.49/-4.63; online_4000: L2 1.0750, std 0.0101, bias -5.03/-4.99.

## Frozen evaluations (in-run; no recorded-prefix evaluations; stochastic seeds per prefix)

| Checkpoint | Cold det / stoch |
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

| Checkpoint | Cold | seed_101 s1..s8 | seed_172 s1..s8 | seed_153 s1..s8 | seed_154 s1..s8 | seed_155 s1..s8 |
| --- | --- | --- | --- | --- | --- | --- |
| pretraining_baseline | F0/5 | P3/3 F1/3 P2/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P3/3 P3/3 P1/3 F1/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_500 | F0/5 | F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P3/3 F1/3 F1/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 P2/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_1000 | F0/5 | F0/3 F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P3/3 P3/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_1500 | F0/5 | F1/3 P2/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P3/3 P2/3 P1/3 P0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F2/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_2000 | F0/5 | F0/3 P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P3/3 F1/3 F0/3 F1/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_2500 | F0/5 | P3/3 P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_3000 | F0/5 | F0/3 P3/3 F2/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P3/3 F0/3 F0/3 F1/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_3500 | F0/5 | P3/3 P2/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P3/3 P2/3 P1/3 P0/3 F1/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_4000 | F0/5 | P0/3 F3/3 F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P3/3 P2/3 F1/3 F0/3 F1/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

The 30-trial results are in the campaign section below.

## Cylinder campaign, lengths 1-8 weighted 2:2:2:2:1:1:1:1 over the 51 training demonstrations (54 verified, 3 held out; `0095`, continuation of `0094` online_4000, 4,000 more updates): 30-trial frozen evaluation

Environment: the archived task with `object_shape: circular` (40 mm; `geometry_transfer.json` records the override under which the inherited checkpoint is trained and evaluated). Evaluated sources: seed_101 (training, 10 controls), seed_172 (training, 8 controls, the shortest demonstration; the parent `0094` has no 30-trial cells on it) and seeds 153, 154, 155 (held out; 13, 15 and 10 controls). 30 fresh-seed stochastic trials plus one deterministic trial per cell (`frozen_checkpoint_evaluation_0001/results_multisource/<stage>/<checkpoint>.json`; cold reset in `results_shared/`, 30 stochastic trials plus one deterministic). A suffix of length s hands over after the source's first n - s recorded controls; on the 8-control seed_172 the length-8 cell is the cold reset itself.

**online_3500** (cold reset 0/30 (deterministic fail))

| source | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed_101 | 28/30 D | 28/30 D | 1/30 | 0/30 | 3/30 | 0/30 | 0/30 | 0/30 |
| seed_172 | 30/30 D | 30/30 D | 30/30 D | 14/30 D | 9/30 D | 0/30 D | 4/30 | 0/30 |
| seed_153 (held out) | 30/30 D | 5/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_154 (held out) | 0/30 | 30/30 D | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_155 (held out) | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| training sources | 58/60 (2 D) | 58/60 (2 D) | 31/60 (1 D) | 14/60 (1 D) | 12/60 (1 D) | 0/60 (1 D) | 4/60 (0 D) | 0/60 (0 D) |
| held-out sources | 30/90 (1 D) | 35/90 (1 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) |
| all 5 | 88/150 | 93/150 | 31/150 | 14/150 | 12/150 | 0/150 | 4/150 | 0/150 |

**online_4000** (cold reset 0/30 (deterministic fail))

| source | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed_101 | 7/30 D | 19/30 | 9/30 | 1/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_172 | 30/30 D | 30/30 D | 30/30 D | 20/30 D | 16/30 | 0/30 | 0/30 | 0/30 |
| seed_153 (held out) | 30/30 D | 3/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_154 (held out) | 0/30 | 30/30 D | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_155 (held out) | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| training sources | 37/60 (2 D) | 49/60 (1 D) | 39/60 (1 D) | 21/60 (1 D) | 16/60 (0 D) | 0/60 (0 D) | 0/60 (0 D) | 0/60 (0 D) |
| held-out sources | 30/90 (1 D) | 33/90 (1 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) |
| all 5 | 67/150 | 82/150 | 39/150 | 21/150 | 16/150 | 0/150 | 0/150 | 0/150 |

**Reading.** On seed_172 the policy completes the last three controls in 30/30 D at both checkpoints, the last four in 14/30 D then 20/30 D, the last five in 9/30 D then 16/30, and the last six in 0/30 at both with one deterministic pass at online_3500; lengths 7-8 are 4/30 and 0/30 at online_3500, 0/30 at online_4000. Its open-loop tolerance table (`control_period_0001/tolerance_seed_172_circular.json`, three replays per radius): controls 4-7 are 3/3 at every radius up to 0.03, control 3 is 3/3 up to 0.003 and 2/3 at 0.01 and 0.03, and controls 0-2 are 0-1 of 3 at 0.01 and 0.03. Lengths 1-4 hand over at controls 7-4 (the tolerant tail), length 5 at control 3 (partly tolerant) and length 6 at control 2 (intolerant), so on this source the policy's competence ends where the open-loop tolerance ends; the in-run cells at online_4000 (P3/3 P3/3 P3/3 P2/3 F1/3 F0/3 F1/3 F0/3) agree with the 30-trial rows on the deterministic column and at lengths 1-6 and 8; the in-run length-7 cell has one success in three against 0/30. On seed_101 the last control is 28/30 D at online_3500 and 7/30 D at online_4000 (parent `0094` online_4000: 30/30 D), length 2 is 28/30 D then 19/30 (parent 4/30), length 3 is 1/30 then 9/30 (parent 14/30 D) and lengths 4-8 are 0-3 of 30; swings of this size between evaluated checkpoints occurred in `0089` (seed_154 30/30 D at online_2500, 0/30 at online_4000) and `0093` (seed_153 0/30 at online_2000, 30/30 D at online_4000). Held out: the last control is 30/90 at both checkpoints (seed_153 30/30 D; seed_154 0/30, from 30/30 D at the parent; seed_155 0/30), while length 2 is 35/90 and 33/90 (seed_154 30/30 D at both, from 0/30 at the parent, plus seed_153 5/30 and 3/30): on seed_154 the policy completes the last two controls from the state after 13 recorded controls and fails the last one alone from the state after 14; held-out lengths 3-8 are 0/90 at both checkpoints. Cold reset: 0/30 with a deterministic fail at both checkpoints; all 62 trials dropped the object at the second policy control (force-qualified angle 1.38-1.76 rad). Online stream by length 1-8: 156/170, 95/151, 63/151, 35/155, 8/82, 9/79, 8/95, 4/98 (changing-policy data). Against the hypothesis (a gained length, as the hexagon continuation `0074`): on the one training source evaluated at both stages (seed_101) no length was gained, on the held-out trio the last control halved and length 2 rose by one source, and the seed_172 rows have no parent comparison; the stage's length 5 on seed_172 (9/30 D at online_3500, 16/30 at online_4000) is the cylinder campaign's highest 30-trial cell beyond length 4 (previously 4/30, `0090` online_4000 seed_154 at length 8; the hexagon stages reached 23/30 D at length 5 on seed_157 in `0083` online_4000). No cold-reset trial succeeded; nothing here is primitive-eligible; the circular object is a disclosed deviation from the paper's hexagonal prism. Continuations: `0097`, `0098` and `0099` continue from the sibling `0096` online_4000, none from this stage.

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
