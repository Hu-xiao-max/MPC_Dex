# Variant 89 `cyl_ms_from82`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_set": [8], "demonstration_include_mpc": false, "actor_bc_lambda": 100.0, "online_policy_interactions": 4000, "checkpoints": [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], "initial_from": ["bootstrap_variant_0082_ms_shared_prefix_from81", "online_4000"], "actor_lr_override": 0.0001, "eval_stochastic_trials": 3, "actor_bc_std_target": 0.01, "prefix_sources": ["cylinder_demonstrations_0001", 120], "suffix_set": [1, 2, 3, 4, 5, 6, 7, 8], "demonstration_suffix_rows": 8, "eval_source_count": 2, "holdout_source_count": 3, "suffix_weights": [2, 2, 2, 2, 1, 1, 1, 1], "env_overrides": {"object_shape": "circular"}, "eval_recorded_prefixes": false, "eval_cold_every_point": true}`. Inherited checkpoint: `bootstrap_variant_0082_ms_shared_prefix_from81/checkpoints/online_4000`. Hypothesis: Route (b), object geometry: the archived environment with object_shape circular (40 mm, the archived diameter; grasp, mass, friction, gains, period unchanged). The hexagon demonstrations are open-loop intolerant over five to nine controls per turn because its edges rolling over fingertips are bifurcation events; cylinder demonstrations carry three to six such decisions (object_shape_0001). The multi-source weighted suffix curriculum that reached 85/59/28 % for the last 1/2/3 controls on the hexagon (0079-0082) is run on verified cylinder demonstrations (archived collector contract), inheriting the best hexagon multi-source checkpoint (0082 online_4000) so that the end-of-turn competence transfers; three sources held out; cold reset evaluated at every checkpoint. Success criterion for the campaign: a frozen checkpoint completing the turn from the cold reset.

Stage status `complete_sac_stage` (exit=0 (inferred from summary.json and all eight checkpoints being written; the launcher did not capture the exit status), 2152 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 6500, 'online_updates': 35750}; 500 critic-pretraining and 4000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: overridden to 0.0001 for this stage (inherited 0.0001; ramp fraction unchanged). Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over all demonstration rows, added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.01)) added to the same actor step. Handoff noise: off.

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'s1': 48, 's2': 42, 's3': 45, 's4': 38, 's5': 13, 's6': 24, 's7': 20, 's8': 26}, force-qualified successes {'s1': 12, 's2': 8, 's3': 1, 's4': 0, 's5': 0, 's6': 0, 's7': 0, 's8': 0} (informational; the gate is the 168-row demonstration buffer).
- Online: 4000 interactions; completed episodes by prefix {'s1': 162, 's2': 138, 's3': 145, 's4': 148, 's5': 78, 's6': 74, 's7': 87, 's8': 93}, successes {'s1': 143, 's2': 93, 's3': 59, 's4': 32, 's5': 10, 's6': 2, 's7': 5, 's8': 1}; outcomes {'drop': 368, 'palm_support': 203, 'success': 345, 'truncated': 9}; handoffs terminated before the policy 0; unfinished last suffix False. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 12794; none entered replay.
- Actor BC loss first/last: 4.218 / 0.01001; spread-anchor loss first/last: 0.001118 / 2.95e-06; alpha raw-equivalent last 1.9171.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 2.5301, std 0.0103, bias -3.06/-2.81; after_pretraining: L2 2.5301, std 0.0103, bias -0.63/-0.65; online_500: L2 1.3958, std 0.0100, bias -2.47/-2.44; online_1000: L2 1.3690, std 0.0100, bias -2.37/-2.30; online_1500: L2 1.3541, std 0.0100, bias -2.20/-2.15; online_2000: L2 1.3745, std 0.0100, bias -2.37/-2.38; online_2500: L2 1.3699, std 0.0100, bias -2.59/-2.59; online_3000: L2 1.3499, std 0.0100, bias -2.86/-2.82; online_3500: L2 1.3448, std 0.0100, bias -2.90/-2.94; online_4000: L2 1.3401, std 0.0100, bias -3.26/-3.22.

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
| pretraining_baseline | F0/5 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_500 | F0/5 | P1/3 F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F2/3 P1/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F2/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_1000 | F0/5 | F0/3 P3/3 P2/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 F3/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_1500 | F0/5 | F0/3 P1/3 P0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P2/3 P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_2000 | F0/5 | F0/3 P3/3 P0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F2/3 F1/3 F1/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 P1/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_2500 | F0/5 | F0/3 P3/3 P1/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P2/3 F2/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_3000 | F0/5 | F0/3 P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F1/3 P2/3 F2/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 P0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_3500 | F0/5 | F0/3 F1/3 F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P3/3 F1/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_4000 | F0/5 | P1/3 F1/3 F0/3 P0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P1/3 P3/3 F1/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

The 30-trial results are in the campaign section below.

## Cylinder campaign: 30-trial frozen evaluation (`0089`, lengths 1-8 from the hexagon checkpoint `0082` online_4000; 24 sources loaded, 21 training and 3 held out)

Environment: the archived task with `object_shape: circular` (40 mm; `geometry_transfer.json` records the override under which the unchanged inherited checkpoint is trained and evaluated). Evaluated sources: seeds 101 and 102 (training) and 153, 154, 155 (held out; 13, 15 and 10 controls). 30 fresh-seed stochastic trials plus one deterministic trial per cell (`frozen_checkpoint_evaluation_0001/results_multisource/<stage>/<checkpoint>.json`; cold reset in `results_shared/`).

**online_2500** (cold reset 0/30 (deterministic fail))

| source | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed_101 | 0/30 | 27/30 D | 7/30 D | 4/30 | 1/30 | 0/30 | 0/30 | 0/30 |
| seed_102 | 30/30 D | 20/30 D | 25/30 D | 11/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_153 (held out) | 30/30 D | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_154 (held out) | 30/30 D | 1/30 | 3/30 | 0/30 | 1/30 | 0/30 | 0/30 | 0/30 |
| seed_155 (held out) | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| training sources | 30/60 (1 D) | 47/60 (2 D) | 32/60 (2 D) | 15/60 (0 D) | 1/60 (0 D) | 0/60 (0 D) | 0/60 (0 D) | 0/60 (0 D) |
| held-out sources | 60/90 (2 D) | 1/90 (0 D) | 3/90 (0 D) | 0/90 (0 D) | 1/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) |
| all five | 90/150 | 48/150 | 35/150 | 15/150 | 2/150 | 0/150 | 0/150 | 0/150 |

**online_4000** (cold reset 0/30 (deterministic fail))

| source | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed_101 | 11/30 D | 0/30 | 17/30 | 3/30 D | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_102 | 30/30 D | 17/30 D | 22/30 D | 5/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_153 (held out) | 30/30 D | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_154 (held out) | 0/30 | 4/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_155 (held out) | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| training sources | 41/60 (2 D) | 17/60 (1 D) | 39/60 (1 D) | 8/60 (1 D) | 0/60 (0 D) | 0/60 (0 D) | 0/60 (0 D) | 0/60 (0 D) |
| held-out sources | 30/90 (1 D) | 4/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) |
| all five | 71/150 | 21/150 | 39/150 | 8/150 | 0/150 | 0/150 | 0/150 | 0/150 |

**Reading.** The transferred hexagon policy completed the cylinder's last control in 12 of 48 warm-up episodes before any update. At online_2500 the last control is completed in 90/150 trials over the five sources: training 30/60 (seed_101 0/30, seed_102 30/30 D) and held-out 60/90 (seeds 153 and 154 at 30/30 D, seed_155 0/30), i.e. an all-or-nothing outcome per source; the hexagon stages' held-out sources were 0 or 30 of 30 per source in `0074` online_4000 (60/60 on seeds 157-158) and `0079` online_4000 (0/90 on seeds 194-196), while `0081` online_4000 had 27/30 D on seed_197. Lengths 2-4 reach 47/60, 32/60, 15/60 on the training sources and 1/90, 3/90, 0/90 held out; lengths 5-8 are 2/150, 0/150, 0/150, 0/150 over all five. Online_4000 against online_2500 over the five sources: lower at lengths 1, 2, 4 and 5 (71/150 vs 90/150; 21/150 vs 48/150; 8/150 vs 15/150; 0/150 vs 2/150), higher at length 3 (39/150 vs 35/150), 0 at lengths 6-8 for both; its held-out last control is 30/90 (seed_153 alone; seed_154 is 0/30 at online_4000 after 30/30 D at online_2500; the in-run three-trial cells show the drop from online_3000: P3/3 at online_2500, F0/3 at online_3000, online_3500 and online_4000) while seed_101 rose to 11/30 D. The hexagon's direct jump to lengths 1-8 (`0076` online_4000: 180, 150, 58 of 240 at lengths 1-3 against the inherited `0074`'s 240, 168, 134) also ended below its start. The staged stage `0091` (lengths 1-3 from the same hexagon checkpoint, 2,000 updates) reached 80/150, 31/150, 31/150 at online_2000 (held-out 41/90, 0/90, 0/90), above this stage's online_4000 at lengths 1-2 and below its online_2500 at every length. No cold-reset trial succeeded; nothing here is primitive-eligible; the circular object is a disclosed deviation from the paper's hexagonal object, and the training-stream counts are changing-policy, assisted, demonstration-mixed episodes, not frozen-policy evidence.

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
