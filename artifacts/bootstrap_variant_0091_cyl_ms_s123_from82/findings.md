# Variant 91 `cyl_ms_s123_from82`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_set": [8], "demonstration_include_mpc": false, "actor_bc_lambda": 100.0, "checkpoints": [500, 1000, 1500, 2000], "initial_from": ["bootstrap_variant_0082_ms_shared_prefix_from81", "online_4000"], "actor_lr_override": 0.0001, "eval_stochastic_trials": 3, "actor_bc_std_target": 0.01, "prefix_sources": ["cylinder_demonstrations_0001", 24], "suffix_set": [1, 2, 3], "demonstration_suffix_rows": 3, "holdout_source_names": ["seed_153", "seed_154", "seed_155"], "eval_source_names": ["seed_101", "seed_102"], "env_overrides": {"object_shape": "circular"}, "eval_recorded_prefixes": false, "eval_cold_every_point": true}`. Inherited checkpoint: `bootstrap_variant_0082_ms_shared_prefix_from81/checkpoints/online_4000`. Hypothesis: Staged start of the cylinder curriculum: on the hexagon the multi-source route only progressed when the suffix set grew gradually (0068: lengths 1-3 over 15 sources reached 10/15, 10/15, 9/15 by 1,000 updates; the direct 1-8 stages regressed), whereas 0089 jumped to lengths 1-8 on the cylinder from the transferred checkpoint. Same 24-source snapshot and held-out sources as 0089/0090 (sorted names, first 24), lengths 1-3 uniform, three demonstration rows per source, 2,000 updates, inheriting the hexagon checkpoint 0082 online_4000; the follow-up extends to 1-6 and 1-8 from its best checkpoint as 0070/0074 did.

Stage status `complete_sac_stage` (exit=0 (inferred from summary.json and all checkpoints being written; the launcher did not capture the exit status), 720 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 6500, 'online_updates': 33750}; 500 critic-pretraining and 2000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: overridden to 0.0001 for this stage (inherited 0.0001; ramp fraction unchanged). Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over all demonstration rows, added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.01)) added to the same actor step. Handoff noise: off.

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'s1': 99, 's2': 72, 's3': 85}, force-qualified successes {'s1': 33, 's2': 9, 's3': 4} (informational; the gate is the 63-row demonstration buffer).
- Online: 2000 interactions; completed episodes by prefix {'s1': 250, 's2': 238, 's3': 247}, successes {'s1': 226, 's2': 148, 's3': 114}; outcomes {'drop': 153, 'palm_support': 90, 'success': 488, 'truncated': 4}; handoffs terminated before the policy 0; unfinished last suffix True. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 12609; none entered replay.
- Actor BC loss first/last: 3.927 / 0.0002141; spread-anchor loss first/last: 0.001013 / 4.652e-07; alpha raw-equivalent last 1.5716.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 2.1988, std 0.0102, bias -2.49/-2.20; after_pretraining: L2 2.1988, std 0.0102, bias +0.01/-0.02; online_500: L2 0.0328, std 0.0100, bias -0.46/-0.52; online_1000: L2 0.0262, std 0.0100, bias -1.18/-1.39; online_1500: L2 0.0222, std 0.0100, bias -1.33/-1.30; online_2000: L2 0.0234, std 0.0100, bias -1.22/-1.26.

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
| pretraining_baseline | F0/5 | P3/3 F0/3 F0/3 | P3/3 F0/3 F0/3 | F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 |
| online_500 | F0/5 | F0/3 P2/3 F0/3 | P3/3 P3/3 P2/3 | F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 | F0/3 F0/3 F0/3 |
| online_1000 | F0/5 | F1/3 F1/3 F1/3 | P3/3 P3/3 P3/3 | F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 | F0/3 F0/3 F0/3 |
| online_1500 | F0/5 | P3/3 P1/3 F0/3 | P3/3 P3/3 P2/3 | F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 | F0/3 F0/3 F0/3 |
| online_2000 | F0/5 | F1/3 F1/3 F1/3 | P3/3 P2/3 P2/3 | P1/3 F0/3 F0/3 | P3/3 F0/3 F0/3 | F0/3 F0/3 F0/3 |

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

The 30-trial results are in the campaign section below.

## Cylinder campaign, staged start (`0091`, lengths 1-3 from the hexagon checkpoint `0082` online_4000, 2,000 updates): 30-trial frozen evaluation

Environment: the archived task with `object_shape: circular` (40 mm; `geometry_transfer.json` records the override under which the unchanged inherited checkpoint is trained and evaluated). Evaluated sources: seeds 101 and 102 (training) and 153, 154, 155 (held out; 13, 15 and 10 controls). 30 fresh-seed stochastic trials plus one deterministic trial per cell (`frozen_checkpoint_evaluation_0001/results_multisource/<stage>/<checkpoint>.json`; cold reset in `results_shared/`).

**online_1000** (cold reset: no 30-trial evaluation at this checkpoint; the in-run cold trials were deterministic fail and 0/5)

| source | s1 | s2 | s3 |
| --- | --- | --- | --- |
| seed_101 | 6/30 | 2/30 | 4/30 |
| seed_102 | 30/30 D | 28/30 D | 27/30 D |
| seed_153 (held out) | 0/30 | 0/30 | 0/30 |
| seed_154 (held out) | 30/30 D | 0/30 | 0/30 |
| seed_155 (held out) | 0/30 | 0/30 | 0/30 |
| training sources | 36/60 (1 D) | 30/60 (1 D) | 31/60 (1 D) |
| held-out sources | 30/90 (1 D) | 0/90 (0 D) | 0/90 (0 D) |
| all five | 66/150 | 30/150 | 31/150 |

**online_2000** (cold reset 0/30 (deterministic fail))

| source | s1 | s2 | s3 |
| --- | --- | --- | --- |
| seed_101 | 9/30 | 7/30 | 3/30 |
| seed_102 | 30/30 D | 24/30 D | 28/30 D |
| seed_153 (held out) | 11/30 D | 0/30 | 0/30 |
| seed_154 (held out) | 30/30 D | 0/30 | 0/30 |
| seed_155 (held out) | 0/30 | 0/30 | 0/30 |
| training sources | 39/60 (1 D) | 31/60 (1 D) | 31/60 (1 D) |
| held-out sources | 41/90 (2 D) | 0/90 (0 D) | 0/90 (0 D) |
| all five | 80/150 | 31/150 | 31/150 |

**Reading.** The hexagon's lengths-1-3 stage `0068` (15 sources, from `0061`) had an online stream of 112/136, 50/131, 41/130 at lengths 1-3 and in-run cells of 10/15, 10/15, 9/15 over three evaluated sources (five stochastic seeds each) at online_1000; this stage's stream is stronger (226/250, 148/238, 114/247; changing-policy data). Frozen, its online_2000 checkpoint reaches 80/150, 31/150, 31/150 over the five sources (training 39/60, 31/60, 31/60; held-out 41/90, 0/90, 0/90): above `0089`'s final checkpoint at lengths 1-2 (71, 21 of 150) and below `0089`'s reference checkpoint (online_2500, 2,500 updates from the same hexagon checkpoint, the higher of its two frozen-evaluated checkpoints at lengths 1-2 and on the held-out sources but not at length 3, where online_4000 has 39/150: 90, 48, 35; held-out 60, 1, 3 of 90) at every length, with no 30-trial evaluation of the update-matched `0089` online_2000 checkpoint. The gradual suffix set therefore gave no clear advantage over the direct jump on the cylinder. The comparison with `0092` (lengths 1-3 continued from `0089` online_2500) followed: `0092` online_2000 reached 51, 40, 45 of 150 (held-out 9, 0, 0 of 90), and neither candidate exceeded `0089` online_2500 at the last control or on the held-out sources, so the extension stages `0093`/`0094` started from `0089` online_2500 (see `0092`'s findings for the departure from its pre-registered rule). No cold-reset trial succeeded; nothing here is primitive-eligible; the circular object is a disclosed deviation from the paper's hexagonal object, and the training-stream counts are changing-policy, assisted, demonstration-mixed episodes, not frozen-policy evidence.

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
