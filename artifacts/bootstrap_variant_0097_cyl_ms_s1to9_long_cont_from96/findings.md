# Variant 97 `cyl_ms_s1to9_long_cont_from96`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_set": [8], "demonstration_include_mpc": false, "actor_bc_lambda": 100.0, "online_policy_interactions": 4000, "checkpoints": [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], "initial_from": ["bootstrap_variant_0096_cyl_ms_s1to9_long_from94", "online_4000"], "actor_lr_override": 0.0001, "eval_stochastic_trials": 3, "actor_bc_std_target": 0.01, "prefix_sources": ["cylinder_demonstrations_0001", 120], "suffix_set": [1, 2, 3, 4, 5, 6, 7, 8, 9], "demonstration_suffix_rows": 9, "suffix_weights": [1, 1, 1, 2, 2, 2, 2, 2, 2], "holdout_source_names": ["seed_153", "seed_154", "seed_155"], "eval_source_names": ["seed_172", "seed_116", "seed_174"], "env_overrides": {"object_shape": "circular"}, "eval_recorded_prefixes": false, "eval_cold_every_point": true}`. Inherited checkpoint: `bootstrap_variant_0096_cyl_ms_s1to9_long_from94/checkpoints/online_4000`. Hypothesis: Continuation of 0096 (lengths 1-9 weighted toward long suffixes over all 54 sources) for 4,000 more updates from its online_4000, whose in-run cells reached deterministic passes at lengths 5-6 on the 8-control source seed_172 (its cold reset is length 8) and 1-3 on seed_116; same evaluated and held-out sources.

Stage status `complete_sac_stage` (?, 1341 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 8000, 'online_updates': 46250}; 500 critic-pretraining and 4000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: overridden to 0.0001 for this stage (inherited 0.0001; ramp fraction unchanged). Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over all demonstration rows, added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.01)) added to the same actor step. Handoff noise: off.

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'s1': 24, 's2': 14, 's3': 18, 's4': 34, 's5': 37, 's6': 28, 's7': 29, 's8': 37, 's9': 35}, force-qualified successes {'s1': 23, 's2': 5, 's3': 11, 's4': 9, 's5': 2, 's6': 2, 's7': 1, 's8': 1, 's9': 1} (informational; the gate is the 458-row demonstration buffer).
- Online: 4000 interactions; completed episodes by prefix {'s1': 63, 's2': 68, 's3': 47, 's4': 113, 's5': 108, 's6': 108, 's7': 120, 's8': 123, 's9': 146}, successes {'s1': 59, 's2': 43, 's3': 17, 's4': 20, 's5': 13, 's6': 5, 's7': 3, 's8': 3, 's9': 0}; outcomes {'drop': 470, 'palm_support': 250, 'truncated': 13, 'success': 163}; handoffs terminated before the policy 0; unfinished last suffix False. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 11461; none entered replay.
- Actor BC loss first/last: 0.0326 / 0.03362; spread-anchor loss first/last: 1.604e-05 / 2.679e-05; alpha raw-equivalent last 5.4638.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 0.6286, std 0.0100, bias -5.15/-5.09; after_pretraining: L2 0.6286, std 0.0100, bias -3.06/-2.97; online_500: L2 0.7044, std 0.0100, bias -5.36/-5.36; online_1000: L2 0.7262, std 0.0100, bias -5.83/-6.07; online_1500: L2 0.6576, std 0.0100, bias -6.07/-6.26; online_2000: L2 0.7319, std 0.0100, bias -6.62/-6.59; online_2500: L2 0.6536, std 0.0101, bias -6.02/-6.20; online_3000: L2 0.6413, std 0.0100, bias -6.78/-6.66; online_3500: L2 0.6251, std 0.0100, bias -6.81/-6.90; online_4000: L2 0.6315, std 0.0100, bias -7.21/-7.17.

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

| Checkpoint | Cold | seed_172 s1..s9 | seed_116 s1..s9 | seed_174 s1..s9 | seed_153 s1..s9 | seed_154 s1..s9 | seed_155 s1..s9 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pretraining_baseline | F0/5 | P3/3 P3/3 P3/3 F2/3 P0/3 P1/3 F0/3 F0/3 F0/3 | P3/3 P2/3 P2/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 P0/3 F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_500 | F0/5 | P3/3 P3/3 P3/3 F0/3 P2/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P1/3 F2/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_1000 | F0/5 | P3/3 P3/3 P1/3 F1/3 F1/3 P1/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P2/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_1500 | F0/5 | P3/3 P3/3 P3/3 F1/3 F1/3 P1/3 F0/3 F0/3 F0/3 | P3/3 P2/3 P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F2/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_2000 | F0/5 | P3/3 P3/3 P3/3 F0/3 F0/3 F1/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P2/3 P0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 F2/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_2500 | F0/5 | P3/3 P3/3 F1/3 F0/3 P0/3 F1/3 F0/3 F0/3 F0/3 | P3/3 P2/3 P1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_3000 | F0/5 | P3/3 P3/3 P3/3 F1/3 F1/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 F2/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_3500 | F0/5 | P3/3 P3/3 F2/3 F0/3 F0/3 F1/3 F0/3 F0/3 F0/3 | P3/3 P2/3 F1/3 F2/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_4000 | F0/5 | P3/3 P3/3 F2/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P2/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

The 30-trial results are in the campaign section below.

## Cylinder campaign, lengths 1-9 weighted 1:1:1:2:2:2:2:2:2 continued (`0097`, from `0096` online_4000, 4,000 more updates): 30-trial frozen evaluation

Environment: the archived task with `object_shape: circular` (40 mm). Evaluated sources as in `0096`: seed_172 (8 controls), seed_116 (9), seed_174 (9), all training, and the held-out seeds 153, 154, 155. 30 fresh-seed stochastic trials plus one deterministic trial per cell; a suffix at least as long as the source is the cold reset itself (marked "cold"). Start checkpoint `0096` online_4000: seed_172 30/30 D, 30/30 D, 30/30 D, 12/30, 7/30 D, 9/30 D, 0/30 at lengths 1-7; seed_116 30/30 D, 25/30 D, 21/30 D, 2/30, 0/30 beyond; seed_174 30/30 D, 0/30, 11/30 D, 3/30, 1/30, 0/30 beyond; held-out last control 9/90.

**online_3000** (cold reset 0/30 (deterministic fail))

| source | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed_172 | 30/30 D | 30/30 D | 25/30 D | 10/30 | 11/30 | 2/30 | 0/30 | 0/30 (cold) | 0/30 (cold) |
| seed_116 | 30/30 D | 28/30 D | 17/30 | 6/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 (cold) |
| seed_174 | 30/30 D | 30/30 D | 15/30 D | 0/30 | 0/30 | 1/30 | 0/30 | 0/30 | 0/30 (cold) |
| seed_153 (held out) | 1/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_154 (held out) | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_155 (held out) | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| training sources | 90/90 (3 D) | 88/90 (3 D) | 57/90 (2 D) | 16/90 (0 D) | 11/90 (0 D) | 3/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) |
| held-out sources | 1/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) |
| all 6 | 91/180 | 88/180 | 57/180 | 16/180 | 11/180 | 3/180 | 0/180 | 0/180 | 0/180 |

**online_4000** (cold reset 0/30 (deterministic fail))

| source | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed_172 | 30/30 D | 30/30 D | 25/30 | 4/30 | 6/30 | 1/30 | 0/30 | 0/30 (cold) | 0/30 (cold) |
| seed_116 | 30/30 D | 25/30 D | 23/30 D | 4/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 (cold) |
| seed_174 | 30/30 D | 30/30 D | 11/30 | 1/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 (cold) |
| seed_153 (held out) | 6/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_154 (held out) | 0/30 | 1/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_155 (held out) | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| training sources | 90/90 (3 D) | 85/90 (3 D) | 59/90 (1 D) | 9/90 (0 D) | 6/90 (0 D) | 1/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) |
| held-out sources | 6/90 (0 D) | 1/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) |
| all 6 | 96/180 | 86/180 | 59/180 | 9/180 | 6/180 | 1/180 | 0/180 | 0/180 | 0/180 |

**Reading.** Four thousand more updates of the same recipe did not extend the frontier: on seed_172 the last three controls fell from 30/30 D to 25/30 (deterministic pass at online_3000 only), the last four from 12/30 to 10/30 then 4/30, the last five rose from 7/30 D to 11/30 at online_3000 and fell to 6/30 at online_4000, and the last six fell from 9/30 D to 2/30 then 1/30, without a deterministic pass beyond length 3 at either checkpoint; on seed_116 the last three fell to 17/30 without a deterministic pass at online_3000 and rose to 23/30 D at online_4000 (from 21/30 D) and the last four to 4-6 of 30 (from 2/30); on seed_174 the last two rose from 0/30 to 30/30 D at both checkpoints (the state the start checkpoint could not use), the last three were 15/30 D then 11/30 without the deterministic pass (from 11/30 D), length 4 fell from 3/30 to 0/30 then 1/30 and lengths 5-6 stayed at 0-1 of 30. Training-source totals at lengths 4-6: 16, 11, 3 of 90 at online_3000 and 9, 6, 1 at online_4000, against the start's 17, 8, 9. Held out: the last control is 1/90 then 6/90 (seed_153 alone), lengths 2-9 at most 1/90. Cold reset: 0/30 with a deterministic fail at both checkpoints; the 62 trials dropped the object after two or three policy controls (online_3000: 3 controls in 26, 2 in 5; online_4000: 2 in 20, 3 in 11), force-qualified angle -5.0 to 2.2 rad. Online stream by length 1-9: 59/63, 43/68, 17/47, 20/113, 13/108, 5/108, 3/120, 3/123, 0/146 (changing-policy data), the same profile as the parent's. The states after seed_172's first and second recorded controls (lengths 7 and 6) are at 0/30 and 1-2 of 30, the cold reset at 0/30: the continuation moved competence between handoff states of the three short sources (seed_174's last two gained, seed_172's last four to six lost) without reaching a state deeper than the start's deepest (seed_172's length 6). No cold-reset trial succeeded; nothing here is primitive-eligible; the circular object is a disclosed deviation from the paper's hexagonal prism. `cylinder_first_control_0001` gives the reason the last controls before the cold reset resist: the policies imitate recorded open-loop trajectories whose first control tolerates 1e-5 to 1e-6 of perturbation and carry no corrective law, while the re-planning MPC tolerates 0.03-0.10 of execution noise.

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
