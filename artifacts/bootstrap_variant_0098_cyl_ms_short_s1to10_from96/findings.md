# Variant 98 `cyl_ms_short_s1to10_from96`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_set": [8], "demonstration_include_mpc": false, "actor_bc_lambda": 100.0, "online_policy_interactions": 4000, "checkpoints": [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], "initial_from": ["bootstrap_variant_0096_cyl_ms_s1to9_long_from94", "online_4000"], "actor_lr_override": 0.0001, "eval_stochastic_trials": 3, "actor_bc_std_target": 0.01, "prefix_sources": ["cylinder_demonstrations_0001", 120], "suffix_set": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "demonstration_suffix_rows": 10, "suffix_weights": [1, 1, 1, 2, 2, 2, 3, 3, 3, 3], "holdout_source_names": ["seed_155"], "eval_source_names": ["seed_172", "seed_116", "seed_174"], "env_overrides": {"object_shape": "circular"}, "eval_recorded_prefixes": false, "eval_cold_every_point": true, "max_source_controls": 12}`. Inherited checkpoint: `bootstrap_variant_0096_cyl_ms_s1to9_long_from94/checkpoints/online_4000`. Hypothesis: Concentration on the short cylinder demonstrations (at most 12 controls: 21 sources, seed_155 held out) from 0096 online_4000, with lengths 1-10 weighted 1:1:1:2:2:2:3:3:3:3 so that most episodes hand off within the first few controls and, for the 8-10-control sources, a suffix of 8-10 is the cold reset; evaluated on the three shortest sources and the held-out seed_155, cold reset at every checkpoint. Tests whether the remaining two to four controls to the cold reset close when the stream is devoted to the trajectories that are closest to it.

Stage status `complete_sac_stage` (?, 832 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 8000, 'online_updates': 46250}; 500 critic-pretraining and 4000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: overridden to 0.0001 for this stage (inherited 0.0001; ramp fraction unchanged). Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over all demonstration rows, added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.01)) added to the same actor step. Handoff noise: off.

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'s1': 14, 's2': 15, 's3': 10, 's4': 30, 's5': 21, 's6': 25, 's7': 35, 's8': 28, 's9': 38, 's10': 40}, force-qualified successes {'s1': 14, 's2': 11, 's3': 4, 's4': 3, 's5': 0, 's6': 0, 's7': 0, 's8': 0, 's9': 0, 's10': 0} (informational; the gate is the 186-row demonstration buffer).
- Online: 4000 interactions; completed episodes by prefix {'s1': 57, 's2': 64, 's3': 59, 's4': 101, 's5': 105, 's6': 98, 's7': 148, 's8': 158, 's9': 158, 's10': 203}, successes {'s1': 54, 's2': 45, 's3': 16, 's4': 4, 's5': 5, 's6': 0, 's7': 0, 's8': 0, 's9': 0, 's10': 0}; outcomes {'drop': 700, 'palm_support': 323, 'success': 124, 'truncated': 4}; handoffs terminated before the policy 0; unfinished last suffix True. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 5806; none entered replay.
- Actor BC loss first/last: 0.1547 / 0.06126; spread-anchor loss first/last: 6.354e-05 / 9.569e-06; alpha raw-equivalent last 5.3885.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 2.0154, std 0.0101, bias -5.14/-5.05; after_pretraining: L2 2.0154, std 0.0101, bias -2.49/-2.66; online_500: L2 1.2733, std 0.0100, bias -4.91/-4.98; online_1000: L2 1.3793, std 0.0100, bias -5.25/-5.36; online_1500: L2 1.3084, std 0.0100, bias -5.66/-5.80; online_2000: L2 1.3339, std 0.0100, bias -6.02/-6.07; online_2500: L2 1.2341, std 0.0100, bias -6.03/-6.00; online_3000: L2 1.2610, std 0.0100, bias -5.95/-6.19; online_3500: L2 1.2665, std 0.0100, bias -6.42/-6.51; online_4000: L2 1.2895, std 0.0100, bias -6.45/-6.41.

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

| Checkpoint | Cold | seed_172 s1..s10 | seed_116 s1..s10 | seed_174 s1..s10 | seed_155 s1..s10 |
| --- | --- | --- | --- | --- | --- |
| pretraining_baseline | F0/5 | P3/3 P3/3 P3/3 F0/3 P0/3 P0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P3/3 F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 P1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_500 | F0/5 | P3/3 P3/3 P3/3 F0/3 F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 F2/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_1000 | F0/5 | P3/3 P3/3 P3/3 P1/3 P0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P2/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F1/3 F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_1500 | F0/5 | P3/3 P3/3 P3/3 P2/3 F0/3 P0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P2/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P3/3 F0/3 F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_2000 | F0/5 | P3/3 P3/3 P3/3 P3/3 P2/3 F1/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 F3/3 P0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F2/3 P0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_2500 | F0/5 | P3/3 P3/3 P3/3 P1/3 F1/3 F1/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P2/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_3000 | F0/5 | P3/3 P3/3 P3/3 F2/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P2/3 P2/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P2/3 F1/3 F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_3500 | F0/5 | P3/3 P3/3 P3/3 P1/3 F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P2/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_4000 | F0/5 | P3/3 P3/3 P3/3 P1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 F3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P2/3 P1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

The 30-trial results are in the campaign section below.

## Cylinder campaign, the demonstrations of at most 12 controls, lengths 1-10 weighted 1:1:1:2:2:2:3:3:3:3 (`0098`, from `0096` online_4000, 4,000 updates): 30-trial frozen evaluation

Environment: the archived task with `object_shape: circular` (40 mm). Sources: the template's `max_source_controls: 12` filter keeps 20 of the 54 demonstrations, 19 as training sources and seed_155 (10 controls) held out (the pre-registered hypothesis text says 21; the stamped protocol has 19 plus 1). Evaluated: seed_172 (8 controls), seed_116 (9), seed_174 (9), all training, and seed_155. 30 fresh-seed stochastic trials plus one deterministic trial per cell; a suffix at least as long as the source is the cold reset itself (marked "cold"). The start checkpoint `0096` online_4000 has 30-trial cells on the same three short sources (seed_172: 30/30 D, 30/30 D, 30/30 D, 12/30, 7/30 D, 9/30 D, 0/30 at lengths 1-7; seed_116: 30/30 D, 25/30 D, 21/30 D, 2/30, 0, 0, 0, 0; seed_174: 30/30 D, 0/30, 11/30 D, 3/30, 1/30, 0, 0, 0).

**online_2000** (cold reset 0/30 (deterministic fail))

| source | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | s10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed_172 | 30/30 D | 30/30 D | 30/30 D | 19/30 D | 5/30 D | 8/30 | 0/30 | 0/30 (cold) | 0/30 (cold) | 0/30 (cold) |
| seed_116 | 30/30 D | 30/30 D | 19/30 | 11/30 D | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 (cold) | 0/30 (cold) |
| seed_174 | 30/30 D | 20/30 | 15/30 D | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 (cold) | 0/30 (cold) |
| seed_155 (held out) | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 (cold) |
| training sources | 90/90 (3 D) | 80/90 (2 D) | 64/90 (2 D) | 30/90 (2 D) | 5/90 (1 D) | 8/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) |
| held-out sources | 0/30 (0 D) | 0/30 (0 D) | 0/30 (0 D) | 0/30 (0 D) | 0/30 (0 D) | 0/30 (0 D) | 0/30 (0 D) | 0/30 (0 D) | 0/30 (0 D) | 0/30 (0 D) |
| all 4 | 90/120 | 80/120 | 64/120 | 30/120 | 5/120 | 8/120 | 0/120 | 0/120 | 0/120 | 0/120 |

**online_4000** (cold reset 0/30 (deterministic fail))

| source | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | s10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed_172 | 30/30 D | 30/30 D | 30/30 D | 4/30 D | 1/30 | 0/30 | 0/30 | 0/30 (cold) | 0/30 (cold) | 0/30 (cold) |
| seed_116 | 30/30 D | 30/30 D | 23/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 (cold) | 0/30 (cold) |
| seed_174 | 30/30 D | 24/30 D | 5/30 D | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 (cold) | 0/30 (cold) |
| seed_155 (held out) | 0/30 | 2/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 (cold) |
| training sources | 90/90 (3 D) | 84/90 (3 D) | 58/90 (2 D) | 4/90 (1 D) | 1/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) |
| held-out sources | 0/30 (0 D) | 2/30 (0 D) | 0/30 (0 D) | 0/30 (0 D) | 0/30 (0 D) | 0/30 (0 D) | 0/30 (0 D) | 0/30 (0 D) | 0/30 (0 D) | 0/30 (0 D) |
| all 4 | 90/120 | 86/120 | 58/120 | 4/120 | 1/120 | 0/120 | 0/120 | 0/120 | 0/120 | 0/120 |

**Reading.** At online_2000 seed_172's last four controls rose to 19/30 D (from 12/30 at the start) while its last five and six fell to 5/30 D and 8/30 (from 7/30 D and 9/30 D); seed_116's last three fell to 19/30 (from 21/30 D) and its last four rose to 11/30 D (from 2/30); seed_174's last two rose from 0/30 to 20/30 and its last three from 11/30 D to 15/30 D. At online_4000 every cell beyond length 3 is at 0-4 of 30 (seed_172 4/30 D at length 4, 1/30 at length 5; seed_116 0/30 from length 4; seed_174 0/30 from length 4): the four cells that were nonzero at both the start and online_2000 (seed_172 lengths 4-6, seed_116 length 4) are below both, seed_174's lengths 4-5 fell from 3/30 and 1/30 at the start to 0/30 at both checkpoints, and the remaining cells were 0/30 throughout, while lengths 1-3 on the training sources stay at 90/90, 84/90 and 58/90. The held-out seed_155 is 0/30 at every length except 2/30 at length 2 at online_4000. Cold reset: 0/30 with a deterministic fail at both checkpoints; all 62 trials of the k0 cells dropped the object at the second policy control (force-qualified angle 0.95-1.70 rad), as did the 496 trials of the eight clamped cells per checkpoint in the tables above. Online stream by length 1-10: 54/57, 45/64, 16/59, 4/101, 5/105, 0/98, 0/148, 0/158, 0/158, 0/203 (changing-policy data): lengths 7-10 carry 12 of the 21 sampling-weight units, and none of the 765 completed episodes at lengths 6-10 succeeded, so the stream at those lengths carried no successful suffix to learn from. Against the hypothesis (closing the last two to four controls by devoting the stream to the shortest trajectories): by deepest deterministic pass the frontier at online_2000 gained one length on seed_116 (3 to 4), lost one on seed_172 (6 to 5) and stayed at 3 on seed_174 (whose last two rose from 0/30 to 20/30 inside its frontier), and it receded at online_4000, and the cold reset and the states after one recorded control stayed at 0/30. No cold-reset trial succeeded; nothing here is primitive-eligible; the circular object is a disclosed deviation from the paper's hexagonal prism. `cylinder_first_control_0001` gives the reason the last controls before the cold reset resist (measured on the `0096` and `0099` actors, not this stage's): the policies imitate recorded open-loop trajectories whose first control tolerates 1e-5 to 1e-6 of perturbation and carry no corrective law, while the re-planning MPC tolerates 0.03-0.10 of execution noise.

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
