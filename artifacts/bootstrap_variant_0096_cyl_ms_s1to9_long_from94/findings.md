# Variant 96 `cyl_ms_s1to9_long_from94`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_set": [8], "demonstration_include_mpc": false, "actor_bc_lambda": 100.0, "online_policy_interactions": 4000, "checkpoints": [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], "initial_from": ["bootstrap_variant_0094_cyl_ms_s1to6_all_from89", "online_4000"], "actor_lr_override": 0.0001, "eval_stochastic_trials": 3, "actor_bc_std_target": 0.01, "prefix_sources": ["cylinder_demonstrations_0001", 120], "suffix_set": [1, 2, 3, 4, 5, 6, 7, 8, 9], "demonstration_suffix_rows": 9, "suffix_weights": [1, 1, 1, 2, 2, 2, 2, 2, 2], "holdout_source_names": ["seed_153", "seed_154", "seed_155"], "eval_source_names": ["seed_172", "seed_116", "seed_174"], "env_overrides": {"object_shape": "circular"}, "eval_recorded_prefixes": false, "eval_cold_every_point": true}`. Inherited checkpoint: `bootstrap_variant_0094_cyl_ms_s1to6_all_from89/checkpoints/online_4000`. Hypothesis: Long-suffix emphasis from 0094 online_4000: lengths 1-9 weighted 1:1:1:2:2:2:2:2:2 over all 54 sources, so two thirds of the episodes hand off at least four controls before the end and, for the 8-9-control sources (seeds 172, 116, 174), a suffix of 8-9 is the cold reset itself; evaluated on those three shortest sources plus the held-out trio, with the cold reset at every checkpoint. Tests whether the frontier can be pushed through the short sources, whose intolerant prefix is three to six decisions.

Stage status `complete_sac_stage` (exit=0 (inferred from summary.json and all checkpoints being written; the launcher did not capture the exit status), 1105 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 7500, 'online_updates': 42250}; 500 critic-pretraining and 4000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: overridden to 0.0001 for this stage (inherited 0.0001; ramp fraction unchanged). Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over all demonstration rows, added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.01)) added to the same actor step. Handoff noise: off.

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'s1': 24, 's2': 14, 's3': 18, 's4': 34, 's5': 37, 's6': 28, 's7': 29, 's8': 37, 's9': 35}, force-qualified successes {'s1': 24, 's2': 8, 's3': 12, 's4': 12, 's5': 3, 's6': 3, 's7': 1, 's8': 0, 's9': 0} (informational; the gate is the 458-row demonstration buffer).
- Online: 4000 interactions; completed episodes by prefix {'s1': 62, 's2': 58, 's3': 42, 's4': 108, 's5': 97, 's6': 98, 's7': 107, 's8': 115, 's9': 128}, successes {'s1': 55, 's2': 37, 's3': 20, 's4': 22, 's5': 11, 's6': 5, 's7': 3, 's8': 6, 's9': 2}; outcomes {'drop': 429, 'palm_support': 211, 'success': 161, 'truncated': 14}; handoffs terminated before the policy 0; unfinished last suffix True. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 10678; none entered replay.
- Actor BC loss first/last: 1.001 / 0.0326; spread-anchor loss first/last: 0.0007423 / 1.77e-05; alpha raw-equivalent last 3.6662.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 2.3932, std 0.0110, bias -4.18/-4.20; after_pretraining: L2 2.3932, std 0.0110, bias -2.19/-2.23; online_500: L2 0.7753, std 0.0100, bias -5.73/-5.71; online_1000: L2 0.6842, std 0.0100, bias -5.30/-5.46; online_1500: L2 0.6613, std 0.0100, bias -4.49/-4.44; online_2000: L2 0.6337, std 0.0100, bias -4.30/-4.12; online_2500: L2 0.6495, std 0.0100, bias -4.29/-4.29; online_3000: L2 0.6421, std 0.0100, bias -4.36/-4.38; online_3500: L2 0.6733, std 0.0100, bias -4.91/-4.96; online_4000: L2 0.6286, std 0.0100, bias -5.15/-5.09.

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
| pretraining_baseline | F0/5 | P3/3 P3/3 P3/3 P1/3 P1/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P2/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_500 | F0/5 | P3/3 P3/3 P2/3 P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_1000 | F0/5 | P3/3 P3/3 P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_1500 | F0/5 | P3/3 P3/3 P3/3 F2/3 F1/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P2/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_2000 | F0/5 | P3/3 P3/3 P3/3 F0/3 P0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P3/3 F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P2/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_2500 | F0/5 | P3/3 P3/3 P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F1/3 P2/3 P1/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_3000 | F0/5 | P3/3 P3/3 P3/3 F1/3 F2/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_3500 | F0/5 | P3/3 P3/3 P3/3 F0/3 F1/3 F1/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P0/3 F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_4000 | F0/5 | P3/3 P3/3 P3/3 F2/3 P0/3 P1/3 F0/3 F0/3 F0/3 | P3/3 P2/3 P1/3 F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 P1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

The 30-trial results are in the campaign section below.

## Cylinder campaign, lengths 1-9 weighted 1:1:1:2:2:2:2:2:2 over the 51 training demonstrations (54 verified, 3 held out; `0096`, from `0094` online_4000, 4,000 updates): 30-trial frozen evaluation

Environment: the archived task with `object_shape: circular` (40 mm; `geometry_transfer.json` records the override). Evaluated sources: the three shortest demonstrations, seed_172 (8 controls), seed_116 (9) and seed_174 (9), all training sources, and the held-out seeds 153, 154, 155 (13, 15 and 10 controls); the parent `0094` has 30-trial cells only on seeds 101, 102 and the held-out trio, so the short sources are compared with the sibling `0095` (seed_172 only). 30 fresh-seed stochastic trials plus one deterministic trial per cell; a suffix at least as long as the source is the cold reset itself (marked "cold"; the evaluator clamps such cells to prefix 0 as the in-run evaluation does), so the cold reset appears both as the shared cell k0 (`results_shared/`) and as those cells.

**online_3500** (cold reset 0/30 (deterministic fail))

| source | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed_172 | 30/30 D | 30/30 D | 30/30 D | 7/30 | 6/30 | 2/30 | 0/30 | 0/30 (cold) | 0/30 (cold) |
| seed_116 | 30/30 D | 30/30 D | 15/30 D | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 (cold) |
| seed_174 | 30/30 D | 0/30 | 10/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 (cold) |
| seed_153 (held out) | 2/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_154 (held out) | 0/30 | 1/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_155 (held out) | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| training sources | 90/90 (3 D) | 60/90 (2 D) | 55/90 (2 D) | 7/90 (0 D) | 6/90 (0 D) | 2/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) |
| held-out sources | 2/90 (0 D) | 1/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) |
| all 6 | 92/180 | 61/180 | 55/180 | 7/180 | 6/180 | 2/180 | 0/180 | 0/180 | 0/180 |

**online_4000** (cold reset 0/30 (deterministic fail))

| source | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed_172 | 30/30 D | 30/30 D | 30/30 D | 12/30 | 7/30 D | 9/30 D | 0/30 | 0/30 (cold) | 0/30 (cold) |
| seed_116 | 30/30 D | 25/30 D | 21/30 D | 2/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 (cold) |
| seed_174 | 30/30 D | 0/30 | 11/30 D | 3/30 | 1/30 | 0/30 | 0/30 | 0/30 | 0/30 (cold) |
| seed_153 (held out) | 9/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_154 (held out) | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_155 (held out) | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| training sources | 90/90 (3 D) | 55/90 (2 D) | 62/90 (3 D) | 17/90 (0 D) | 8/90 (1 D) | 9/90 (1 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) |
| held-out sources | 9/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) |
| all 6 | 99/180 | 55/180 | 62/180 | 17/180 | 8/180 | 9/180 | 0/180 | 0/180 | 0/180 |

**Reading.** On seed_172 the policy completes the last three controls in 30/30 D at both checkpoints; the last four rise from 7/30 to 12/30 (no deterministic pass; `0095` online_4000 had 20/30 D), the last five are 6/30 then 7/30 D, and the last six 2/30 then 9/30 D: the length-6 cell at online_4000 is the cylinder campaign's highest 30-trial cell beyond length 5 (previous maximum 4/30: `0095` online_3500 seed_172 at length 7 and `0090` online_4000 seed_154 at length 8) and carries a deterministic pass, i.e. the policy completes the turn from the state after seed_172's second recorded control; the in-run cell at online_4000 (P1/3) agrees. Length 7 (after one recorded control) is 0/30 at both checkpoints and the cold reset 0/30 with a deterministic fail; the 62 cold trials dropped the object after 3-5 policy controls (3: 32, 4: 27, 5: 3) with force-qualified angles between -4.2 and 2.0 rad. On seed_116 the last two controls are 30/30 D then 25/30 D and the last three 15/30 D then 21/30 D, with 0-2 of 30 beyond; on seed_174 the last control is 30/30 D, the last two 0/30 at both checkpoints, the last three 10/30 then 11/30 D and lengths 4-5 at online_4000 3/30 and 1/30: on this source the policy completes the last control and the last three but not the last two, as `0095`'s held-out seed_154 completes the last two and not the last one (both are handoff states, not lengths, that the policy has or has not learned). Held out: the last control is 2/90 then 9/90 (seed_153 alone; `0094` online_4000 had 60/90 and `0095` 30/90), length 2 is 1/90 then 0/90 (seed_154 alone) and lengths 3-9 are 0/90 at both checkpoints: weighting four fifths of the stream (12 of 15 weight units; 653 of 815 online episodes) toward suffixes of four to nine controls lost the held-out last control that the lengths-1-6 parent had. Online stream by length 1-9: 55/62, 37/58, 20/42, 22/108, 11/97, 5/98, 3/107, 6/115, 2/128 (changing-policy data); the six successes at length 8 and two at length 9 are assisted handoffs on long sources (seeds 179, 188, 186, 152; 3-25 replayed controls), and all 13 online episodes that started from the cold reset of the 8-9-control sources ended in a drop (`summary.json`: online_cold_force_successes 0). Against the hypothesis (pushing the frontier through the short sources): on seed_172 it reached deterministic passes at lengths 5 and 6 with 7/30 and 9/30 stochastic (the sibling `0095` online_3500 had deterministic passes at lengths 4-6 with 14/30, 9/30 and 0/30; what is new is the stochastic count at length 6), it did not move on seeds 116 and 174 (lengths 4-9 at 0-3 of 30), and the state after one recorded control (evaluated on seeds 172, 116, 174 and 155) and the cold reset remain at 0/30. No cold-reset trial succeeded; nothing here is primitive-eligible; the circular object is a disclosed deviation from the paper's hexagonal prism. Continuations from this checkpoint: `0097` (same recipe, 4,000 more updates), `0098` (sources of at most 12 controls, lengths 1-10) and `0099` (seed_172 as a spine with shared cold-start prefixes). `cylinder_first_control_0001` measures why the last one to three controls before the cold reset resist: the first control of each of the eleven demonstrations probed tolerates open-loop perturbations of at most 1e-5 to 1e-6, while the re-planning MPC tolerates 0.03-0.10 of execution noise.

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
