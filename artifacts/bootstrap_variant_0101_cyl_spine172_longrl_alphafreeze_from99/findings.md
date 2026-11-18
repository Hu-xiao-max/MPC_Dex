# Variant 101 `cyl_spine172_longrl_alphafreeze_from99`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_set": [8], "demonstration_include_mpc": false, "actor_bc_lambda": 100.0, "actor_bc_rows": "per_item", "online_policy_interactions": 40000, "checkpoints": [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000], "initial_from": ["bootstrap_variant_0099_cyl_spine172_from96", "online_4000"], "actor_lr_override": 0.0001, "eval_stochastic_trials": 3, "alpha_freeze": true, "actor_bc_std_target": 0.01, "prefix_sources": ["cylinder_demonstrations_0001", 120], "suffix_set": [1, 2, 3, 4, 5, 6, 7, 8, 9], "demonstration_suffix_rows": 12, "suffix_weights": [1, 1, 1, 2, 2, 2, 2, 2, 2], "holdout_source_names": ["seed_153", "seed_154", "seed_155"], "eval_source_names": ["seed_172", "seed_116", "seed_174"], "start_prefix_set": [0, 1, 2, 3, 4], "start_prefix_weights": [3, 3, 2, 1, 1], "start_prefix_share": 0.6, "actor_bc_last_rows": 8, "spine_source": "seed_172", "actor_bc_shared_lambda": 50.0, "env_overrides": {"object_shape": "circular"}, "eval_recorded_prefixes": false, "eval_cold_every_point": true}`. Inherited checkpoint: `bootstrap_variant_0099_cyl_spine172_from96/checkpoints/online_4000`. Hypothesis: Replicate of 0100 with the temperature frozen at its inherited value. In 0100 the learned temperature rose from 0.027 to 1.39 over 40,000 interactions because the policy entropy (-82 to -51) never approaches the -16 target under the 0.01 spread anchor; the critic value at the supervised row drifted from -6.6 to -106, the actor BC loss grew 44-fold and the stream reward turned negative after 24,000 interactions, while the cold reset stayed at 0 successes in 2,698 training episodes and 0/5 at every checkpoint. Freezing alpha (template knob alpha_freeze, as in the wave-3b hexagon stages) removes that drift so that the long cold-reset run tests exploration rather than temperature runaway. Everything else as 0100. Success criterion unchanged: a frozen cold-reset turn.

Stage status `complete_sac_stage` (?, 3985 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 8500, 'online_updates': 86250}; 500 critic-pretraining and 40000 online updates as prescribed. Temperature: frozen at the inherited value for this stage; alpha loss logged only, alpha optimizer untouched. Actor step: overridden to 0.0001 for this stage (inherited 0.0001; ramp fraction unchanged). Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over the last 8 rows of each demonstration episode plus its shared cold-start prefix rows; additionally, per distinct shared state, lambda_shared 50.0 * (MSE of the actor mean to the label + MSE of the log-std to log(std target)) summed over the distinct shared states (one term per state, however many sources sit on it), added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.01)) added to the same actor step. Handoff noise: off.

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'s1': 10, 's2': 5, 's3': 5, 's4': 14, 's5': 10, 's6': 18, 's7': 9, 's8': 15, 's9': 16, 'k0': 48, 'k1': 37, 'k2': 40, 'k3': 16, 'k4': 13}, force-qualified successes {'s1': 10, 's2': 4, 's3': 3, 's4': 2, 's5': 1, 's6': 4, 's7': 0, 's8': 1, 's9': 0, 'k0': 0, 'k1': 2, 'k2': 9, 'k3': 2, 'k4': 9} (informational; the gate is the 587-row demonstration buffer).
- Online: 40000 interactions; completed episodes by prefix {'s1': 369, 's2': 357, 's3': 331, 's4': 671, 's5': 674, 's6': 717, 's7': 704, 's8': 690, 's9': 684, 'k0': 2290, 'k1': 2295, 'k2': 1455, 'k3': 808, 'k4': 807}, successes {'s1': 354, 's2': 253, 's3': 132, 's4': 156, 's5': 81, 's6': 49, 's7': 28, 's8': 28, 's9': 0, 'k0': 0, 'k1': 4, 'k2': 732, 'k3': 181, 'k4': 625}; outcomes {'drop': 6154, 'palm_support': 4021, 'success': 2623, 'truncated': 54}; handoffs terminated before the policy 0; unfinished last suffix True. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 64208; none entered replay.
- Actor BC loss first/last: 0.001436 / 0.001317; spread-anchor loss first/last: 1.208e-05 / 1.239e-05; alpha raw-equivalent last 5.4472.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 1.5765, std 0.0100, bias -6.63/-6.75; after_pretraining: L2 1.5765, std 0.0100, bias -3.83/-3.92; online_5000: L2 1.5761, std 0.0100, bias -5.11/-4.89; online_10000: L2 1.5642, std 0.0100, bias -4.46/-4.72; online_15000: L2 1.5919, std 0.0100, bias -5.90/-5.57; online_20000: L2 1.5738, std 0.0100, bias -5.37/-5.48; online_25000: L2 1.5675, std 0.0100, bias -5.45/-5.46; online_30000: L2 1.5878, std 0.0100, bias -5.67/-5.63; online_35000: L2 1.5930, std 0.0100, bias -5.32/-5.27; online_40000: L2 1.5763, std 0.0100, bias -5.08/-5.19.

## Frozen evaluations (in-run; no recorded-prefix evaluations; stochastic seeds per prefix)

| Checkpoint | Cold det / stoch |
| --- | --- |
| pretraining_baseline | fail / 0/5 |
| online_5000 | fail / 0/5 |
| online_10000 | fail / 0/5 |
| online_15000 | fail / 0/5 |
| online_20000 | fail / 0/5 |
| online_25000 | fail / 0/5 |
| online_30000 | fail / 0/5 |
| online_35000 | fail / 0/5 |
| online_40000 | fail / 0/5 |

## Multi-source evaluation (in-run; a cell is the deterministic trial P/F followed by stochastic successes/trials; per source the cells run over the suffix lengths; shared k = the state after k shared cold-start controls)

| Checkpoint | Cold | Shared k1 | Shared k2 | Shared k3 | Shared k4 | seed_172 s1..s9 | seed_116 s1..s9 | seed_174 s1..s9 | seed_153 s1..s9 | seed_154 s1..s9 | seed_155 s1..s9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pretraining_baseline | F0/5 | F0/3 | P1/3 | F2/3 | P3/3 | P3/3 P3/3 P3/3 P2/3 F0/3 P3/3 F0/3 F0/3 F0/3 | P3/3 P3/3 F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F2/3 P0/3 P0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_5000 | F0/5 | F0/3 | F2/3 | P1/3 | P3/3 | P3/3 P3/3 P3/3 P3/3 P0/3 F1/3 F0/3 F0/3 F0/3 | P3/3 P3/3 F2/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F2/3 F2/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_10000 | F0/5 | F0/3 | F1/3 | F1/3 | P3/3 | P3/3 P3/3 P3/3 P1/3 F1/3 F1/3 F0/3 F0/3 F0/3 | P3/3 F2/3 P2/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_15000 | F0/5 | F0/3 | P3/3 | F0/3 | P3/3 | P3/3 P3/3 P3/3 P3/3 F0/3 P2/3 F0/3 F0/3 F0/3 | P3/3 P2/3 P2/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 P2/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_20000 | F0/5 | F0/3 | P3/3 | F1/3 | P3/3 | P3/3 P3/3 P3/3 P3/3 F0/3 P2/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 P1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_25000 | F0/5 | F0/3 | P3/3 | F2/3 | F3/3 | P3/3 P3/3 P3/3 F2/3 F0/3 P3/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P2/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_30000 | F0/5 | F0/3 | P3/3 | P0/3 | P1/3 | P3/3 P3/3 P3/3 P3/3 P0/3 P2/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P2/3 P0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_35000 | F0/5 | F0/3 | P1/3 | F0/3 | P3/3 | P3/3 P3/3 P3/3 P3/3 F0/3 P1/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P2/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 F0/3 F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_40000 | F0/5 | F0/3 | P3/3 | F0/3 | F3/3 | P3/3 P3/3 P3/3 F3/3 F0/3 P1/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F1/3 F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

The 30-trial results are in the campaign section below.

## Cylinder campaign, long cold-reset reinforcement learning with the temperature frozen (`0101`, replicate of `0100` with `alpha_freeze`, from `0099` online_4000, 40,000 interactions): 30-trial frozen evaluation

Environment: the archived task with `object_shape: circular` (40 mm). Recipe as `0100` (60 % of the episodes from seed_172's recorded prefixes 0-4 with weights 3:3:2:1:1, so that prefixes 0 and 1 carry six tenths of them; the long-weighted suffix stream over the 51 training sources for the rest; the same supervision as `0099`) with the temperature frozen at its inherited value (normalized 0.0272; `temperature_override.json`). The learner stayed stable: actor BC loss 0.0014 to 0.0013, spread-anchor loss 1.2e-5 first and last (range 1.0e-5 to 3.5e-5), critic loss 5.1 to 0.76, the critic's mean bias over the first demonstration episode -6.6 to -5.1, the stream's raw reward sum rising to 369,000 (it falls at 16,768 of 39,999 single updates, by at most 266, and at no 1,000-interaction block; `0100`: -153,000). Evaluated: seed_172 (8 controls), seed_116 (9), seed_174 (9), the held-out seeds 153, 154, 155, and the shared cells k0-k4 at online_20000 (the in-run k2 and k4 cells both 3/3 with deterministic passes) and online_40000. A suffix at least as long as the source is the cold reset itself (marked "cold").

**online_20000** (cold reset 0/30 (deterministic fail))

| source | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed_172 | 30/30 D | 30/30 D | 29/30 D | 22/30 D | 4/30 | 22/30 D | 0/30 | 0/30 (cold) | 0/30 (cold) |
| seed_116 | 30/30 D | 25/30 D | 24/30 D | 1/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 (cold) |
| seed_174 | 30/30 D | 1/30 | 17/30 D | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 (cold) |
| seed_153 (held out) | 29/30 D | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_154 (held out) | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_155 (held out) | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| training sources | 90/90 (3 D) | 56/90 (2 D) | 70/90 (3 D) | 23/90 (1 D) | 4/90 (0 D) | 22/90 (1 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) |
| held-out sources | 29/90 (1 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) |
| all 6 | 119/180 | 56/180 | 70/180 | 23/180 | 4/180 | 22/180 | 0/180 | 0/180 | 0/180 |

**online_40000** (cold reset 0/30 (deterministic fail))

| source | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed_172 | 30/30 D | 30/30 D | 28/30 D | 24/30 | 1/30 | 17/30 D | 0/30 | 0/30 (cold) | 0/30 (cold) |
| seed_116 | 30/30 D | 30/30 D | 28/30 D | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 (cold) |
| seed_174 | 30/30 D | 6/30 | 16/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 (cold) |
| seed_153 (held out) | 19/30 D | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_154 (held out) | 17/30 D | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_155 (held out) | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| training sources | 90/90 (3 D) | 66/90 (2 D) | 72/90 (2 D) | 24/90 (0 D) | 1/90 (0 D) | 17/90 (1 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) |
| held-out sources | 36/90 (2 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) |
| all 6 | 126/180 | 66/180 | 72/180 | 24/180 | 1/180 | 17/180 | 0/180 | 0/180 | 0/180 |

Shared cells (30 stochastic trials plus one deterministic): online_20000 k0 0/30 (every trial dropped, 25 at the second policy control), k1 0/30 (palm support at the second control in all 31 trials), k2 17/30 with a deterministic pass, k3 8/30, k4 27/30 with a deterministic pass; online_40000 k0 0/30 (all dropped at the second control), k1 0/30 (palm support at the second control in all 31), k2 17/30 with a deterministic pass, k3 4/30, k4 26/30 (deterministic fail).

**Reading.** With the temperature frozen the long run learned where `0100` diverged: the state after seed_172's second recorded control went from 6/30 D (`0099` online_4000) to 17/30 D at both evaluated checkpoints (seed_172 length 6: 22/30 D and 17/30 D; in-run 3/3 with a deterministic pass at five of eight checkpoints; stream 732 successes in 1,455 episodes), the state after its fourth control from 22/30 D to 27/30 D and 26/30, the last four controls from 23/30 D to 22/30 D and 24/30, and the held-out last control from 24/90 to 29/90 and 36/90 (seeds 153 and 154 at 19/30 D and 17/30 D at online_40000; seed_155 0/30 throughout); against the same reference other cells fell: seed_174's last four from 9/30 D to 0/30 at both checkpoints, seed_172's last five from 8/30 to 4/30 and 1/30, the state after three controls from 7/30 to 8/30 and 4/30, so the gain is confined to the states after two and four controls, seed_116's last two and three controls and the held-out last control. The state after one recorded control fell from 2/30 (`0099` online_4000) to 0/30 at both checkpoints and the cold reset stayed at 0/30, with 0/3 and 0/5 at all nine in-run points (the baseline and eight checkpoints); in the stream the state after one control produced 4 successes in 2,295 episodes, all within the first 1,800 interactions and none in the remaining 38,200, and the cold reset 0 in 2,349 (2,346 drops, 3 palm support). At the state after one control every frozen trial ends by palm support at its second policy control (both checkpoints, 62 of 62); at the cold reset by a drop at the second (56 of 62). Against the hypothesis (exploration rather than temperature runaway as the test): the stable run reached its frontier at the state after two controls and did not extend it in 40,000 interactions; the four early stochastic successes from the state after one control were not amplified. What the data show is that the successes from the state after one control that existed (2/30 frozen at the inherited checkpoint, 2 of 37 warm-up episodes, 4 early stream episodes) were not retained or amplified by this recipe, not that the state is unreachable; `cylinder_first_control_0001` gives the open-loop tolerance of the recorded first control at the cold reset (1e-5 to 1e-6; the second control was not measured one by one) and `cylinder_dart_0001` shows the planner's own control at both states is set-valued, so neither imitation route supplies a target there. No cold-reset trial succeeded; nothing here is primitive-eligible; the circular object is a disclosed deviation from the paper's hexagonal prism.

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
