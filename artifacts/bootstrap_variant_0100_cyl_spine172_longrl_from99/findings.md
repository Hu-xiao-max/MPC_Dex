# Variant 100 `cyl_spine172_longrl_from99`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_set": [8], "demonstration_include_mpc": false, "actor_bc_lambda": 100.0, "actor_bc_rows": "per_item", "online_policy_interactions": 40000, "checkpoints": [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000], "initial_from": ["bootstrap_variant_0099_cyl_spine172_from96", "online_4000"], "actor_lr_override": 0.0001, "eval_stochastic_trials": 3, "actor_bc_std_target": 0.01, "prefix_sources": ["cylinder_demonstrations_0001", 120], "suffix_set": [1, 2, 3, 4, 5, 6, 7, 8, 9], "demonstration_suffix_rows": 12, "suffix_weights": [1, 1, 1, 2, 2, 2, 2, 2, 2], "holdout_source_names": ["seed_153", "seed_154", "seed_155"], "eval_source_names": ["seed_172", "seed_116", "seed_174"], "start_prefix_set": [0, 1, 2, 3, 4], "start_prefix_weights": [3, 3, 2, 1, 1], "start_prefix_share": 0.6, "actor_bc_last_rows": 8, "spine_source": "seed_172", "actor_bc_shared_lambda": 50.0, "env_overrides": {"object_shape": "circular"}, "eval_recorded_prefixes": false, "eval_cold_every_point": true}`. Inherited checkpoint: `bootstrap_variant_0099_cyl_spine172_from96/checkpoints/online_4000`. Hypothesis: Long cold-reset reinforcement learning on the cylinder, the one lever not yet tried there: ten times the interactions of every earlier stage (40,000; the hexagon run bootstrap_0004 stopped at 53,800 with no success) from the 0099 online_4000 actor, which completes the turn from the states after seed_172 recorded controls 2-4 in 6/30, 7/30 and 22/30 trials and reached one changing-policy cold-reset success in 97 episodes. 60 % of the episodes start at the spine prefixes 0-4 with weights 3:3:2:1:1 (the cold reset and the state after one control carry half of them), the rest keep the long-weighted suffix stream over the 51 training sources; same supervision as 0099. cylinder_first_control_0001 section 4 shows that closed-loop strategies tolerant of 0.03-0.10 execution error exist in this task, so the question is whether SAC can find one from a policy that drops at its third control. Evaluated on the three shortest sources, the held-out trio, the shared cells k0-k4 and the cold reset at every 5,000 interactions. Success criterion unchanged: a frozen cold-reset turn.

Stage status `complete_sac_stage` (?, 4600 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 8500, 'online_updates': 86250}; 500 critic-pretraining and 40000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: overridden to 0.0001 for this stage (inherited 0.0001; ramp fraction unchanged). Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over the last 8 rows of each demonstration episode plus its shared cold-start prefix rows; additionally, per distinct shared state, lambda_shared 50.0 * (MSE of the actor mean to the label + MSE of the log-std to log(std target)) summed over the distinct shared states (one term per state, however many sources sit on it), added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.01)) added to the same actor step. Handoff noise: off.

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'s1': 10, 's2': 5, 's3': 5, 's4': 14, 's5': 10, 's6': 18, 's7': 9, 's8': 15, 's9': 16, 'k0': 48, 'k1': 37, 'k2': 40, 'k3': 16, 'k4': 13}, force-qualified successes {'s1': 10, 's2': 4, 's3': 3, 's4': 2, 's5': 1, 's6': 4, 's7': 0, 's8': 1, 's9': 0, 'k0': 0, 'k1': 2, 'k2': 9, 'k3': 2, 'k4': 9} (informational; the gate is the 587-row demonstration buffer).
- Online: 40000 interactions; completed episodes by prefix {'s1': 433, 's2': 408, 's3': 394, 's4': 771, 's5': 798, 's6': 834, 's7': 816, 's8': 796, 's9': 792, 'k0': 2698, 'k1': 2677, 'k2': 1731, 'k3': 932, 'k4': 931}, successes {'s1': 402, 's2': 246, 's3': 98, 's4': 75, 's5': 49, 's6': 31, 's7': 14, 's8': 7, 's9': 0, 'k0': 0, 'k1': 0, 'k2': 241, 'k3': 52, 'k4': 126}; outcomes {'drop': 8952, 'palm_support': 4674, 'success': 1341, 'truncated': 44}; handoffs terminated before the policy 0; unfinished last suffix False. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 74679; none entered replay.
- Actor BC loss first/last: 0.001436 / 0.06283; spread-anchor loss first/last: 1.208e-05 / 0.004037; alpha raw-equivalent last 278.7175.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 1.5765, std 0.0100, bias -6.63/-6.75; after_pretraining: L2 1.5765, std 0.0100, bias -3.83/-3.92; online_5000: L2 1.5727, std 0.0100, bias -8.16/-7.96; online_10000: L2 1.5554, std 0.0100, bias -13.28/-13.52; online_15000: L2 1.5585, std 0.0100, bias -20.56/-20.70; online_20000: L2 1.5538, std 0.0101, bias -28.69/-29.53; online_25000: L2 1.5539, std 0.0100, bias -41.69/-41.63; online_30000: L2 1.5741, std 0.0100, bias -52.81/-54.34; online_35000: L2 1.5829, std 0.0100, bias -75.59/-75.99; online_40000: L2 1.5124, std 0.0101, bias -106.24/-106.85.

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
| pretraining_baseline | F0/5 | F0/3 | P1/3 | F0/3 | P2/3 | P3/3 P3/3 P3/3 P2/3 F2/3 P0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 F2/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 P0/3 P0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P2/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_5000 | F0/5 | F0/3 | F0/3 | P1/3 | P3/3 | P3/3 P3/3 P3/3 P3/3 P0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_10000 | F0/5 | F0/3 | F0/3 | F0/3 | F0/3 | P3/3 P3/3 F3/3 F0/3 F0/3 F1/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F2/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_15000 | F0/5 | F0/3 | P2/3 | F0/3 | F0/3 | P3/3 P3/3 P2/3 F0/3 F0/3 P2/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F1/3 P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_20000 | F0/5 | F0/3 | F1/3 | F0/3 | F0/3 | P3/3 P3/3 F2/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F2/3 P2/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_25000 | F0/5 | F0/3 | F0/3 | F0/3 | F0/3 | P3/3 P3/3 P3/3 F0/3 F0/3 F1/3 F0/3 F0/3 F0/3 | P3/3 P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_30000 | F0/5 | F0/3 | F0/3 | F0/3 | F0/3 | P3/3 P3/3 F2/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P2/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_35000 | F0/5 | F0/3 | F0/3 | F0/3 | F0/3 | P3/3 P3/3 P2/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_40000 | F0/5 | F0/3 | F0/3 | F0/3 | F0/3 | P3/3 P3/3 P1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

The 30-trial results are in the campaign section below.

## Cylinder campaign, long cold-reset reinforcement learning on the seed_172 spine (`0100`, from `0099` online_4000, 40,000 interactions): 30-trial frozen evaluation

Environment: the archived task with `object_shape: circular` (40 mm). Recipe as `0099` except ten times the interactions (40,000, checkpoints every 5,000) and 60 % of the episodes starting at seed_172's recorded prefixes 0-4 with weights 3:3:2:1:1 (the cold reset and the state after one control carry six tenths of them; the stamped hypothesis says 'half'); the temperature was left learning as in every earlier cylinder stage (`0089`-`0099`). Evaluated: seed_172 (8 controls), seed_116 (9), seed_174 (9), the held-out seeds 153, 154, 155, and the shared cells k0-k4 at online_15000 (the one online checkpoint whose in-run deterministic k2 trial passed; the inherited actor's baseline trial also passed, P1/3) and online_40000. A suffix at least as long as the source is the cold reset itself (marked "cold").

**online_15000** (cold reset 0/30 (deterministic fail))

| source | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed_172 | 30/30 D | 30/30 D | 24/30 D | 0/30 | 0/30 | 13/30 D | 0/30 | 0/30 (cold) | 0/30 (cold) |
| seed_116 | 30/30 D | 29/30 D | 29/30 D | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 (cold) |
| seed_174 | 30/30 D | 1/30 | 14/30 D | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 (cold) |
| seed_153 (held out) | 2/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_154 (held out) | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_155 (held out) | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| training sources | 90/90 (3 D) | 60/90 (2 D) | 67/90 (3 D) | 0/90 (0 D) | 0/90 (0 D) | 13/90 (1 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) |
| held-out sources | 2/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) |
| all 6 | 92/180 | 60/180 | 67/180 | 0/180 | 0/180 | 13/180 | 0/180 | 0/180 | 0/180 |

**online_40000** (cold reset 0/30 (deterministic fail))

| source | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed_172 | 30/30 D | 30/30 D | 27/30 D | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 (cold) | 0/30 (cold) |
| seed_116 | 30/30 D | 28/30 D | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 (cold) |
| seed_174 | 30/30 D | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 (cold) |
| seed_153 (held out) | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_154 (held out) | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| seed_155 (held out) | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 |
| training sources | 90/90 (3 D) | 58/90 (2 D) | 27/90 (1 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) |
| held-out sources | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) | 0/90 (0 D) |
| all 6 | 90/180 | 58/180 | 27/180 | 0/180 | 0/180 | 0/180 | 0/180 | 0/180 | 0/180 |

Shared cells (30 stochastic trials plus one deterministic): online_15000 k0 0/30 (every trial dropped at the second policy control), k1 0/30 (palm support at the second or third control), k2 16/30 with a deterministic pass (the campaign's highest k2 cell; `0099` online_4000 had 6/30 D), k3 0/30 (palm support at the second control in every trial) and k4 0/30 (palm support at the second or third control in 30 trials, one drop at the fourth); online_40000 k0-k4 all 0/30 with deterministic fails (drops or palm support within two to four controls).

**Reading.** Ten times the interactions produced no cold-reset success anywhere: 0/5 in-run at all eight checkpoints, 0/30 at the two evaluated ones, and 0 successes in 2,764 cold-reset training episodes (2,698 at the k0 cell, 66 suffix-stream episodes clamped to the cold reset; changing-policy data); the state after one spine control is likewise 0/3 at every checkpoint, 0/30 at both evaluated ones and 0 of 2,677 training episodes. The states after two to four controls kept training successes (241/1,731, 52/932, 126/931), but all 52 k3 successes came before 12,000 interactions and all 241 k2 successes before 28,000; at 15,000 the frozen policy completes the turn from the state after two controls in 16/30 (seed_172 length 6: 13/30 D) while the states after three and four controls, which it completed at the start (`0099`: 7/30 and 22/30 D), are 0/30, and by 40,000 every cell beyond length 3 is 0/30 on every source. The run diverged: the learned temperature rose from 0.027 to 1.39 while the policy entropy (-82 to -51) stayed far below the -16 target that the 0.01 spread anchor makes unreachable, the critic's mean bias over the first demonstration episode's rows (Q minus the discounted return) drifted from -6.6 to -106 and its loss from 5 to 370, the actor BC loss grew from 0.0014 to 0.063 and the stream's raw reward sum, 26,000 at 20,000 interactions, was -153,000 at the end. Against the hypothesis (SAC finding a cold-reset strategy from a policy that drops at its second control; the stamped text says 'third'): the run found none in 2,764 attempts and lost the spine competence beyond length 3 that it started with; because the temperature runaway confounds the exploration question, the frozen-temperature replicate `0101` (template knob `alpha_freeze`) repeats the run. No cold-reset trial succeeded; nothing here is primitive-eligible; the circular object is a disclosed deviation from the paper's hexagonal prism.

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
