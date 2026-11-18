# Variant 74 `ms_suffix1to6_cont_from70`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_set": [8], "demonstration_include_mpc": false, "actor_bc_lambda": 100.0, "online_policy_interactions": 4000, "checkpoints": [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], "initial_from": ["bootstrap_variant_0070_ms_suffix1to6_from68", "online_2000"], "actor_lr_override": 0.0001, "eval_stochastic_trials": 3, "actor_bc_std_target": 0.01, "prefix_sources": ["mpc_cold_demonstrations_0001", 60], "suffix_set": [1, 2, 3, 4, 5, 6], "demonstration_suffix_rows": 6, "eval_source_count": 2, "holdout_source_count": 2}`. Inherited checkpoint: `bootstrap_variant_0070_ms_suffix1to6_from68/checkpoints/online_2000`. Hypothesis: Continue the suffix-1-6 stage for 4,000 more updates over every available demonstration: its rates were still rising at online 1,500 (20/25, 16/25, 10/25, 6/25, 6/25, 4/25).

Stage status `complete_sac_stage` (exit=0, 1463 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 4500, 'online_updates': 19750}; 500 critic-pretraining and 4000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: overridden to 0.0001 for this stage (inherited 0.0001; ramp fraction unchanged). Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over all demonstration rows, added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.01)) added to the same actor step. Handoff noise: off.

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'s1': 50, 's2': 49, 's3': 41, 's4': 31, 's5': 38, 's6': 47}, force-qualified successes {'s1': 40, 's2': 39, 's3': 20, 's4': 9, 's5': 8, 's6': 2} (informational; the gate is the 210-row demonstration buffer).
- Online: 4000 interactions; completed episodes by prefix {'s1': 148, 's2': 147, 's3': 135, 's4': 140, 's5': 139, 's6': 150}, successes {'s1': 144, 's2': 119, 's3': 56, 's4': 39, 's5': 28, 's6': 9}; outcomes {'palm_support': 181, 'drop': 264, 'success': 395, 'truncated': 19}; handoffs terminated before the policy 0; unfinished last suffix True. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 15416; none entered replay.
- Actor BC loss first/last: 1.133 / 0.0002175; spread-anchor loss first/last: 0.002935 / 3.559e-06; alpha raw-equivalent last 0.3872.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 0.0093, std 0.0100, bias -0.42/-0.38; after_pretraining: L2 0.0093, std 0.0100, bias -0.08/-0.10; online_500: L2 0.2282, std 0.0097, bias -0.73/-0.72; online_1000: L2 0.1329, std 0.0099, bias -0.42/-0.38; online_1500: L2 0.0351, std 0.0099, bias -0.48/-0.48; online_2000: L2 0.0170, std 0.0100, bias -0.58/-0.56; online_2500: L2 0.0132, std 0.0100, bias -0.62/-0.57; online_3000: L2 0.0149, std 0.0100, bias -0.68/-0.74; online_3500: L2 0.0194, std 0.0100, bias -0.73/-0.78; online_4000: L2 0.0083, std 0.0100, bias -0.75/-0.76.

## Frozen evaluations (in-run; 3 stochastic seeds per prefix)

| Checkpoint | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch | Cold det / stoch |
| --- | --- | --- | --- | --- |
| pretraining_baseline | fail / 0/3 | fail / 0/3 | fail / 1/3 | fail |
| online_500 | fail / 0/3 | fail / 0/3 | fail / 0/3 | fail |
| online_1000 | fail / 0/3 | fail / 0/3 | fail / 0/3 | fail |
| online_1500 | fail / 0/3 | fail / 0/3 | fail / 0/3 | fail |
| online_2000 | fail / 0/3 | fail / 0/3 | fail / 0/3 | fail |
| online_2500 | fail / 0/3 | fail / 0/3 | pass / 1/3 | fail |
| online_3000 | fail / 0/3 | fail / 0/3 | pass / 1/3 | fail |
| online_3500 | fail / 0/3 | fail / 0/3 | pass / 2/3 | fail |
| online_4000 | fail / 0/3 | fail / 0/3 | fail / 2/3 | fail / 0/5 |

The cold column counts only the cold-reset trials (an earlier rendering of this table had added the multi-source cells to it; corrected 2026-10-03).

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

No large-sample result for this stage at the time of writing.

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
