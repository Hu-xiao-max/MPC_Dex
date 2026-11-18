# Variant 73 `ms_suffix1to6_all_long_from68`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_set": [8], "demonstration_include_mpc": false, "actor_bc_lambda": 100.0, "online_policy_interactions": 4000, "checkpoints": [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], "initial_from": ["bootstrap_variant_0068_ms_suffix123_from61", "online_2000"], "actor_lr_override": 0.0001, "eval_stochastic_trials": 3, "actor_bc_std_target": 0.01, "prefix_sources": ["mpc_cold_demonstrations_0001", 60], "suffix_set": [1, 2, 3, 4, 5, 6], "demonstration_suffix_rows": 6, "eval_source_count": 2, "holdout_source_count": 2}`. Inherited checkpoint: `bootstrap_variant_0068_ms_suffix123_from61/checkpoints/online_2000`. Hypothesis: Alternative second stage: suffix lengths 1-6 over every available demonstration (34 training, 2 held out) for 4,000 updates from the suffix-1-3 checkpoint, against 0070 (23 sources, 2,000 updates) whose suffix-3 rate fell while longer suffixes barely started.

Stage status `complete_sac_stage` (exit=0, 1564 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 4000, 'online_updates': 17750}; 500 critic-pretraining and 4000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: overridden to 0.0001 for this stage (inherited 0.0001; ramp fraction unchanged). Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over all demonstration rows, added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.01)) added to the same actor step. Handoff noise: off.

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'s1': 50, 's2': 49, 's3': 41, 's4': 31, 's5': 38, 's6': 47}, force-qualified successes {'s1': 40, 's2': 27, 's3': 16, 's4': 4, 's5': 1, 's6': 2} (informational; the gate is the 210-row demonstration buffer).
- Online: 4000 interactions; completed episodes by prefix {'s1': 123, 's2': 127, 's3': 118, 's4': 119, 's5': 117, 's6': 129}, successes {'s1': 117, 's2': 82, 's3': 52, 's4': 24, 's5': 20, 's6': 10}; outcomes {'palm_support': 168, 'drop': 236, 'truncated': 24, 'success': 305}; handoffs terminated before the policy 0; unfinished last suffix True. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 13678; none entered replay.
- Actor BC loss first/last: 2.941 / 0.0003847; spread-anchor loss first/last: 0.05019 / 8.379e-06; alpha raw-equivalent last 0.3166.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 3.4380, std 0.0081, bias -0.86/-0.78; after_pretraining: L2 3.4380, std 0.0081, bias -0.07/-0.08; online_500: L2 2.5735, std 0.0101, bias -0.58/-0.57; online_1000: L2 1.2265, std 0.0100, bias -0.87/-0.86; online_1500: L2 0.4713, std 0.0100, bias -1.19/-1.19; online_2000: L2 0.1928, std 0.0100, bias -1.10/-1.08; online_2500: L2 0.0521, std 0.0100, bias -0.79/-0.77; online_3000: L2 0.0077, std 0.0100, bias -0.53/-0.50; online_3500: L2 0.0211, std 0.0100, bias -0.63/-0.61; online_4000: L2 0.0275, std 0.0100, bias -0.67/-0.64.

## Frozen evaluations (in-run; 3 stochastic seeds per prefix)

| Checkpoint | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch | Cold det / stoch |
| --- | --- | --- | --- | --- |
| pretraining_baseline | fail / 0/3 | fail / 0/3 | pass / 3/3 | fail |
| online_500 | fail / 0/3 | fail / 0/3 | pass / 1/3 | fail |
| online_1000 | fail / 0/3 | fail / 0/3 | pass / 2/3 | fail |
| online_1500 | fail / 0/3 | fail / 0/3 | fail / 0/3 | fail |
| online_2000 | fail / 0/3 | fail / 0/3 | fail / 1/3 | fail |
| online_2500 | fail / 0/3 | fail / 0/3 | fail / 1/3 | fail |
| online_3000 | fail / 0/3 | fail / 0/3 | fail / 0/3 | fail |
| online_3500 | fail / 0/3 | fail / 0/3 | fail / 0/3 | fail |
| online_4000 | fail / 0/3 | fail / 0/3 | fail / 0/3 | fail / 0/5 |

The cold column counts only the cold-reset trials (an earlier rendering of this table had added the multi-source cells to it; corrected 2026-10-03).

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

No large-sample result for this stage at the time of writing.

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
