# Variant 78 `ms_suffix1to7_from74`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_set": [8], "demonstration_include_mpc": false, "actor_bc_lambda": 100.0, "online_policy_interactions": 4000, "checkpoints": [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], "initial_from": ["bootstrap_variant_0074_ms_suffix1to6_cont_from70", "online_4000"], "actor_lr_override": 0.0001, "eval_stochastic_trials": 3, "actor_bc_std_target": 0.01, "prefix_sources": ["mpc_cold_demonstrations_0001", 80], "suffix_set": [1, 2, 3, 4, 5, 6, 7], "demonstration_suffix_rows": 7, "eval_source_count": 2, "holdout_source_count": 2}`. Inherited checkpoint: `bootstrap_variant_0074_ms_suffix1to6_cont_from70/checkpoints/online_4000`. Hypothesis: Smaller curriculum step: suffix lengths 1-7 from 0074 online_4000 (one added control instead of two).

Stage status `complete_sac_stage` (exit=0, 1356 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 5000, 'online_updates': 23750}; 500 critic-pretraining and 4000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: overridden to 0.0001 for this stage (inherited 0.0001; ramp fraction unchanged). Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over all demonstration rows, added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.01)) added to the same actor step. Handoff noise: off.

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'s1': 41, 's2': 51, 's3': 29, 's4': 31, 's5': 26, 's6': 37, 's7': 41}, force-qualified successes {'s1': 40, 's2': 40, 's3': 18, 's4': 8, 's5': 7, 's6': 7, 's7': 0} (informational; the gate is the 245-row demonstration buffer).
- Online: 4000 interactions; completed episodes by prefix {'s1': 130, 's2': 115, 's3': 122, 's4': 117, 's5': 122, 's6': 114, 's7': 124}, successes {'s1': 128, 's2': 96, 's3': 59, 's4': 34, 's5': 27, 's6': 15, 's7': 10}; outcomes {'drop': 312, 'palm_support': 147, 'success': 369, 'truncated': 16}; handoffs terminated before the policy 0; unfinished last suffix True. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 14699; none entered replay.
- Actor BC loss first/last: 0.4673 / 0.0002181; spread-anchor loss first/last: 0.0008599 / 4.844e-06; alpha raw-equivalent last 0.5777.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 3.6204, std 0.0104, bias -1.06/-1.16; after_pretraining: L2 3.6204, std 0.0104, bias -0.34/-0.37; online_500: L2 0.2938, std 0.0101, bias -0.62/-0.59; online_1000: L2 0.0526, std 0.0100, bias -0.75/-0.77; online_1500: L2 0.0109, std 0.0100, bias -0.81/-0.81; online_2000: L2 0.0030, std 0.0100, bias -0.78/-0.82; online_2500: L2 0.0160, std 0.0100, bias -0.84/-0.85; online_3000: L2 0.0101, std 0.0100, bias -0.98/-1.01; online_3500: L2 0.0075, std 0.0100, bias -1.13/-1.14; online_4000: L2 0.0088, std 0.0100, bias -1.24/-1.25.

## Frozen evaluations (in-run; 3 stochastic seeds per prefix)

| Checkpoint | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch | Cold det / stoch |
| --- | --- | --- | --- | --- |
| pretraining_baseline | fail / 0/3 | fail / 0/3 | fail / 2/3 | fail |
| online_500 | fail / 0/3 | fail / 0/3 | fail / 0/3 | fail |
| online_1000 | fail / 0/3 | fail / 0/3 | fail / 0/3 | fail |
| online_1500 | fail / 0/3 | fail / 0/3 | fail / 0/3 | fail |
| online_2000 | fail / 0/3 | fail / 0/3 | fail / 0/3 | fail |
| online_2500 | fail / 0/3 | fail / 0/3 | fail / 0/3 | fail |
| online_3000 | fail / 0/3 | fail / 0/3 | fail / 0/3 | fail |
| online_3500 | fail / 0/3 | fail / 0/3 | fail / 0/3 | fail |
| online_4000 | fail / 0/3 | fail / 0/3 | pass / 1/3 | fail / 0/5 |

The cold column counts only the cold-reset trials (an earlier rendering of this table had added the multi-source cells to it; corrected 2026-10-03).

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

No large-sample result for this stage at the time of writing.

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
