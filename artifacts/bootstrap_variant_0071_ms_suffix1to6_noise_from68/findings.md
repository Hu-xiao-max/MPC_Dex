# Variant 71 `ms_suffix1to6_noise_from68`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_set": [6, 7, 8], "demonstration_include_mpc": false, "actor_bc_lambda": 100.0, "handoff_noise_sigma": 0.015, "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "initial_from": ["bootstrap_variant_0068_ms_suffix123_from61", "online_2000"], "actor_lr_override": 0.0001, "actor_bc_std_target": 0.01, "prefix_sources": ["mpc_cold_demonstrations_0001", 25], "suffix_set": [1, 2, 3, 4, 5, 6], "demonstration_suffix_rows": 6, "handoff_noise_suffixes": [3, 4, 5, 6], "holdout_source_count": 2}`. Inherited checkpoint: `bootstrap_variant_0068_ms_suffix123_from61/checkpoints/online_2000`. Hypothesis: As ms_suffix1to6_from68 with sigma .015 handoff noise for suffixes 3-6.

Stage status `complete_sac_stage` (exit=0, 3506 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 4000, 'online_updates': 15750}; 500 critic-pretraining and 2000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: overridden to 0.0001 for this stage (inherited 0.0001; ramp fraction unchanged). Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over all demonstration rows, added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.01)) added to the same actor step. Handoff noise: off.

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'s1': 50, 's2': 49, 's3+noise': 41, 's4+noise': 31, 's5+noise': 38, 's6+noise': 47}, force-qualified successes {'s1': 38, 's2': 25, 's3+noise': 9, 's4+noise': 5, 's5+noise': 0, 's6+noise': 2} (informational; the gate is the 138-row demonstration buffer).
- Online: 2000 interactions; completed episodes by prefix {'s1': 51, 's2': 55, 's3+noise': 51, 's4+noise': 51, 's5+noise': 49, 's6+noise': 57}, successes {'s1': 45, 's2': 32, 's3+noise': 15, 's4+noise': 10, 's5+noise': 3, 's6+noise': 0}; outcomes {'drop': 105, 'palm_support': 100, 'success': 105, 'truncated': 4}; handoffs terminated before the policy 0; unfinished last suffix True. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 7973; none entered replay.
- Actor BC loss first/last: 2.521 / 0.003295; spread-anchor loss first/last: 0.04464 / 5.542e-05; alpha raw-equivalent last 0.2585.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 3.4380, std 0.0081, bias -0.86/-0.78; after_pretraining: L2 3.4380, std 0.0081, bias -0.17/-0.16; online_250: L2 2.6454, std 0.0098, bias -0.46/-0.48; online_500: L2 1.9852, std 0.0096, bias -0.54/-0.53; online_750: L2 1.2203, std 0.0094, bias -0.53/-0.55; online_1000: L2 0.7065, std 0.0095, bias -0.51/-0.57; online_1250: L2 0.3455, std 0.0097, bias -0.51/-0.49; online_1500: L2 0.1598, std 0.0099, bias -0.45/-0.41; online_1750: L2 0.0681, std 0.0100, bias -0.52/-0.45; online_2000: L2 0.0191, std 0.0100, bias -0.44/-0.40.

## Frozen evaluations (in-run; 5 stochastic seeds per prefix)

| Checkpoint | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch | Cold det / stoch |
| --- | --- | --- | --- | --- |
| pretraining_baseline | fail / 0/5 | fail / 0/5 | pass / 5/5 | fail |
| online_250 | fail / 0/5 | fail / 0/5 | fail / 0/5 | fail |
| online_500 | fail / 0/5 | fail / 0/5 | fail / 0/5 | fail |
| online_750 | fail / 0/5 | fail / 1/5 | fail / 1/5 | fail |
| online_1000 | fail / 0/5 | fail / 0/5 | fail / 0/5 | fail |
| online_1250 | fail / 1/5 | fail / 0/5 | fail / 0/5 | fail |
| online_1500 | fail / 0/5 | fail / 0/5 | fail / 0/5 | fail |
| online_1750 | fail / 0/5 | fail / 0/5 | fail / 1/5 | fail |
| online_2000 | fail / 0/5 | fail / 0/5 | pass / 1/5 | fail / 0/5 |

The cold column counts only the cold-reset trials (an earlier rendering of this table had added the multi-source cells to it; corrected 2026-10-03).

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

No large-sample result for this stage at the time of writing.

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
