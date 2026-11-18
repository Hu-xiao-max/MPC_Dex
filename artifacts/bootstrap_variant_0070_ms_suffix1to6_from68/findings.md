# Variant 70 `ms_suffix1to6_from68`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_set": [6, 7, 8], "demonstration_include_mpc": false, "actor_bc_lambda": 100.0, "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "initial_from": ["bootstrap_variant_0068_ms_suffix123_from61", "online_2000"], "actor_lr_override": 0.0001, "actor_bc_std_target": 0.01, "prefix_sources": ["mpc_cold_demonstrations_0001", 25], "suffix_set": [1, 2, 3, 4, 5, 6], "demonstration_suffix_rows": 6, "holdout_source_count": 2}`. Inherited checkpoint: `bootstrap_variant_0068_ms_suffix123_from61/checkpoints/online_2000`. Hypothesis: Second multi-source stage: suffix lengths 1-6 over 23 training demonstrations with the last six rows pinned, two demonstrations held out of training and evaluated only, inheriting the suffix-1-3 stage.

Stage status `complete_sac_stage` (exit=0, 2647 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 4000, 'online_updates': 15750}; 500 critic-pretraining and 2000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: overridden to 0.0001 for this stage (inherited 0.0001; ramp fraction unchanged). Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over all demonstration rows, added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.01)) added to the same actor step. Handoff noise: off.

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'s1': 50, 's2': 49, 's3': 41, 's4': 31, 's5': 38, 's6': 47}, force-qualified successes {'s1': 39, 's2': 21, 's3': 15, 's4': 3, 's5': 0, 's6': 1} (informational; the gate is the 138-row demonstration buffer).
- Online: 2000 interactions; completed episodes by prefix {'s1': 53, 's2': 58, 's3': 53, 's4': 54, 's5': 52, 's6': 58}, successes {'s1': 45, 's2': 24, 's3': 21, 's4': 10, 's5': 5, 's6': 3}; outcomes {'drop': 111, 'palm_support': 102, 'success': 108, 'truncated': 7}; handoffs terminated before the policy 0; unfinished last suffix True. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 8146; none entered replay.
- Actor BC loss first/last: 2.521 / 0.003261; spread-anchor loss first/last: 0.04464 / 5.598e-05; alpha raw-equivalent last 0.2586.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 3.4380, std 0.0081, bias -0.86/-0.78; after_pretraining: L2 3.4380, std 0.0081, bias -0.09/-0.07; online_250: L2 2.6461, std 0.0098, bias -0.44/-0.40; online_500: L2 1.9851, std 0.0096, bias -0.50/-0.48; online_750: L2 1.2154, std 0.0094, bias -0.48/-0.47; online_1000: L2 0.6927, std 0.0095, bias -0.51/-0.48; online_1250: L2 0.3273, std 0.0097, bias -0.43/-0.42; online_1500: L2 0.1487, std 0.0099, bias -0.41/-0.46; online_1750: L2 0.0558, std 0.0100, bias -0.37/-0.42; online_2000: L2 0.0093, std 0.0100, bias -0.42/-0.38.

## Frozen evaluations (in-run; 5 stochastic seeds per prefix)

| Checkpoint | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch | Cold det / stoch |
| --- | --- | --- | --- | --- |
| pretraining_baseline | fail / 0/5 | fail / 0/5 | pass / 5/5 | fail |
| online_250 | fail / 0/5 | fail / 1/5 | fail / 0/5 | fail |
| online_500 | fail / 0/5 | fail / 0/5 | fail / 0/5 | fail |
| online_750 | fail / 0/5 | fail / 0/5 | fail / 0/5 | fail |
| online_1000 | fail / 0/5 | fail / 0/5 | fail / 2/5 | fail |
| online_1250 | fail / 0/5 | fail / 0/5 | fail / 1/5 | fail |
| online_1500 | fail / 0/5 | fail / 0/5 | fail / 0/5 | fail |
| online_1750 | fail / 0/5 | fail / 0/5 | pass / 0/5 | fail |
| online_2000 | fail / 0/5 | fail / 0/5 | fail / 1/5 | fail / 0/5 |

The cold column counts only the cold-reset trials (an earlier rendering of this table had added the multi-source cells to it; corrected 2026-10-03).

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

No large-sample result for this stage at the time of writing.

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
