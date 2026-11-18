# Variant 77 `ms_suffix1to8_cont_from76`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_set": [8], "demonstration_include_mpc": false, "actor_bc_lambda": 100.0, "online_policy_interactions": 4000, "checkpoints": [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], "initial_from": ["bootstrap_variant_0076_ms_suffix1to8_from74", "online_4000"], "actor_lr_override": 0.0001, "eval_stochastic_trials": 3, "actor_bc_std_target": 0.01, "prefix_sources": ["mpc_cold_demonstrations_0001", 80], "suffix_set": [1, 2, 3, 4, 5, 6, 7, 8], "demonstration_suffix_rows": 8, "eval_source_count": 2, "holdout_source_count": 2}`. Inherited checkpoint: `bootstrap_variant_0076_ms_suffix1to8_from74/checkpoints/online_4000`. Hypothesis: Continue the suffix-1-8 stage for 4,000 more updates: 0076 regressed at 30 trials (180/240, 150/240, 58/240 for lengths 1-3 versus 240/168/134 before), as 0074 did before recovering within one stage-length.

Stage status `complete_sac_stage` (exit=0, 1403 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 5500, 'online_updates': 27750}; 500 critic-pretraining and 4000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: overridden to 0.0001 for this stage (inherited 0.0001; ramp fraction unchanged). Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over all demonstration rows, added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.01)) added to the same actor step. Handoff noise: off.

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'s1': 37, 's2': 44, 's3': 29, 's4': 30, 's5': 24, 's6': 25, 's7': 32, 's8': 35}, force-qualified successes {'s1': 36, 's2': 34, 's3': 15, 's4': 14, 's5': 5, 's6': 3, 's7': 0, 's8': 0} (informational; the gate is the 280-row demonstration buffer).
- Online: 4000 interactions; completed episodes by prefix {'s1': 120, 's2': 94, 's3': 102, 's4': 107, 's5': 98, 's6': 109, 's7': 105, 's8': 108}, successes {'s1': 118, 's2': 78, 's3': 52, 's4': 39, 's5': 18, 's6': 23, 's7': 8, 's8': 11}; outcomes {'drop': 326, 'success': 347, 'palm_support': 155, 'truncated': 15}; handoffs terminated before the policy 0; unfinished last suffix True. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 14143; none entered replay.
- Actor BC loss first/last: 0.0002146 / 0.0002016; spread-anchor loss first/last: 1.925e-06 / 2.23e-06; alpha raw-equivalent last 0.8619.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 0.0055, std 0.0100, bias -1.44/-1.46; after_pretraining: L2 0.0055, std 0.0100, bias -0.77/-0.79; online_500: L2 0.0042, std 0.0100, bias -1.14/-1.12; online_1000: L2 0.0074, std 0.0100, bias -1.18/-1.21; online_1500: L2 0.0020, std 0.0100, bias -1.55/-1.56; online_2000: L2 0.0050, std 0.0100, bias -1.67/-1.63; online_2500: L2 0.0025, std 0.0100, bias -1.68/-1.66; online_3000: L2 0.0039, std 0.0100, bias -1.80/-1.84; online_3500: L2 0.0027, std 0.0100, bias -1.72/-1.75; online_4000: L2 0.0071, std 0.0100, bias -2.02/-2.05.

## Frozen evaluations (in-run; 3 stochastic seeds per prefix)

| Checkpoint | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch | Cold det / stoch |
| --- | --- | --- | --- | --- |
| pretraining_baseline | fail / 0/3 | fail / 0/3 | pass / 3/3 | fail |
| online_500 | fail / 0/3 | fail / 0/3 | pass / 3/3 | fail |
| online_1000 | fail / 0/3 | fail / 0/3 | pass / 3/3 | fail |
| online_1500 | fail / 0/3 | fail / 0/3 | pass / 3/3 | fail |
| online_2000 | fail / 0/3 | fail / 0/3 | pass / 3/3 | fail |
| online_2500 | fail / 0/3 | fail / 0/3 | pass / 3/3 | fail |
| online_3000 | fail / 0/3 | fail / 0/3 | pass / 3/3 | fail |
| online_3500 | fail / 0/3 | fail / 0/3 | pass / 3/3 | fail |
| online_4000 | fail / 0/3 | fail / 0/3 | pass / 3/3 | fail / 0/5 |

The cold column counts only the cold-reset trials (an earlier rendering of this table had added the multi-source cells to it; corrected 2026-10-03).

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

No large-sample result for this stage at the time of writing.

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
