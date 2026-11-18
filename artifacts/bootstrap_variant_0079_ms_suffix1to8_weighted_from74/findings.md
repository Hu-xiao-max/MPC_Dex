# Variant 79 `ms_suffix1to8_weighted_from74`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_set": [8], "demonstration_include_mpc": false, "actor_bc_lambda": 100.0, "online_policy_interactions": 4000, "checkpoints": [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], "initial_from": ["bootstrap_variant_0074_ms_suffix1to6_cont_from70", "online_4000"], "actor_lr_override": 0.0001, "eval_stochastic_trials": 3, "actor_bc_std_target": 0.01, "prefix_sources": ["mpc_cold_demonstrations_0001", 120], "suffix_set": [1, 2, 3, 4, 5, 6, 7, 8], "demonstration_suffix_rows": 8, "eval_source_count": 2, "holdout_source_count": 3, "suffix_weights": [2, 2, 2, 2, 1, 1, 1, 1]}`. Inherited checkpoint: `bootstrap_variant_0074_ms_suffix1to6_cont_from70/checkpoints/online_4000`. Hypothesis: Root-requested rerun from 0074 online_4000 over the enlarged demonstration set with suffix sampling weights 2:2:2:2:1:1:1:1 for lengths 1-8, so the short suffixes keep the share they had in the 1-6 stage (uniform 1-8 traded short-suffix competence for little gain at longer suffixes in 0076/0077); three demonstrations held out.

Stage status `complete_sac_stage` (exit=0, 1776 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 5000, 'online_updates': 23750}; 500 critic-pretraining and 4000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: overridden to 0.0001 for this stage (inherited 0.0001; ramp fraction unchanged). Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over all demonstration rows, added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.01)) added to the same actor step. Handoff noise: off.

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'s1': 48, 's2': 42, 's3': 45, 's4': 38, 's5': 13, 's6': 24, 's7': 20, 's8': 26}, force-qualified successes {'s1': 40, 's2': 25, 's3': 17, 's4': 7, 's5': 3, 's6': 0, 's7': 0, 's8': 2} (informational; the gate is the 424-row demonstration buffer).
- Online: 4000 interactions; completed episodes by prefix {'s1': 149, 's2': 124, 's3': 128, 's4': 129, 's5': 75, 's6': 68, 's7': 81, 's8': 82}, successes {'s1': 138, 's2': 76, 's3': 61, 's4': 35, 's5': 16, 's6': 6, 's7': 2, 's8': 2}; outcomes {'drop': 311, 'palm_support': 172, 'success': 336, 'truncated': 17}; handoffs terminated before the policy 0; unfinished last suffix True. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 14327; none entered replay.
- Actor BC loss first/last: 1.749 / 0.000991; spread-anchor loss first/last: 0.003182 / 1.239e-05; alpha raw-equivalent last 0.5782.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 4.5567, std 0.0108, bias -1.21/-1.44; after_pretraining: L2 4.5567, std 0.0108, bias -0.25/-0.27; online_500: L2 0.3017, std 0.0101, bias -1.29/-1.32; online_1000: L2 0.0746, std 0.0100, bias -1.28/-1.28; online_1500: L2 0.0125, std 0.0100, bias -1.04/-1.03; online_2000: L2 0.0062, std 0.0100, bias -1.19/-1.09; online_2500: L2 0.0076, std 0.0100, bias -1.18/-1.19; online_3000: L2 0.0063, std 0.0100, bias -1.11/-1.18; online_3500: L2 0.0065, std 0.0100, bias -1.26/-1.23; online_4000: L2 0.0036, std 0.0100, bias -1.34/-1.26.

## Frozen evaluations (in-run; 3 stochastic seeds per prefix)

| Checkpoint | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch | Cold det / stoch |
| --- | --- | --- | --- | --- |
| pretraining_baseline | fail / 0/3 | fail / 0/3 | fail / 2/3 | fail |
| online_500 | fail / 0/3 | fail / 0/3 | pass / 3/3 | fail |
| online_1000 | fail / 0/3 | fail / 0/3 | pass / 3/3 | fail |
| online_1500 | fail / 0/3 | fail / 0/3 | pass / 3/3 | fail |
| online_2000 | fail / 0/3 | fail / 0/3 | fail / 1/3 | fail |
| online_2500 | fail / 0/3 | fail / 0/3 | pass / 3/3 | fail |
| online_3000 | fail / 0/3 | fail / 0/3 | pass / 1/3 | fail |
| online_3500 | fail / 0/3 | fail / 0/3 | fail / 3/3 | fail |
| online_4000 | fail / 0/3 | fail / 1/3 | pass / 3/3 | fail / 0/5 |

The cold column counts only the cold-reset trials (an earlier rendering of this table had added the multi-source cells to it; corrected 2026-10-03).

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

No large-sample result for this stage at the time of writing.

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
