# Variant 76 `ms_suffix1to8_from74`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_set": [8], "demonstration_include_mpc": false, "actor_bc_lambda": 100.0, "online_policy_interactions": 4000, "checkpoints": [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], "initial_from": ["bootstrap_variant_0074_ms_suffix1to6_cont_from70", "online_4000"], "actor_lr_override": 0.0001, "eval_stochastic_trials": 3, "actor_bc_std_target": 0.01, "prefix_sources": ["mpc_cold_demonstrations_0001", 80], "suffix_set": [1, 2, 3, 4, 5, 6, 7, 8], "demonstration_suffix_rows": 8, "eval_source_count": 2, "holdout_source_count": 2}`. Inherited checkpoint: `bootstrap_variant_0074_ms_suffix1to6_cont_from70/checkpoints/online_4000`. Hypothesis: Multi-source curriculum step requested by root: suffix lengths 1-8 over every available demonstration (two held out), last eight rows pinned, inheriting 0074 online_4000 (30-trial: 240/240, 168/240, 134/240, 49/240, 43/240, 23/240 for suffix lengths 1-6 on eight demonstrations).

Stage status `complete_sac_stage` (exit=0, 1298 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 5000, 'online_updates': 23750}; 500 critic-pretraining and 4000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: overridden to 0.0001 for this stage (inherited 0.0001; ramp fraction unchanged). Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over all demonstration rows, added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.01)) added to the same actor step. Handoff noise: off.

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'s1': 37, 's2': 44, 's3': 29, 's4': 30, 's5': 24, 's6': 25, 's7': 32, 's8': 35}, force-qualified successes {'s1': 36, 's2': 36, 's3': 16, 's4': 6, 's5': 6, 's6': 4, 's7': 1, 's8': 0} (informational; the gate is the 280-row demonstration buffer).
- Online: 4000 interactions; completed episodes by prefix {'s1': 113, 's2': 86, 's3': 99, 's4': 101, 's5': 92, 's6': 103, 's7': 102, 's8': 103}, successes {'s1': 113, 's2': 58, 's3': 47, 's4': 32, 's5': 13, 's6': 17, 's7': 7, 's8': 6}; outcomes {'drop': 313, 'palm_support': 184, 'success': 293, 'truncated': 9}; handoffs terminated before the policy 0; unfinished last suffix False. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 13546; none entered replay.
- Actor BC loss first/last: 0.9199 / 0.0002301; spread-anchor loss first/last: 0.001746 / 1.688e-06; alpha raw-equivalent last 0.5779.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 4.5567, std 0.0108, bias -1.21/-1.44; after_pretraining: L2 4.5567, std 0.0108, bias -0.45/-0.45; online_500: L2 0.0281, std 0.0100, bias -0.96/-0.93; online_1000: L2 0.0160, std 0.0100, bias -0.86/-0.91; online_1500: L2 0.0029, std 0.0100, bias -1.08/-1.10; online_2000: L2 0.0120, std 0.0100, bias -1.14/-1.19; online_2500: L2 0.0046, std 0.0100, bias -1.21/-1.24; online_3000: L2 0.0277, std 0.0100, bias -1.38/-1.35; online_3500: L2 0.0038, std 0.0100, bias -1.45/-1.43; online_4000: L2 0.0055, std 0.0100, bias -1.44/-1.46.

## Frozen evaluations (in-run; 3 stochastic seeds per prefix)

| Checkpoint | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch | Cold det / stoch |
| --- | --- | --- | --- | --- |
| pretraining_baseline | fail / 0/3 | fail / 0/3 | fail / 2/3 | fail |
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
