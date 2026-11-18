# Variant 72 `ms_suffix4to10_from70`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_set": [8], "demonstration_include_mpc": false, "actor_bc_lambda": 100.0, "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "initial_from": ["bootstrap_variant_0070_ms_suffix1to6_from68", "online_500"], "actor_lr_override": 0.0001, "eval_stochastic_trials": 3, "actor_bc_std_target": 0.01, "prefix_sources": ["mpc_cold_demonstrations_0001", 60], "suffix_set": [4, 5, 6, 7, 8, 9, 10], "demonstration_suffix_rows": 10, "eval_source_count": 2, "holdout_source_count": 2}`. Inherited checkpoint: `bootstrap_variant_0070_ms_suffix1to6_from68/checkpoints/online_500`. Hypothesis: Third multi-source stage: suffix lengths 4-10 (demonstrations of at most 10 controls become cold-start episodes), last ten rows pinned, every available demonstration except two held-out ones used for training (the suffix-1-3 checkpoint memorized per-trajectory suffixes: 61/120, 0/120, 3/120 on unseen demonstrations), lighter in-run evaluation (two training sources plus the two held-out, three stochastic trials) with the 30-trial evaluator for the rest. Inherits the suffix-1-6 stage at online 500 (its best in-run point) rather than waiting for its slow completion.

Stage status `complete_sac_stage` (exit=0, 1302 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 4500, 'online_updates': 16250}; 500 critic-pretraining and 2000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: overridden to 0.0001 for this stage (inherited 0.0001; ramp fraction unchanged). Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over all demonstration rows, added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.01)) added to the same actor step. Handoff noise: off.

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'s4': 41, 's5': 51, 's6': 29, 's7': 31, 's8': 26, 's9': 37, 's10': 41}, force-qualified successes {'s4': 5, 's5': 2, 's6': 1, 's7': 0, 's8': 0, 's9': 0, 's10': 1} (informational; the gate is the 350-row demonstration buffer).
- Online: 2000 interactions; completed episodes by prefix {'s4': 31, 's5': 37, 's6': 38, 's7': 30, 's8': 37, 's9': 36, 's10': 40}, successes {'s4': 2, 's5': 3, 's6': 0, 's7': 0, 's8': 0, 's9': 1, 's10': 0}; outcomes {'drop': 128, 'palm_support': 115, 'success': 6}; handoffs terminated before the policy 0; unfinished last suffix True. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 5196; none entered replay.
- Actor BC loss first/last: 1.941 / 0.05689; spread-anchor loss first/last: 0.008568 / 0.0002861; alpha raw-equivalent last 0.2709.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 2.0669, std 0.0100, bias -1.13/-0.95; after_pretraining: L2 2.0669, std 0.0100, bias -0.73/-0.71; online_250: L2 1.3150, std 0.0101, bias -1.29/-1.28; online_500: L2 0.7170, std 0.0101, bias -1.47/-1.48; online_750: L2 0.2755, std 0.0099, bias -1.57/-1.59; online_1000: L2 0.1407, std 0.0097, bias -1.58/-1.58; online_1250: L2 0.0728, std 0.0096, bias -1.41/-1.42; online_1500: L2 0.0420, std 0.0096, bias -1.30/-1.29; online_1750: L2 0.0328, std 0.0098, bias -1.14/-1.15; online_2000: L2 0.0257, std 0.0100, bias -1.03/-1.05.

## Frozen evaluations (in-run; 3 stochastic seeds per prefix)

| Checkpoint | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch | Cold det / stoch |
| --- | --- | --- | --- | --- |
| pretraining_baseline | fail / 0/3 | fail / 0/3 | fail / 0/3 | fail |
| online_250 | fail / 0/3 | fail / 0/3 | pass / 2/3 | fail |
| online_500 | fail / 0/3 | fail / 0/3 | fail / 1/3 | fail |
| online_750 | fail / 0/3 | fail / 1/3 | fail / 3/3 | fail |
| online_1000 | fail / 0/3 | fail / 0/3 | fail / 0/3 | fail |
| online_1250 | fail / 0/3 | fail / 0/3 | fail / 0/3 | fail |
| online_1500 | fail / 0/3 | fail / 0/3 | fail / 1/3 | fail |
| online_1750 | fail / 0/3 | pass / 0/3 | pass / 3/3 | fail |
| online_2000 | fail / 0/3 | fail / 0/3 | pass / 3/3 | fail / 0/5 |

The cold column counts only the cold-reset trials (an earlier rendering of this table had added the multi-source cells to it; corrected 2026-10-03).

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

No large-sample result for this stage at the time of writing.

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
