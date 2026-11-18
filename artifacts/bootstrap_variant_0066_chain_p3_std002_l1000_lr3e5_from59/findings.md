# Variant 66 `chain_p3_std002_l1000_lr3e5_from59`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_controls": 3, "prefix_set": [3, 4, 5, 6, 7, 8], "demonstration_include_mpc": false, "demonstration_recorded_rows": [3, 9], "actor_bc_lambda": 1000.0, "actor_bc_rows": "first_k", "handoff_noise_sigma": 0.015, "handoff_noise_prefixes": [4, 5, 6, 7, 8], "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "initial_from": ["bootstrap_variant_0059_chain_p3_pinned_lowlr_from57", "online_2000"], "actor_bc_first_rows": 4, "actor_lr_override": 3e-05, "eval_stochastic_trials": 10, "actor_bc_std_target": 0.002}`. Inherited checkpoint: `bootstrap_variant_0059_chain_p3_pinned_lowlr_from57/checkpoints/online_2000`. Hypothesis: As chain_p3_std002_l1000_from59 with a ten-times smaller actor step than the default, to hold the pinned means within the sub-.005 basins.

Stage status `complete_sac_stage` (exit=0, 809 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 3000, 'online_updates': 12000}; 500 critic-pretraining and 2000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: overridden to 3e-05 for this stage (inherited 0.0001; ramp fraction unchanged). Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over the first 4 rows of each demonstration episode, added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.002)) added to the same actor step. Handoff noise: when sigma > 0 and the sampled prefix is listed, the final prefix control is executed as recorded action + N(0, sigma^2) per dimension in normalized action space, clipped to [-1, 1], from the bit-exact recorded state before it; it is a non-policy row, never appended to any buffer, and its episode is counted under key "<prefix>+noise".

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'3': 54, '4+noise': 51, '5+noise': 41, '6+noise': 30, '7+noise': 43, '8+noise': 37}, force-qualified successes {'3': 11, '4+noise': 4, '5+noise': 3, '6+noise': 15, '7+noise': 16, '8+noise': 28} (informational; the gate is the 6-row demonstration buffer).
- Online: 2000 interactions; completed episodes by prefix {'3': 75, '4+noise': 89, '5+noise': 76, '6+noise': 62, '7+noise': 76, '8+noise': 68}, successes {'3': 3, '4+noise': 0, '5+noise': 5, '6+noise': 10, '7+noise': 15, '8+noise': 50}; outcomes {'drop': 218, 'palm_support': 144, 'success': 83, 'truncated': 1}; handoffs terminated before the policy 19; unfinished last suffix False. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 3757; none entered replay.
- Actor BC loss first/last: 0.0001367 / 1.681e-06; spread-anchor loss first/last: 2.587 / 1.136e-08; alpha raw-equivalent last 0.1792.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 0.0122, std 0.0100, bias +0.06/-0.15; after_pretraining: L2 0.0122, std 0.0100, bias +0.03/+0.01; online_250: L2 0.0257, std 0.0020, bias -0.24/-0.15; online_500: L2 0.0049, std 0.0020, bias -0.42/-0.42; online_750: L2 0.0077, std 0.0020, bias -0.50/-0.47; online_1000: L2 0.0038, std 0.0020, bias -0.56/-0.58; online_1250: L2 0.0034, std 0.0020, bias -0.58/-0.56; online_1500: L2 0.0040, std 0.0020, bias -0.31/-0.30; online_1750: L2 0.0029, std 0.0020, bias -0.27/-0.30; online_2000: L2 0.0024, std 0.0020, bias -0.32/-0.38.

## Frozen evaluations (in-run; 10 stochastic seeds per prefix)

| Checkpoint | Prefix-3 det / stoch | Prefix-4 det / stoch | Prefix-5 det / stoch | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch | Prefix-4 noisy | Prefix-5 noisy | Prefix-6 noisy | Prefix-7 noisy | Prefix-8 noisy | Cold det / stoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pretraining_baseline | fail / 0/10 | pass / 3/10 | pass / 6/10 | pass / 9/10 | pass / 10/10 | pass / 10/10 | 0/5 | 1/5 | 0/5 | 4/5 | 4/5 | fail |
| online_250 | fail / 0/10 | fail / 1/10 | fail / 1/10 | fail / 1/10 | fail / 0/10 | pass / 10/10 | 0/5 | 0/5 | 1/5 | 0/5 | 4/5 | fail |
| online_500 | fail / 1/10 | pass / 1/10 | pass / 10/10 | pass / 9/10 | pass / 10/10 | pass / 10/10 | 0/5 | 0/5 | 1/5 | 3/5 | 4/5 | fail |
| online_750 | fail / 1/10 | fail / 3/10 | pass / 10/10 | pass / 7/10 | pass / 8/10 | pass / 10/10 | 0/5 | 0/5 | 0/5 | 2/5 | 4/5 | fail |
| online_1000 | fail / 1/10 | fail / 3/10 | pass / 9/10 | pass / 10/10 | pass / 10/10 | pass / 10/10 | 0/5 | 0/5 | 0/5 | 2/5 | 4/5 | fail |
| online_1250 | fail / 1/10 | fail / 3/10 | pass / 4/10 | fail / 5/10 | fail / 6/10 | pass / 10/10 | 0/5 | 0/5 | 0/5 | 2/5 | 4/5 | fail |
| online_1500 | fail / 0/10 | pass / 1/10 | fail / 1/10 | fail / 0/10 | pass / 7/10 | pass / 10/10 | 1/5 | 0/5 | 1/5 | 2/5 | 4/5 | fail |
| online_1750 | fail / 3/10 | pass / 5/10 | fail / 2/10 | fail / 0/10 | pass / 8/10 | pass / 10/10 | 0/5 | 1/5 | 0/5 | 2/5 | 4/5 | fail |
| online_2000 | fail / 1/10 | fail / 1/10 | fail / 3/10 | fail / 2/10 | pass / 7/10 | pass / 10/10 | 0/5 | 0/5 | 0/5 | 2/5 | 4/5 | fail / 0/5 |

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

No large-sample result for this stage at the time of writing.

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
