# Variant 65 `chain_p3_std002_l1000_from59`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_controls": 3, "prefix_set": [3, 4, 5, 6, 7, 8], "demonstration_include_mpc": false, "demonstration_recorded_rows": [3, 9], "actor_bc_lambda": 1000.0, "actor_bc_rows": "first_k", "handoff_noise_sigma": 0.015, "handoff_noise_prefixes": [4, 5, 6, 7, 8], "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "initial_from": ["bootstrap_variant_0059_chain_p3_pinned_lowlr_from57", "online_2000"], "actor_bc_first_rows": 4, "actor_lr_override": 0.0001, "eval_stochastic_trials": 10, "actor_bc_std_target": 0.002}`. Inherited checkpoint: `bootstrap_variant_0059_chain_p3_pinned_lowlr_from57/checkpoints/online_2000`. Hypothesis: Near-deterministic early handoffs: the basin probe with the 0059 policy shows L2 .005 deviations of recorded controls 4 and 5 already fail 5/8 and 2/8, so the spread anchor moves to .002 (L2 about .008) and the pin to lambda 1,000 at obs3-obs6.

Stage status `complete_sac_stage` (exit=0, 794 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 3000, 'online_updates': 12000}; 500 critic-pretraining and 2000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: overridden to 0.0001 for this stage (inherited 0.0001; ramp fraction unchanged). Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over the first 4 rows of each demonstration episode, added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.002)) added to the same actor step. Handoff noise: when sigma > 0 and the sampled prefix is listed, the final prefix control is executed as recorded action + N(0, sigma^2) per dimension in normalized action space, clipped to [-1, 1], from the bit-exact recorded state before it; it is a non-policy row, never appended to any buffer, and its episode is counted under key "<prefix>+noise".

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'3': 54, '4+noise': 51, '5+noise': 41, '6+noise': 30, '7+noise': 43, '8+noise': 37}, force-qualified successes {'3': 11, '4+noise': 4, '5+noise': 3, '6+noise': 15, '7+noise': 16, '8+noise': 28} (informational; the gate is the 6-row demonstration buffer).
- Online: 2000 interactions; completed episodes by prefix {'3': 77, '4+noise': 95, '5+noise': 79, '6+noise': 72, '7+noise': 90, '8+noise': 76}, successes {'3': 11, '4+noise': 2, '5+noise': 4, '6+noise': 18, '7+noise': 29, '8+noise': 57}; outcomes {'drop': 218, 'success': 121, 'palm_support': 149, 'truncated': 1}; handoffs terminated before the policy 21; unfinished last suffix False. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 4024; none entered replay.
- Actor BC loss first/last: 0.0001367 / 2.635e-06; spread-anchor loss first/last: 2.587 / 1.813e-08; alpha raw-equivalent last 0.1792.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 0.0122, std 0.0100, bias +0.06/-0.15; after_pretraining: L2 0.0122, std 0.0100, bias +0.03/+0.01; online_250: L2 0.0068, std 0.0020, bias -0.22/-0.09; online_500: L2 0.0062, std 0.0020, bias -0.45/-0.44; online_750: L2 0.0038, std 0.0020, bias -0.45/-0.44; online_1000: L2 0.0036, std 0.0020, bias -0.39/-0.33; online_1250: L2 0.0039, std 0.0020, bias -0.21/-0.24; online_1500: L2 0.0097, std 0.0020, bias -0.12/-0.19; online_1750: L2 0.0024, std 0.0020, bias -0.15/-0.15; online_2000: L2 0.0010, std 0.0020, bias +0.04/-0.05.

## Frozen evaluations (in-run; 10 stochastic seeds per prefix)

| Checkpoint | Prefix-3 det / stoch | Prefix-4 det / stoch | Prefix-5 det / stoch | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch | Prefix-4 noisy | Prefix-5 noisy | Prefix-6 noisy | Prefix-7 noisy | Prefix-8 noisy | Cold det / stoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pretraining_baseline | fail / 0/10 | pass / 3/10 | pass / 6/10 | pass / 9/10 | pass / 10/10 | pass / 10/10 | 0/5 | 1/5 | 0/5 | 4/5 | 4/5 | fail |
| online_250 | fail / 0/10 | fail / 0/10 | pass / 4/10 | pass / 3/10 | pass / 10/10 | pass / 10/10 | 0/5 | 0/5 | 0/5 | 3/5 | 4/5 | fail |
| online_500 | fail / 2/10 | fail / 4/10 | pass / 6/10 | pass / 7/10 | pass / 7/10 | pass / 10/10 | 0/5 | 0/5 | 1/5 | 4/5 | 4/5 | fail |
| online_750 | fail / 1/10 | fail / 5/10 | pass / 8/10 | fail / 6/10 | pass / 10/10 | pass / 10/10 | 0/5 | 0/5 | 2/5 | 3/5 | 4/5 | fail |
| online_1000 | fail / 1/10 | fail / 2/10 | fail / 5/10 | fail / 0/10 | fail / 6/10 | pass / 10/10 | 0/5 | 0/5 | 1/5 | 3/5 | 4/5 | fail |
| online_1250 | fail / 3/10 | pass / 3/10 | fail / 3/10 | fail / 0/10 | fail / 0/10 | pass / 10/10 | 0/5 | 0/5 | 1/5 | 4/5 | 4/5 | fail |
| online_1500 | fail / 2/10 | pass / 6/10 | fail / 1/10 | fail / 1/10 | fail / 0/10 | pass / 10/10 | 0/5 | 0/5 | 0/5 | 3/5 | 4/5 | fail |
| online_1750 | fail / 1/10 | pass / 4/10 | fail / 0/10 | fail / 1/10 | fail / 0/10 | pass / 10/10 | 0/5 | 0/5 | 0/5 | 2/5 | 4/5 | fail |
| online_2000 | pass / 2/10 | fail / 2/10 | pass / 6/10 | pass / 10/10 | pass / 10/10 | pass / 10/10 | 0/5 | 0/5 | 1/5 | 3/5 | 4/5 | fail / 0/5 |

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

| Checkpoint | Prefix 5 | Prefix 6 | Prefix 7 | Prefix 8 | Cold | Prefix-6 Wilson 95% |
| --- | --- | --- | --- | --- | --- | --- |
| online_250 | 11/30 D | 11/30 D | 28/30 D | 30/30 D | 0/30 | [0.22, 0.54] |
| online_500 | 17/30 D | 24/30 D | 27/30 D | 30/30 D | 0/30 | [0.63, 0.90] |
| online_750 | 19/30 D | 15/30 | 27/30 D | 30/30 D | 0/30 | [0.33, 0.67] |
| online_1000 | 16/30 | 3/30 | 22/30 | 30/30 D | 0/30 | [0.03, 0.26] |
| online_1250 | 5/30 | 2/30 | 1/30 | 30/30 D | 0/30 | [0.02, 0.21] |
| online_1500 | 4/30 | 5/30 | 1/30 | 30/30 D | 0/30 | [0.07, 0.34] |
| online_1750 | 3/30 | 1/30 | 2/30 | 30/30 D | 0/30 | [0.01, 0.17] |
| online_2000 | 18/30 D | 29/30 D | 30/30 D | 30/30 D | 0/30 | [0.83, 0.99] |

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
