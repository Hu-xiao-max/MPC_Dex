# Variant 67 `chain_p3_cont2_from61`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_controls": 3, "prefix_set": [3, 4, 5, 6, 7, 8], "demonstration_include_mpc": false, "demonstration_recorded_rows": [3, 9], "actor_bc_lambda": 100.0, "actor_bc_rows": "first_k", "handoff_noise_sigma": 0.015, "handoff_noise_prefixes": [4, 5, 6, 7, 8], "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "initial_from": ["bootstrap_variant_0061_chain_p3_cont_from59", "online_1750"], "actor_bc_first_rows": 4, "actor_lr_override": 0.0001, "eval_stochastic_trials": 10, "actor_bc_std_target": 0.01}`. Inherited checkpoint: `bootstrap_variant_0061_chain_p3_cont_from59/checkpoints/online_1750`. Hypothesis: Third stage-length at prefix 3 from the best 30-trial prefix-3 checkpoint so far (0061 online_1750: prefix-3 8/30 D, prefix-4 16/30 D, prefix-5 16/30, prefix-6 28/30, prefixes 7-8 30/30): prefix 3 rose from 3/30 to 8/30 over one continuation.

Stage status `complete_sac_stage` (exit=0, 1368 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 3500, 'online_updates': 13750}; 500 critic-pretraining and 2000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: overridden to 0.0001 for this stage (inherited 0.0001; ramp fraction unchanged). Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over the first 4 rows of each demonstration episode, added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.01)) added to the same actor step. Handoff noise: when sigma > 0 and the sampled prefix is listed, the final prefix control is executed as recorded action + N(0, sigma^2) per dimension in normalized action space, clipped to [-1, 1], from the bit-exact recorded state before it; it is a non-policy row, never appended to any buffer, and its episode is counted under key "<prefix>+noise".

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'3': 54, '4+noise': 51, '5+noise': 41, '6+noise': 30, '7+noise': 43, '8+noise': 37}, force-qualified successes {'3': 10, '4+noise': 2, '5+noise': 4, '6+noise': 16, '7+noise': 26, '8+noise': 29} (informational; the gate is the 6-row demonstration buffer).
- Online: 2000 interactions; completed episodes by prefix {'3': 90, '4+noise': 103, '5+noise': 98, '6+noise': 88, '7+noise': 103, '8+noise': 100}, successes {'3': 13, '4+noise': 10, '5+noise': 11, '6+noise': 38, '7+noise': 63, '8+noise': 77}; outcomes {'drop': 254, 'palm_support': 113, 'success': 212, 'truncated': 3}; handoffs terminated before the policy 24; unfinished last suffix True. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 4575; none entered replay.
- Actor BC loss first/last: 3.832e-05 / 5.559e-05; spread-anchor loss first/last: 3.127e-06 / 1.884e-06; alpha raw-equivalent last 0.2080.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 0.0061, std 0.0100, bias +0.01/-0.02; after_pretraining: L2 0.0061, std 0.0100, bias +0.13/+0.01; online_250: L2 0.0274, std 0.0100, bias -0.13/-0.11; online_500: L2 0.0143, std 0.0100, bias +0.02/+0.00; online_750: L2 0.0153, std 0.0100, bias +0.30/+0.14; online_1000: L2 0.0052, std 0.0100, bias +0.20/+0.17; online_1250: L2 0.0094, std 0.0100, bias -0.11/-0.07; online_1500: L2 0.0061, std 0.0100, bias -0.06/-0.18; online_1750: L2 0.0064, std 0.0100, bias -0.15/-0.14; online_2000: L2 0.0071, std 0.0100, bias -0.21/-0.15.

## Frozen evaluations (in-run; 10 stochastic seeds per prefix)

| Checkpoint | Prefix-3 det / stoch | Prefix-4 det / stoch | Prefix-5 det / stoch | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch | Prefix-4 noisy | Prefix-5 noisy | Prefix-6 noisy | Prefix-7 noisy | Prefix-8 noisy | Cold det / stoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pretraining_baseline | pass / 3/10 | pass / 6/10 | pass / 6/10 | pass / 9/10 | pass / 10/10 | pass / 10/10 | 0/5 | 0/5 | 2/5 | 3/5 | 4/5 | fail |
| online_250 | pass / 0/10 | fail / 0/10 | fail / 2/10 | pass / 8/10 | pass / 9/10 | pass / 10/10 | 0/5 | 0/5 | 3/5 | 5/5 | 4/5 | fail |
| online_500 | fail / 0/10 | pass / 0/10 | fail / 1/10 | fail / 2/10 | pass / 10/10 | pass / 10/10 | 0/5 | 0/5 | 0/5 | 3/5 | 4/5 | fail |
| online_750 | pass / 0/10 | fail / 6/10 | fail / 4/10 | pass / 10/10 | pass / 10/10 | pass / 10/10 | 0/5 | 0/5 | 4/5 | 5/5 | 4/5 | fail |
| online_1000 | fail / 0/10 | fail / 6/10 | pass / 5/10 | pass / 9/10 | pass / 10/10 | pass / 10/10 | 2/5 | 0/5 | 3/5 | 5/5 | 4/5 | fail |
| online_1250 | fail / 1/10 | fail / 1/10 | pass / 3/10 | pass / 10/10 | pass / 10/10 | pass / 10/10 | 0/5 | 0/5 | 1/5 | 3/5 | 4/5 | fail |
| online_1500 | fail / 1/10 | pass / 2/10 | pass / 5/10 | pass / 9/10 | pass / 9/10 | pass / 10/10 | 0/5 | 0/5 | 3/5 | 4/5 | 4/5 | fail |
| online_1750 | fail / 3/10 | pass / 3/10 | pass / 5/10 | pass / 10/10 | pass / 10/10 | pass / 10/10 | 0/5 | 0/5 | 4/5 | 5/5 | 4/5 | fail |
| online_2000 | fail / 2/10 | pass / 6/10 | pass / 2/10 | pass / 10/10 | pass / 10/10 | pass / 10/10 | 0/5 | 0/5 | 3/5 | 5/5 | 4/5 | fail / 0/5 |

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

| Checkpoint | Prefix 5 | Prefix 6 | Prefix 7 | Prefix 8 | Cold | Prefix-6 Wilson 95% |
| --- | --- | --- | --- | --- | --- | --- |
| online_250 | 3/30 | 18/30 D | 30/30 D | 30/30 D | 0/30 | [0.42, 0.75] |
| online_500 | 6/30 | 6/30 | 25/30 D | 30/30 D | 0/30 | [0.10, 0.37] |
| online_750 | 13/30 | 29/30 D | 30/30 D | 30/30 D | 0/30 | [0.83, 0.99] |
| online_1000 | 9/30 D | 30/30 D | 29/30 D | 30/30 D | 0/30 | [0.89, 1.00] |
| online_1250 | 9/30 D | 29/30 D | 26/30 D | 30/30 D | 0/30 | [0.83, 0.99] |
| online_1500 | 13/30 D | 27/30 D | 21/30 D | 30/30 D | 0/30 | [0.74, 0.97] |
| online_1750 | 15/30 D | 26/30 D | 30/30 D | 30/30 D | 0/30 | [0.70, 0.95] |
| online_2000 | 10/30 D | 27/30 D | 30/30 D | 30/30 D | 0/30 | [0.74, 0.97] |

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
