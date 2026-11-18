# Variant 63 `full_pinned_lowlr_noise_from57`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_controls": 0, "prefix_set": [0, 1, 2, 3, 4, 5, 6, 7, 8], "demonstration_include_mpc": false, "demonstration_recorded_rows": [0, 9], "actor_bc_lambda": 100.0, "actor_bc_rows": "first_k", "handoff_noise_sigma": 0.015, "handoff_noise_prefixes": [4, 5, 6, 7, 8], "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "initial_from": ["bootstrap_variant_0057_chain_p4_pinned_lowlr_from56", "online_2000"], "actor_bc_first_rows": 7, "actor_lr_override": 0.0001, "eval_stochastic_trials": 10, "actor_bc_std_target": 0.01}`. Inherited checkpoint: `bootstrap_variant_0057_chain_p4_pinned_lowlr_from56/checkpoints/online_2000`. Hypothesis: All remaining handoffs at once from the prefix-4 stage: recorded controls 1-7 pinned at obs0-obs6 with the .01 spread anchor, uniform prefixes 0-8 in the stream (cold-start episodes included), sigma .015 noise on prefixes 4-8, actor step 1e-4; the only stage since 0014 that collects cold episodes.

Stage status `complete_sac_stage` (exit=0, 804 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 2500, 'online_updates': 10000}; 500 critic-pretraining and 2000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: overridden to 0.0001 for this stage (inherited 0.0001; ramp fraction unchanged). Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over the first 7 rows of each demonstration episode, added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.01)) added to the same actor step. Handoff noise: when sigma > 0 and the sampled prefix is listed, the final prefix control is executed as recorded action + N(0, sigma^2) per dimension in normalized action space, clipped to [-1, 1], from the bit-exact recorded state before it; it is a non-policy row, never appended to any buffer, and its episode is counted under key "<prefix>+noise".

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'0': 40, '1': 33, '2': 32, '3': 31, '4+noise': 17, '5+noise': 23, '6+noise': 32, '7+noise': 19, '8+noise': 29}, force-qualified successes {'0': 0, '1': 0, '2': 0, '3': 0, '4+noise': 0, '5+noise': 2, '6+noise': 14, '7+noise': 10, '8+noise': 22} (informational; the gate is the 9-row demonstration buffer).
- Online: 2000 interactions; completed episodes by prefix {'0': 48, '1': 60, '2': 58, '3': 56, '4+noise': 41, '5+noise': 46, '6+noise': 46, '7+noise': 64, '8+noise': 41}, successes {'0': 0, '1': 0, '2': 1, '3': 4, '4+noise': 1, '5+noise': 2, '6+noise': 20, '7+noise': 34, '8+noise': 26}; outcomes {'palm_support': 131, 'drop': 239, 'success': 88, 'truncated': 2}; handoffs terminated before the policy 13; unfinished last suffix True. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 2725; none entered replay.
- Actor BC loss first/last: 0.2995 / 7.477e-05; spread-anchor loss first/last: 1.115 / 2.098e-06; alpha raw-equivalent last 0.1429.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 1.8323, std 0.0182, bias -0.99/-0.95; after_pretraining: L2 1.8323, std 0.0182, bias -0.15/-0.06; online_250: L2 1.1177, std 0.0101, bias -0.28/-0.23; online_500: L2 0.6502, std 0.0104, bias -0.42/-0.36; online_750: L2 0.1078, std 0.0101, bias -0.37/-0.33; online_1000: L2 0.0607, std 0.0100, bias -0.27/-0.20; online_1250: L2 0.0594, std 0.0100, bias -0.07/-0.09; online_1500: L2 0.0765, std 0.0101, bias +0.05/-0.00; online_1750: L2 0.0620, std 0.0100, bias -0.22/-0.11; online_2000: L2 0.0454, std 0.0100, bias -0.22/-0.08.

## Frozen evaluations (in-run; 10 stochastic seeds per prefix)

| Checkpoint | Prefix-1 det / stoch | Prefix-2 det / stoch | Prefix-3 det / stoch | Prefix-4 det / stoch | Prefix-5 det / stoch | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch | Prefix-4 noisy | Prefix-5 noisy | Prefix-6 noisy | Prefix-7 noisy | Prefix-8 noisy | Cold det / stoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pretraining_baseline | fail / 0/10 | fail / 0/10 | fail / 0/10 | pass / 3/10 | pass / 8/10 | pass / 10/10 | pass / 10/10 | pass / 10/10 | 1/5 | 0/5 | 2/5 | 5/5 | 4/5 | fail / 0/5 |
| online_250 | fail / 0/10 | fail / 0/10 | fail / 0/10 | fail / 2/10 | pass / 3/10 | pass / 10/10 | pass / 10/10 | pass / 10/10 | 0/5 | 0/5 | 2/5 | 4/5 | 4/5 | fail / 0/5 |
| online_500 | fail / 0/10 | fail / 1/10 | fail / 1/10 | pass / 1/10 | pass / 7/10 | fail / 8/10 | fail / 8/10 | pass / 10/10 | 0/5 | 0/5 | 1/5 | 4/5 | 4/5 | fail / 0/5 |
| online_750 | fail / 0/10 | fail / 0/10 | fail / 0/10 | fail / 1/10 | fail / 2/10 | pass / 7/10 | pass / 10/10 | pass / 10/10 | 0/5 | 0/5 | 3/5 | 4/5 | 4/5 | fail / 0/5 |
| online_1000 | fail / 0/10 | fail / 1/10 | fail / 1/10 | fail / 2/10 | pass / 7/10 | pass / 7/10 | pass / 10/10 | pass / 10/10 | 0/5 | 0/5 | 3/5 | 5/5 | 4/5 | fail / 0/5 |
| online_1250 | fail / 0/10 | fail / 0/10 | fail / 2/10 | pass / 2/10 | fail / 2/10 | fail / 0/10 | fail / 6/10 | pass / 10/10 | 0/5 | 0/5 | 2/5 | 3/5 | 4/5 | fail / 0/5 |
| online_1500 | fail / 0/10 | fail / 0/10 | fail / 0/10 | pass / 2/10 | pass / 2/10 | pass / 3/10 | pass / 9/10 | pass / 10/10 | 0/5 | 0/5 | 1/5 | 2/5 | 4/5 | fail / 0/5 |
| online_1750 | fail / 0/10 | fail / 0/10 | fail / 0/10 | fail / 2/10 | fail / 0/10 | pass / 6/10 | pass / 7/10 | pass / 10/10 | 0/5 | 0/5 | 2/5 | 3/5 | 4/5 | fail / 0/5 |
| online_2000 | fail / 0/10 | fail / 0/10 | fail / 0/10 | pass / 1/10 | fail / 1/10 | pass / 6/10 | pass / 9/10 | pass / 10/10 | 0/5 | 0/5 | 3/5 | 4/5 | 4/5 | fail / 0/5 |

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

| Checkpoint | Prefix 5 | Prefix 6 | Prefix 7 | Prefix 8 | Cold | Prefix-6 Wilson 95% |
| --- | --- | --- | --- | --- | --- | --- |
| online_250 | 12/30 D | 25/30 D | 30/30 D | 30/30 D | 0/30 | [0.66, 0.93] |
| online_500 | 16/30 D | 23/30 | 22/30 | 30/30 D | 0/30 | [0.59, 0.88] |
| online_750 | 18/30 | 22/30 D | 30/30 D | 30/30 D | 0/30 | [0.56, 0.86] |
| online_1000 | 20/30 D | 26/30 D | 30/30 D | 30/30 D | 0/30 | [0.70, 0.95] |
| online_1250 | 6/30 | 11/30 | 14/30 | 30/30 D | 0/30 | [0.22, 0.54] |
| online_1500 | 9/30 D | 15/30 D | 27/30 D | 30/30 D | 0/30 | [0.33, 0.67] |
| online_1750 | 5/30 | 17/30 D | 14/30 D | 30/30 D | 0/30 | [0.39, 0.73] |
| online_2000 | 11/30 | 21/30 D | 28/30 D | 30/30 D | 0/30 | [0.52, 0.83] |

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
