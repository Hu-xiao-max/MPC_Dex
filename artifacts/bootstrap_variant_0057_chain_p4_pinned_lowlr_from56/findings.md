# Variant 57 `chain_p4_pinned_lowlr_from56`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_controls": 4, "prefix_set": [4, 5, 6, 7, 8], "demonstration_include_mpc": false, "demonstration_recorded_rows": [4, 9], "actor_bc_lambda": 100.0, "actor_bc_rows": "first_k", "handoff_noise_sigma": 0.015, "handoff_noise_prefixes": [5, 6, 7, 8], "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "initial_from": ["bootstrap_variant_0056_chain_p5_pinned_lowlr", "online_2000"], "actor_bc_first_rows": 3, "actor_lr_override": 0.0001, "eval_stochastic_trials": 10, "actor_bc_std_target": 0.01}`. Inherited checkpoint: `bootstrap_variant_0056_chain_p5_pinned_lowlr/checkpoints/online_2000`. Hypothesis: Prefix 4 inherited from the prefix-5 stage with the smaller actor step (online 2,000 in-run: prefix-5 deterministic pass and 7/10, prefix-6 10/10, 7-8 10/10), pinning recorded controls 5-7 at obs4-obs6, sigma .015 noise on prefixes 5-8, actor step 1e-4.

Stage status `complete_sac_stage` (exit=0, 486 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 2000, 'online_updates': 8000}; 500 critic-pretraining and 2000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: overridden to 0.0001 for this stage (inherited 0.0001; ramp fraction unchanged). Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over the first 3 rows of each demonstration episode, added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.01)) added to the same actor step. Handoff noise: when sigma > 0 and the sampled prefix is listed, the final prefix control is executed as recorded action + N(0, sigma^2) per dimension in normalized action space, clipped to [-1, 1], from the bit-exact recorded state before it; it is a non-policy row, never appended to any buffer, and its episode is counted under key "<prefix>+noise".

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'4': 63, '5+noise': 56, '6+noise': 46, '7+noise': 47, '8+noise': 44}, force-qualified successes {'4': 0, '5+noise': 7, '6+noise': 14, '7+noise': 22, '8+noise': 27} (informational; the gate is the 5-row demonstration buffer).
- Online: 2000 interactions; completed episodes by prefix {'4': 102, '5+noise': 116, '6+noise': 97, '7+noise': 112, '8+noise': 113}, successes {'4': 22, '5+noise': 12, '6+noise': 34, '7+noise': 54, '8+noise': 82}; outcomes {'drop': 265, 'palm_support': 68, 'success': 204, 'truncated': 3}; handoffs terminated before the policy 23; unfinished last suffix False. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 4747; none entered replay.
- Actor BC loss first/last: 0.01447 / 5.963e-05; spread-anchor loss first/last: 0.1278 / 1.907e-06; alpha raw-equivalent last 0.1170.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 0.5045, std 0.0177, bias -0.24/-0.35; after_pretraining: L2 0.5045, std 0.0177, bias -0.03/-0.09; online_250: L2 0.0356, std 0.0100, bias -0.04/-0.13; online_500: L2 0.0436, std 0.0100, bias -0.00/+0.15; online_750: L2 0.0337, std 0.0100, bias +0.22/+0.14; online_1000: L2 0.0299, std 0.0100, bias +0.07/+0.12; online_1250: L2 0.0164, std 0.0100, bias -0.03/+0.04; online_1500: L2 0.0067, std 0.0100, bias -0.00/-0.02; online_1750: L2 0.0229, std 0.0100, bias -0.07/-0.04; online_2000: L2 0.0090, std 0.0100, bias -0.01/-0.11.

## Frozen evaluations (in-run; 10 stochastic seeds per prefix)

| Checkpoint | Prefix-4 det / stoch | Prefix-5 det / stoch | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch | Prefix-5 noisy | Prefix-6 noisy | Prefix-7 noisy | Prefix-8 noisy | Cold det / stoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pretraining_baseline | fail / 0/10 | pass / 7/10 | pass / 10/10 | pass / 10/10 | pass / 10/10 | 0/5 | 1/5 | 4/5 | 4/5 | fail |
| online_250 | fail / 1/10 | fail / 3/10 | pass / 10/10 | pass / 10/10 | pass / 10/10 | 0/5 | 1/5 | 3/5 | 4/5 | fail |
| online_500 | fail / 1/10 | pass / 3/10 | pass / 7/10 | pass / 10/10 | pass / 10/10 | 0/5 | 2/5 | 2/5 | 4/5 | fail |
| online_750 | fail / 1/10 | pass / 2/10 | pass / 5/10 | pass / 10/10 | pass / 10/10 | 0/5 | 2/5 | 5/5 | 4/5 | fail |
| online_1000 | fail / 2/10 | pass / 4/10 | pass / 9/10 | pass / 10/10 | pass / 10/10 | 1/5 | 2/5 | 5/5 | 4/5 | fail |
| online_1250 | fail / 3/10 | fail / 3/10 | pass / 8/10 | pass / 9/10 | pass / 10/10 | 1/5 | 1/5 | 4/5 | 4/5 | fail |
| online_1500 | fail / 4/10 | fail / 3/10 | fail / 5/10 | pass / 10/10 | pass / 10/10 | 0/5 | 4/5 | 3/5 | 4/5 | fail |
| online_1750 | fail / 4/10 | fail / 5/10 | fail / 8/10 | pass / 9/10 | pass / 10/10 | 0/5 | 1/5 | 1/5 | 4/5 | fail |
| online_2000 | pass / 3/10 | pass / 8/10 | pass / 10/10 | pass / 10/10 | pass / 10/10 | 0/5 | 2/5 | 5/5 | 4/5 | fail / 0/5 |

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

| Checkpoint | Prefix 5 | Prefix 6 | Prefix 7 | Prefix 8 | Cold | Prefix-6 Wilson 95% |
| --- | --- | --- | --- | --- | --- | --- |
| online_250 | 20/30 | 23/30 D | 30/30 D | 30/30 D | 0/30 | [0.59, 0.88] |
| online_500 | 7/30 D | 21/30 D | 30/30 D | 30/30 D | 0/30 | [0.52, 0.83] |
| online_750 | 9/30 D | 15/30 D | 30/30 D | 30/30 D | 0/30 | [0.33, 0.67] |
| online_1000 | 12/30 D | 28/30 D | 30/30 D | 30/30 D | 0/30 | [0.79, 0.98] |
| online_1250 | 14/30 | 29/30 D | 30/30 D | 30/30 D | 0/30 | [0.83, 0.99] |
| online_1500 | 4/30 | 10/30 | 30/30 D | 30/30 D | 0/30 | [0.19, 0.51] |
| online_1750 | 12/30 | 22/30 | 30/30 D | 30/30 D | 0/30 | [0.56, 0.86] |
| online_2000 | 16/30 D | 29/30 D | 30/30 D | 30/30 D | 0/30 | [0.83, 0.99] |

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
