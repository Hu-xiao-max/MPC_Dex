# Variant 61 `chain_p3_cont_from59`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_controls": 3, "prefix_set": [3, 4, 5, 6, 7, 8], "demonstration_include_mpc": false, "demonstration_recorded_rows": [3, 9], "actor_bc_lambda": 100.0, "actor_bc_rows": "first_k", "handoff_noise_sigma": 0.015, "handoff_noise_prefixes": [4, 5, 6, 7, 8], "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "initial_from": ["bootstrap_variant_0059_chain_p3_pinned_lowlr_from57", "online_2000"], "actor_bc_first_rows": 4, "actor_lr_override": 0.0001, "eval_stochastic_trials": 10, "actor_bc_std_target": 0.01}`. Inherited checkpoint: `bootstrap_variant_0059_chain_p3_pinned_lowlr_from57/checkpoints/online_2000`. Hypothesis: Continue the prefix-3 stage for another 2,000 updates from its final checkpoint (prefix-3 0-2/10 in-run, later prefixes retained), since prefix 4 needed more than one stage-length to appear.

Stage status `complete_sac_stage` (exit=0, 803 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 3000, 'online_updates': 12000}; 500 critic-pretraining and 2000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: overridden to 0.0001 for this stage (inherited 0.0001; ramp fraction unchanged). Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over the first 4 rows of each demonstration episode, added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.01)) added to the same actor step. Handoff noise: when sigma > 0 and the sampled prefix is listed, the final prefix control is executed as recorded action + N(0, sigma^2) per dimension in normalized action space, clipped to [-1, 1], from the bit-exact recorded state before it; it is a non-policy row, never appended to any buffer, and its episode is counted under key "<prefix>+noise".

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'3': 54, '4+noise': 51, '5+noise': 41, '6+noise': 30, '7+noise': 43, '8+noise': 37}, force-qualified successes {'3': 11, '4+noise': 4, '5+noise': 3, '6+noise': 15, '7+noise': 16, '8+noise': 28} (informational; the gate is the 6-row demonstration buffer).
- Online: 2000 interactions; completed episodes by prefix {'3': 89, '4+noise': 101, '5+noise': 97, '6+noise': 86, '7+noise': 100, '8+noise': 100}, successes {'3': 5, '4+noise': 1, '5+noise': 10, '6+noise': 28, '7+noise': 60, '8+noise': 67}; outcomes {'drop': 271, 'palm_support': 128, 'success': 171, 'truncated': 3}; handoffs terminated before the policy 24; unfinished last suffix True. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 4525; none entered replay.
- Actor BC loss first/last: 0.0001367 / 2.981e-05; spread-anchor loss first/last: 6.06e-06 / 2.914e-06; alpha raw-equivalent last 0.1747.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 0.0122, std 0.0100, bias +0.06/-0.15; after_pretraining: L2 0.0122, std 0.0100, bias +0.03/+0.01; online_250: L2 0.0140, std 0.0100, bias +0.06/-0.10; online_500: L2 0.0128, std 0.0100, bias -0.14/-0.09; online_750: L2 0.0125, std 0.0100, bias -0.19/-0.09; online_1000: L2 0.0157, std 0.0100, bias -0.17/-0.08; online_1250: L2 0.0155, std 0.0100, bias -0.18/-0.10; online_1500: L2 0.0221, std 0.0100, bias -0.09/-0.13; online_1750: L2 0.0061, std 0.0100, bias +0.01/-0.02; online_2000: L2 0.0088, std 0.0100, bias -0.01/+0.00.

## Frozen evaluations (in-run; 10 stochastic seeds per prefix)

| Checkpoint | Prefix-3 det / stoch | Prefix-4 det / stoch | Prefix-5 det / stoch | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch | Prefix-4 noisy | Prefix-5 noisy | Prefix-6 noisy | Prefix-7 noisy | Prefix-8 noisy | Cold det / stoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pretraining_baseline | fail / 0/10 | pass / 3/10 | pass / 6/10 | pass / 9/10 | pass / 10/10 | pass / 10/10 | 0/5 | 1/5 | 0/5 | 4/5 | 4/5 | fail |
| online_250 | fail / 0/10 | fail / 0/10 | fail / 2/10 | fail / 6/10 | pass / 10/10 | pass / 10/10 | 0/5 | 0/5 | 4/5 | 4/5 | 4/5 | fail |
| online_500 | fail / 0/10 | fail / 0/10 | fail / 0/10 | fail / 2/10 | pass / 10/10 | pass / 10/10 | 0/5 | 0/5 | 2/5 | 2/5 | 4/5 | fail |
| online_750 | fail / 0/10 | fail / 0/10 | pass / 4/10 | fail / 9/10 | pass / 10/10 | pass / 10/10 | 0/5 | 0/5 | 0/5 | 1/5 | 4/5 | fail |
| online_1000 | fail / 0/10 | fail / 1/10 | fail / 5/10 | pass / 10/10 | pass / 10/10 | pass / 10/10 | 0/5 | 0/5 | 2/5 | 4/5 | 3/5 | fail |
| online_1250 | fail / 0/10 | fail / 0/10 | pass / 0/10 | pass / 4/10 | pass / 10/10 | pass / 10/10 | 0/5 | 0/5 | 3/5 | 3/5 | 4/5 | fail |
| online_1500 | fail / 1/10 | pass / 4/10 | fail / 6/10 | pass / 9/10 | pass / 10/10 | pass / 10/10 | 0/5 | 1/5 | 2/5 | 3/5 | 4/5 | fail |
| online_1750 | pass / 3/10 | pass / 6/10 | pass / 6/10 | pass / 9/10 | pass / 10/10 | pass / 10/10 | 0/5 | 0/5 | 2/5 | 3/5 | 4/5 | fail |
| online_2000 | fail / 2/10 | fail / 4/10 | pass / 6/10 | pass / 10/10 | pass / 10/10 | pass / 10/10 | 1/5 | 0/5 | 1/5 | 4/5 | 4/5 | fail / 0/5 |

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

| Checkpoint | Prefix 5 | Prefix 6 | Prefix 7 | Prefix 8 | Cold | Prefix-6 Wilson 95% |
| --- | --- | --- | --- | --- | --- | --- |
| online_250 | 7/30 | 21/30 | 29/30 D | 30/30 D | 0/30 | [0.52, 0.83] |
| online_500 | 4/30 | 6/30 | 30/30 D | 30/30 D | 0/30 | [0.10, 0.37] |
| online_750 | 10/30 D | 25/30 | 30/30 D | 30/30 D | 0/30 | [0.66, 0.93] |
| online_1000 | 17/30 | 29/30 D | 29/30 D | 30/30 D | 0/30 | [0.83, 0.99] |
| online_1250 | 10/30 D | 17/30 D | 30/30 D | 30/30 D | 0/30 | [0.39, 0.73] |
| online_1500 | 9/30 | 29/30 D | 30/30 D | 30/30 D | 0/30 | [0.83, 0.99] |
| online_1750 | 16/30 D | 28/30 D | 30/30 D | 30/30 D | 0/30 | [0.79, 0.98] |
| online_2000 | 14/30 D | 29/30 D | 30/30 D | 30/30 D | 0/30 | [0.83, 0.99] |

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
