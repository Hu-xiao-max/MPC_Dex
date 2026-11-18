# Variant 50 `chain_p5_pinned_noise015_r2`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_controls": 5, "prefix_set": [5, 6, 7, 8], "demonstration_include_mpc": false, "demonstration_recorded_rows": [5, 9], "actor_bc_lambda": 100.0, "actor_bc_rows": "first_k", "handoff_noise_sigma": 0.015, "handoff_noise_prefixes": [6, 7, 8], "seed_matched": false, "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "initial_from": ["bootstrap_variant_0039_bc_first_l100_std01", "online_2000"], "actor_bc_first_rows": 2, "eval_stochastic_trials": 10, "actor_bc_std_target": 0.01}`. Inherited checkpoint: `bootstrap_variant_0039_bc_first_l100_std01/checkpoints/online_2000`. Hypothesis: Seed replicate of chain_p5_pinned_noise015_from39.

Stage status `complete_sac_stage` (exit=0, 589 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 1500, 'online_updates': 6000}; 500 critic-pretraining and 2000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: inherited 3e-4. Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over the first 2 rows of each demonstration episode, added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.01)) added to the same actor step. Handoff noise: when sigma > 0 and the sampled prefix is listed, the final prefix control is executed as recorded action + N(0, sigma^2) per dimension in normalized action space, clipped to [-1, 1], from the bit-exact recorded state before it; it is a non-policy row, never appended to any buffer, and its episode is counted under key "<prefix>+noise".

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'5': 60, '6+noise': 61, '7+noise': 82, '8+noise': 53}, force-qualified successes {'5': 4, '6+noise': 18, '7+noise': 45, '8+noise': 33} (informational; the gate is the 4-row demonstration buffer).
- Online: 2000 interactions; completed episodes by prefix {'5': 153, '6+noise': 156, '7+noise': 130, '8+noise': 141}, successes {'5': 43, '6+noise': 40, '7+noise': 53, '8+noise': 80}; outcomes {'drop': 303, 'success': 216, 'palm_support': 61}; handoffs terminated before the policy 0; unfinished last suffix False. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 5403; none entered replay.
- Actor BC loss first/last: 0.03805 / 4.435e-05; spread-anchor loss first/last: 0.3313 / 1.464e-06; alpha raw-equivalent last 0.0953.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 0.4777, std 0.0225, bias -0.46/-0.43; after_pretraining: L2 0.4777, std 0.0225, bias +0.00/+0.01; online_250: L2 0.0244, std 0.0100, bias +0.16/+0.10; online_500: L2 0.0053, std 0.0100, bias +0.05/+0.06; online_750: L2 0.0207, std 0.0100, bias -0.01/+0.10; online_1000: L2 0.0270, std 0.0100, bias +0.15/+0.19; online_1250: L2 0.0336, std 0.0100, bias +0.03/+0.06; online_1500: L2 0.0238, std 0.0100, bias +0.12/+0.06; online_1750: L2 0.0235, std 0.0100, bias +0.05/-0.03; online_2000: L2 0.0179, std 0.0100, bias +0.03/+0.12.

## Frozen evaluations (in-run; 10 stochastic seeds per prefix)

| Checkpoint | Prefix-5 det / stoch | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch | Prefix-6 noisy | Prefix-7 noisy | Prefix-8 noisy | Cold det / stoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pretraining_baseline | fail / 0/10 | pass / 10/10 | pass / 10/10 | pass / 10/10 | 0/5 | 2/5 | 3/5 | fail |
| online_250 | fail / 1/10 | pass / 7/10 | pass / 10/10 | pass / 10/10 | 1/5 | 2/5 | 3/5 | fail |
| online_500 | pass / 3/10 | fail / 5/10 | pass / 10/10 | pass / 10/10 | 1/5 | 1/5 | 3/5 | fail |
| online_750 | fail / 2/10 | fail / 3/10 | fail / 0/10 | pass / 10/10 | 0/5 | 0/5 | 3/5 | fail |
| online_1000 | pass / 3/10 | pass / 8/10 | pass / 9/10 | pass / 10/10 | 0/5 | 2/5 | 3/5 | fail |
| online_1250 | fail / 3/10 | fail / 1/10 | pass / 7/10 | pass / 10/10 | 1/5 | 0/5 | 3/5 | fail |
| online_1500 | pass / 3/10 | pass / 3/10 | pass / 7/10 | pass / 10/10 | 1/5 | 1/5 | 3/5 | fail |
| online_1750 | fail / 3/10 | fail / 2/10 | pass / 10/10 | pass / 10/10 | 2/5 | 1/5 | 3/5 | fail |
| online_2000 | fail / 6/10 | pass / 7/10 | pass / 10/10 | pass / 10/10 | 0/5 | 2/5 | 3/5 | fail / 0/5 |

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

| Checkpoint | Prefix 5 | Prefix 6 | Prefix 7 | Prefix 8 | Cold | Prefix-6 Wilson 95% |
| --- | --- | --- | --- | --- | --- | --- |
| online_250 | 3/30 | 19/30 D | 30/30 D | 30/30 D | 0/30 | [0.46, 0.78] |
| online_500 | 8/30 D | 18/30 | 30/30 D | 30/30 D | 0/30 | [0.42, 0.75] |
| online_750 | 9/30 | 6/30 | 0/30 | 30/30 D | 0/30 | [0.10, 0.37] |
| online_1000 | 12/30 D | 24/30 D | 30/30 D | 30/30 D | 0/30 | [0.63, 0.90] |
| online_1250 | 10/30 | 4/30 | 18/30 D | 30/30 D | 0/30 | [0.05, 0.30] |
| online_1500 | 9/30 D | 16/30 D | 8/30 D | 30/30 D | 0/30 | [0.36, 0.70] |
| online_1750 | 7/30 | 5/30 | 30/30 D | 30/30 D | 0/30 | [0.07, 0.34] |
| online_2000 | 11/30 | 26/30 D | 30/30 D | 30/30 D | 0/30 | [0.70, 0.95] |

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
