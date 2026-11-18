# Variant 53 `chain_p5_pinned_w55`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_controls": 5, "prefix_set": [5, 6, 7, 8], "demonstration_include_mpc": false, "demonstration_recorded_rows": [5, 9], "actor_bc_lambda": 100.0, "actor_bc_rows": "first_k", "handoff_noise_sigma": 0.015, "handoff_noise_prefixes": [6, 7, 8], "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "initial_from": ["bootstrap_variant_0039_bc_first_l100_std01", "online_2000"], "actor_bc_first_rows": 2, "eval_stochastic_trials": 10, "actor_bc_std_target": 0.01, "prefix_weights": [0.55, 0.15, 0.15, 0.15]}`. Inherited checkpoint: `bootstrap_variant_0039_bc_first_l100_std01/checkpoints/online_2000`. Hypothesis: More on-policy prefix-5 episodes (55% of draws) so SAC sees more successes and failures at the knife-edge handoff.

Stage status `complete_sac_stage` (exit=0, 635 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 1500, 'online_updates': 6000}; 500 critic-pretraining and 2000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: inherited 3e-4. Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over the first 2 rows of each demonstration episode, added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.01)) added to the same actor step. Handoff noise: when sigma > 0 and the sampled prefix is listed, the final prefix control is executed as recorded action + N(0, sigma^2) per dimension in normalized action space, clipped to [-1, 1], from the bit-exact recorded state before it; it is a non-policy row, never appended to any buffer, and its episode is counted under key "<prefix>+noise".

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'5': 147, '6+noise': 29, '7+noise': 38, '8+noise': 42}, force-qualified successes {'5': 5, '6+noise': 6, '7+noise': 18, '8+noise': 21} (informational; the gate is the 4-row demonstration buffer).
- Online: 2000 interactions; completed episodes by prefix {'5': 257, '6+noise': 74, '7+noise': 69, '8+noise': 66}, successes {'5': 43, '6+noise': 12, '7+noise': 18, '8+noise': 42}; outcomes {'drop': 319, 'palm_support': 32, 'success': 115}; handoffs terminated before the policy 0; unfinished last suffix True. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 4256; none entered replay.
- Actor BC loss first/last: 0.03805 / 9.157e-05; spread-anchor loss first/last: 0.3313 / 2.146e-06; alpha raw-equivalent last 0.0956.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 0.4777, std 0.0225, bias -0.46/-0.43; after_pretraining: L2 0.4777, std 0.0225, bias +0.17/+0.16; online_250: L2 0.0251, std 0.0100, bias -0.15/-0.15; online_500: L2 0.0221, std 0.0100, bias -0.22/-0.07; online_750: L2 0.0173, std 0.0100, bias -0.12/-0.10; online_1000: L2 0.0436, std 0.0100, bias -0.21/-0.26; online_1250: L2 0.0115, std 0.0100, bias -0.11/-0.22; online_1500: L2 0.0130, std 0.0100, bias -0.20/+0.09; online_1750: L2 0.0169, std 0.0100, bias -0.17/-0.05; online_2000: L2 0.0176, std 0.0100, bias +0.00/-0.08.

## Frozen evaluations (in-run; 10 stochastic seeds per prefix)

| Checkpoint | Prefix-5 det / stoch | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch | Prefix-6 noisy | Prefix-7 noisy | Prefix-8 noisy | Cold det / stoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pretraining_baseline | fail / 0/10 | pass / 9/10 | pass / 10/10 | pass / 10/10 | 1/5 | 3/5 | 4/5 | fail |
| online_250 | fail / 0/10 | fail / 0/10 | fail / 0/10 | pass / 10/10 | 0/5 | 0/5 | 3/5 | fail |
| online_500 | fail / 2/10 | fail / 3/10 | pass / 10/10 | pass / 10/10 | 0/5 | 0/5 | 4/5 | fail |
| online_750 | pass / 5/10 | pass / 8/10 | pass / 10/10 | pass / 10/10 | 1/5 | 2/5 | 4/5 | fail |
| online_1000 | fail / 1/10 | pass / 9/10 | pass / 10/10 | pass / 10/10 | 1/5 | 0/5 | 4/5 | fail |
| online_1250 | fail / 3/10 | fail / 4/10 | pass / 10/10 | pass / 10/10 | 1/5 | 1/5 | 3/5 | fail |
| online_1500 | fail / 2/10 | fail / 3/10 | pass / 9/10 | pass / 10/10 | 2/5 | 2/5 | 4/5 | fail |
| online_1750 | fail / 2/10 | fail / 0/10 | fail / 0/10 | pass / 10/10 | 0/5 | 1/5 | 4/5 | fail |
| online_2000 | fail / 1/10 | pass / 2/10 | fail / 4/10 | pass / 10/10 | 0/5 | 0/5 | 4/5 | fail / 0/5 |

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

| Checkpoint | Prefix 5 | Prefix 6 | Prefix 7 | Prefix 8 | Cold | Prefix-6 Wilson 95% |
| --- | --- | --- | --- | --- | --- | --- |
| online_250 | 0/30 | 2/30 | 0/30 | 30/30 D | 0/30 | [0.02, 0.21] |
| online_500 | 3/30 | 8/30 | 29/30 D | 30/30 D | 0/30 | [0.14, 0.44] |
| online_750 | 10/30 D | 27/30 D | 30/30 D | 30/30 D | 0/30 | [0.74, 0.97] |
| online_1000 | 0/30 | 20/30 D | 29/30 D | 30/30 D | 0/30 | [0.49, 0.81] |
| online_1250 | 8/30 | 20/30 | 30/30 D | 30/30 D | 0/30 | [0.49, 0.81] |
| online_1500 | 4/30 | 14/30 | 29/30 D | 30/30 D | 0/30 | [0.30, 0.64] |
| online_1750 | 1/30 | 1/30 | 0/30 | 30/30 D | 0/30 | [0.01, 0.17] |
| online_2000 | 2/30 | 4/30 D | 16/30 | 30/30 D | 0/30 | [0.05, 0.30] |

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
