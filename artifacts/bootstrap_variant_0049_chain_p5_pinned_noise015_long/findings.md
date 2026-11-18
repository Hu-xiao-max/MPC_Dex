# Variant 49 `chain_p5_pinned_noise015_long`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_controls": 5, "prefix_set": [5, 6, 7, 8], "demonstration_include_mpc": false, "demonstration_recorded_rows": [5, 9], "actor_bc_lambda": 100.0, "actor_bc_rows": "first_k", "handoff_noise_sigma": 0.015, "handoff_noise_prefixes": [6, 7, 8], "online_policy_interactions": 4000, "checkpoints": [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], "initial_from": ["bootstrap_variant_0039_bc_first_l100_std01", "online_2000"], "actor_bc_first_rows": 2, "eval_stochastic_trials": 10, "actor_bc_std_target": 0.01}`. Inherited checkpoint: `bootstrap_variant_0039_bc_first_l100_std01/checkpoints/online_2000`. Hypothesis: chain_p5_pinned_noise015_from39 (prefix-5 deterministic passes at four checkpoints, stochastic up to 5/10, prefixes 6-8 retained) with twice the budget: the pinned prefix-6 recipe improved monotonically to 4,000 updates.

Stage status `complete_sac_stage` (exit=0, 890 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 1500, 'online_updates': 8000}; 500 critic-pretraining and 4000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: inherited 3e-4. Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over the first 2 rows of each demonstration episode, added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.01)) added to the same actor step. Handoff noise: when sigma > 0 and the sampled prefix is listed, the final prefix control is executed as recorded action + N(0, sigma^2) per dimension in normalized action space, clipped to [-1, 1], from the bit-exact recorded state before it; it is a non-policy row, never appended to any buffer, and its episode is counted under key "<prefix>+noise".

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'5': 82, '6+noise': 64, '7+noise': 58, '8+noise': 52}, force-qualified successes {'5': 5, '6+noise': 12, '7+noise': 28, '8+noise': 37} (informational; the gate is the 4-row demonstration buffer).
- Online: 4000 interactions; completed episodes by prefix {'5': 281, '6+noise': 265, '7+noise': 283, '8+noise': 280}, successes {'5': 67, '6+noise': 49, '7+noise': 106, '8+noise': 147}; outcomes {'drop': 647, 'palm_support': 92, 'success': 369, 'truncated': 1}; handoffs terminated before the policy 0; unfinished last suffix True. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 8839; none entered replay.
- Actor BC loss first/last: 0.03805 / 8.877e-05; spread-anchor loss first/last: 0.3313 / 2.077e-06; alpha raw-equivalent last 0.1168.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 0.4777, std 0.0225, bias -0.46/-0.43; after_pretraining: L2 0.4777, std 0.0225, bias +0.11/+0.05; online_500: L2 0.0114, std 0.0100, bias -0.11/-0.16; online_1000: L2 0.0140, std 0.0100, bias -0.11/-0.20; online_1500: L2 0.0364, std 0.0100, bias +0.08/+0.03; online_2000: L2 0.0173, std 0.0100, bias +0.14/-0.01; online_2500: L2 0.0241, std 0.0100, bias +0.22/+0.15; online_3000: L2 0.0149, std 0.0100, bias +0.09/+0.09; online_3500: L2 0.0119, std 0.0100, bias +0.10/+0.07; online_4000: L2 0.0207, std 0.0100, bias +0.24/+0.25.

## Frozen evaluations (in-run; 10 stochastic seeds per prefix)

| Checkpoint | Prefix-5 det / stoch | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch | Prefix-6 noisy | Prefix-7 noisy | Prefix-8 noisy | Cold det / stoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pretraining_baseline | fail / 0/10 | pass / 9/10 | pass / 10/10 | pass / 10/10 | 1/5 | 3/5 | 4/5 | fail |
| online_500 | fail / 1/10 | pass / 1/10 | fail / 0/10 | pass / 10/10 | 1/5 | 0/5 | 4/5 | fail |
| online_1000 | pass / 4/10 | fail / 5/10 | pass / 10/10 | pass / 10/10 | 1/5 | 3/5 | 4/5 | fail |
| online_1500 | fail / 2/10 | pass / 8/10 | pass / 10/10 | pass / 10/10 | 2/5 | 2/5 | 4/5 | fail |
| online_2000 | pass / 3/10 | pass / 9/10 | pass / 10/10 | pass / 10/10 | 2/5 | 2/5 | 4/5 | fail |
| online_2500 | fail / 4/10 | pass / 7/10 | pass / 10/10 | pass / 10/10 | 0/5 | 2/5 | 4/5 | fail |
| online_3000 | fail / 5/10 | fail / 6/10 | pass / 10/10 | pass / 10/10 | 2/5 | 3/5 | 4/5 | fail |
| online_3500 | fail / 6/10 | pass / 9/10 | pass / 10/10 | pass / 10/10 | 0/5 | 3/5 | 3/5 | fail |
| online_4000 | fail / 1/10 | fail / 4/10 | pass / 10/10 | pass / 10/10 | 0/5 | 3/5 | 4/5 | fail / 0/5 |

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

| Checkpoint | Prefix 5 | Prefix 6 | Prefix 7 | Prefix 8 | Cold | Prefix-6 Wilson 95% |
| --- | --- | --- | --- | --- | --- | --- |
| online_500 | 2/30 | 7/30 D | 0/30 | 30/30 D | 0/30 | [0.12, 0.41] |
| online_1000 | 10/30 D | 12/30 | 29/30 D | 30/30 D | 0/30 | [0.25, 0.58] |
| online_1500 | 6/30 | 22/30 D | 29/30 D | 30/30 D | 0/30 | [0.56, 0.86] |
| online_2000 | 11/30 D | 26/30 D | 30/30 D | 30/30 D | 0/30 | [0.70, 0.95] |
| online_2500 | 3/30 | 11/30 D | 30/30 D | 30/30 D | 0/30 | [0.22, 0.54] |
| online_3000 | 8/30 | 19/30 | 29/30 D | 30/30 D | 0/30 | [0.46, 0.78] |
| online_3500 | 5/30 | 27/30 D | 30/30 D | 30/30 D | 0/30 | [0.74, 0.97] |
| online_4000 | 10/30 | 15/30 | 30/30 D | 30/30 D | 0/30 | [0.33, 0.67] |

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
