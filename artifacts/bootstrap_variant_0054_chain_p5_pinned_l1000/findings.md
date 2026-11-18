# Variant 54 `chain_p5_pinned_l1000`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_controls": 5, "prefix_set": [5, 6, 7, 8], "demonstration_include_mpc": false, "demonstration_recorded_rows": [5, 9], "actor_bc_lambda": 1000.0, "actor_bc_rows": "first_k", "handoff_noise_sigma": 0.015, "handoff_noise_prefixes": [6, 7, 8], "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "initial_from": ["bootstrap_variant_0039_bc_first_l100_std01", "online_2000"], "actor_bc_first_rows": 2, "eval_stochastic_trials": 10, "actor_bc_std_target": 0.01}`. Inherited checkpoint: `bootstrap_variant_0039_bc_first_l100_std01/checkpoints/online_2000`. Hypothesis: Ten times stronger pin at obs5/obs6: the recorded control 6 basin is narrower than .01 while the pinned mean sits .01-.03 away.

Stage status `complete_sac_stage` (exit=0, 684 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 1500, 'online_updates': 6000}; 500 critic-pretraining and 2000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: inherited 3e-4. Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over the first 2 rows of each demonstration episode, added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.01)) added to the same actor step. Handoff noise: when sigma > 0 and the sampled prefix is listed, the final prefix control is executed as recorded action + N(0, sigma^2) per dimension in normalized action space, clipped to [-1, 1], from the bit-exact recorded state before it; it is a non-policy row, never appended to any buffer, and its episode is counted under key "<prefix>+noise".

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'5': 82, '6+noise': 64, '7+noise': 58, '8+noise': 52}, force-qualified successes {'5': 5, '6+noise': 12, '7+noise': 28, '8+noise': 37} (informational; the gate is the 4-row demonstration buffer).
- Online: 2000 interactions; completed episodes by prefix {'5': 144, '6+noise': 136, '7+noise': 128, '8+noise': 156}, successes {'5': 50, '6+noise': 30, '7+noise': 44, '8+noise': 76}; outcomes {'drop': 296, 'palm_support': 68, 'success': 200}; handoffs terminated before the policy 0; unfinished last suffix True. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 5302; none entered replay.
- Actor BC loss first/last: 0.03805 / 5.843e-06; spread-anchor loss first/last: 0.3313 / 8.147e-08; alpha raw-equivalent last 0.0955.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 0.4777, std 0.0225, bias -0.46/-0.43; after_pretraining: L2 0.4777, std 0.0225, bias +0.11/+0.05; online_250: L2 0.0058, std 0.0100, bias -0.03/-0.14; online_500: L2 0.0103, std 0.0100, bias -0.13/-0.08; online_750: L2 0.0128, std 0.0100, bias +0.00/+0.10; online_1000: L2 0.0153, std 0.0100, bias +0.04/-0.00; online_1250: L2 0.0093, std 0.0100, bias +0.06/+0.07; online_1500: L2 0.0094, std 0.0100, bias -0.12/-0.10; online_1750: L2 0.0052, std 0.0100, bias +0.09/+0.15; online_2000: L2 0.0086, std 0.0100, bias +0.11/+0.08.

## Frozen evaluations (in-run; 10 stochastic seeds per prefix)

| Checkpoint | Prefix-5 det / stoch | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch | Prefix-6 noisy | Prefix-7 noisy | Prefix-8 noisy | Cold det / stoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pretraining_baseline | fail / 0/10 | pass / 9/10 | pass / 10/10 | pass / 10/10 | 1/5 | 3/5 | 4/5 | fail |
| online_250 | pass / 1/10 | fail / 7/10 | pass / 10/10 | pass / 10/10 | 2/5 | 1/5 | 3/5 | fail |
| online_500 | pass / 5/10 | pass / 7/10 | pass / 10/10 | pass / 10/10 | 2/5 | 2/5 | 4/5 | fail |
| online_750 | pass / 3/10 | pass / 9/10 | pass / 10/10 | pass / 10/10 | 1/5 | 2/5 | 4/5 | fail |
| online_1000 | fail / 3/10 | pass / 6/10 | pass / 10/10 | pass / 10/10 | 2/5 | 3/5 | 4/5 | fail |
| online_1250 | pass / 3/10 | fail / 2/10 | pass / 10/10 | pass / 10/10 | 2/5 | 2/5 | 4/5 | fail |
| online_1500 | fail / 1/10 | fail / 3/10 | pass / 6/10 | pass / 10/10 | 2/5 | 2/5 | 4/5 | fail |
| online_1750 | fail / 1/10 | fail / 5/10 | pass / 10/10 | pass / 10/10 | 1/5 | 3/5 | 4/5 | fail |
| online_2000 | fail / 1/10 | fail / 4/10 | pass / 10/10 | pass / 10/10 | 2/5 | 1/5 | 4/5 | fail / 0/5 |

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

| Checkpoint | Prefix 5 | Prefix 6 | Prefix 7 | Prefix 8 | Cold | Prefix-6 Wilson 95% |
| --- | --- | --- | --- | --- | --- | --- |
| online_250 | 10/30 D | 21/30 | 30/30 D | 30/30 D | 0/30 | [0.52, 0.83] |
| online_500 | 13/30 D | 25/30 D | 30/30 D | 30/30 D | 0/30 | [0.66, 0.93] |
| online_750 | 7/30 D | 22/30 D | 30/30 D | 30/30 D | 0/30 | [0.56, 0.86] |
| online_1000 | 7/30 | 21/30 D | 30/30 D | 30/30 D | 0/30 | [0.52, 0.83] |
| online_1250 | 13/30 D | 19/30 | 30/30 D | 30/30 D | 0/30 | [0.46, 0.78] |
| online_1500 | 7/30 | 10/30 | 20/30 D | 30/30 D | 0/30 | [0.19, 0.51] |
| online_1750 | 4/30 | 16/30 | 26/30 D | 30/30 D | 0/30 | [0.36, 0.70] |
| online_2000 | 6/30 | 13/30 | 29/30 D | 30/30 D | 0/30 | [0.27, 0.61] |

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
