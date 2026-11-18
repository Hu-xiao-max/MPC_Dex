# Variant 52 `chain_p4_pinned_noise015_from48`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_controls": 4, "prefix_set": [4, 5, 6, 7, 8], "demonstration_include_mpc": false, "demonstration_recorded_rows": [4, 9], "actor_bc_lambda": 100.0, "actor_bc_rows": "first_k", "handoff_noise_sigma": 0.015, "handoff_noise_prefixes": [5, 6, 7, 8], "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "initial_from": ["bootstrap_variant_0048_chain_p5_pinned_noise015_from39", "online_2000"], "actor_bc_first_rows": 3, "eval_stochastic_trials": 10, "actor_bc_std_target": 0.01}`. Inherited checkpoint: `bootstrap_variant_0048_chain_p5_pinned_noise015_from39/checkpoints/online_2000`. Hypothesis: Next reverse-curriculum step: prefix 4 inherited from the prefix-5 stage (online 2,000: prefix-5 deterministic pass, 3/10 stochastic; prefix 6 9/10; 7-8 10/10), with recorded controls 5-7 pinned at obs4-obs6 and sigma .015 noise on prefixes 5-8.

Stage status `complete_sac_stage` (exit=0, 738 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 2000, 'online_updates': 8000}; 500 critic-pretraining and 2000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: inherited 3e-4. Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over the first 3 rows of each demonstration episode, added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.01)) added to the same actor step. Handoff noise: when sigma > 0 and the sampled prefix is listed, the final prefix control is executed as recorded action + N(0, sigma^2) per dimension in normalized action space, clipped to [-1, 1], from the bit-exact recorded state before it; it is a non-policy row, never appended to any buffer, and its episode is counted under key "<prefix>+noise".

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'4': 63, '5+noise': 56, '6+noise': 46, '7+noise': 47, '8+noise': 44}, force-qualified successes {'4': 0, '5+noise': 2, '6+noise': 8, '7+noise': 18, '8+noise': 26} (informational; the gate is the 5-row demonstration buffer).
- Online: 2000 interactions; completed episodes by prefix {'4': 95, '5+noise': 111, '6+noise': 89, '7+noise': 106, '8+noise': 103}, successes {'4': 10, '5+noise': 3, '6+noise': 20, '7+noise': 30, '8+noise': 71}; outcomes {'drop': 279, 'palm_support': 90, 'success': 134, 'truncated': 1}; handoffs terminated before the policy 23; unfinished last suffix True. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 4530; none entered replay.
- Actor BC loss first/last: 0.01547 / 0.0001008; spread-anchor loss first/last: 0.1288 / 4.545e-06; alpha raw-equivalent last 0.1170.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 0.5313, std 0.0178, bias -0.20/-0.35; after_pretraining: L2 0.5313, std 0.0178, bias +0.02/+0.08; online_250: L2 0.0272, std 0.0100, bias -0.18/-0.23; online_500: L2 0.0095, std 0.0100, bias -0.12/-0.14; online_750: L2 0.0228, std 0.0100, bias +0.07/-0.23; online_1000: L2 0.0380, std 0.0100, bias +0.05/-0.15; online_1250: L2 0.0338, std 0.0100, bias +0.14/-0.04; online_1500: L2 0.0093, std 0.0100, bias +0.01/+0.11; online_1750: L2 0.0275, std 0.0100, bias -0.01/-0.10; online_2000: L2 0.0299, std 0.0100, bias -0.22/-0.13.

## Frozen evaluations (in-run; 10 stochastic seeds per prefix)

| Checkpoint | Prefix-4 det / stoch | Prefix-5 det / stoch | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch | Prefix-5 noisy | Prefix-6 noisy | Prefix-7 noisy | Prefix-8 noisy | Cold det / stoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pretraining_baseline | fail / 0/10 | pass / 3/10 | pass / 9/10 | pass / 10/10 | pass / 10/10 | 0/5 | 2/5 | 2/5 | 4/5 | fail |
| online_250 | fail / 0/10 | fail / 2/10 | fail / 3/10 | fail / 5/10 | pass / 10/10 | 0/5 | 1/5 | 1/5 | 3/5 | fail |
| online_500 | fail / 1/10 | fail / 0/10 | fail / 2/10 | pass / 7/10 | pass / 10/10 | 0/5 | 0/5 | 2/5 | 4/5 | fail |
| online_750 | fail / 0/10 | pass / 1/10 | fail / 3/10 | fail / 0/10 | pass / 10/10 | 0/5 | 0/5 | 0/5 | 4/5 | fail |
| online_1000 | fail / 0/10 | fail / 0/10 | pass / 2/10 | fail / 0/10 | pass / 10/10 | 0/5 | 1/5 | 0/5 | 3/5 | fail |
| online_1250 | fail / 0/10 | fail / 0/10 | pass / 7/10 | pass / 7/10 | pass / 10/10 | 0/5 | 2/5 | 0/5 | 4/5 | fail |
| online_1500 | fail / 2/10 | fail / 1/10 | pass / 4/10 | fail / 2/10 | pass / 10/10 | 0/5 | 2/5 | 2/5 | 3/5 | fail |
| online_1750 | pass / 2/10 | pass / 5/10 | pass / 5/10 | fail / 0/10 | pass / 10/10 | 0/5 | 1/5 | 2/5 | 4/5 | fail |
| online_2000 | fail / 1/10 | fail / 0/10 | fail / 1/10 | fail / 0/10 | pass / 10/10 | 0/5 | 1/5 | 2/5 | 4/5 | fail / 0/5 |

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

| Checkpoint | Prefix 5 | Prefix 6 | Prefix 7 | Prefix 8 | Cold | Prefix-6 Wilson 95% |
| --- | --- | --- | --- | --- | --- | --- |
| online_250 | 8/30 | 10/30 | 13/30 | 30/30 D | 0/30 | [0.19, 0.51] |
| online_500 | 3/30 | 3/30 | 26/30 D | 30/30 D | 0/30 | [0.03, 0.26] |
| online_750 | 3/30 D | 7/30 | 9/30 | 30/30 D | 0/30 | [0.12, 0.41] |
| online_1000 | 2/30 | 5/30 D | 7/30 | 30/30 D | 0/30 | [0.07, 0.34] |
| online_1250 | 3/30 | 13/30 D | 25/30 D | 30/30 D | 0/30 | [0.27, 0.61] |
| online_1500 | 8/30 | 17/30 D | 4/30 | 30/30 D | 0/30 | [0.39, 0.73] |
| online_1750 | 7/30 D | 21/30 D | 2/30 | 30/30 D | 0/30 | [0.52, 0.83] |
| online_2000 | 1/30 | 3/30 | 2/30 | 30/30 D | 0/30 | [0.03, 0.26] |

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
