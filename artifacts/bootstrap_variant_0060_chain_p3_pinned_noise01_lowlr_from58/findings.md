# Variant 60 `chain_p3_pinned_noise01_lowlr_from58`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_controls": 3, "prefix_set": [3, 4, 5, 6, 7, 8], "demonstration_include_mpc": false, "demonstration_recorded_rows": [3, 9], "actor_bc_lambda": 100.0, "actor_bc_rows": "first_k", "handoff_noise_sigma": 0.01, "handoff_noise_prefixes": [4, 5, 6, 7, 8], "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "initial_from": ["bootstrap_variant_0058_chain_p4_pinned_noise01_lowlr_from51", "online_2000"], "actor_bc_first_rows": 4, "actor_lr_override": 0.0001, "eval_stochastic_trials": 10, "actor_bc_std_target": 0.01}`. Inherited checkpoint: `bootstrap_variant_0058_chain_p4_pinned_noise01_lowlr_from51/checkpoints/online_2000`. Hypothesis: Prefix 3 inherited from the sigma .01 prefix-4 stage (0058 online_2000 in-run: prefix-4 1/10, prefix-5 deterministic pass and 5/10, prefixes 6-8 10/10).

Stage status `complete_sac_stage` (exit=0, 506 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 2500, 'online_updates': 9750}; 500 critic-pretraining and 2000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: overridden to 0.0001 for this stage (inherited 0.0001; ramp fraction unchanged). Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over the first 4 rows of each demonstration episode, added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.01)) added to the same actor step. Handoff noise: when sigma > 0 and the sampled prefix is listed, the final prefix control is executed as recorded action + N(0, sigma^2) per dimension in normalized action space, clipped to [-1, 1], from the bit-exact recorded state before it; it is a non-policy row, never appended to any buffer, and its episode is counted under key "<prefix>+noise".

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'3': 54, '4+noise': 51, '5+noise': 41, '6+noise': 30, '7+noise': 43, '8+noise': 37}, force-qualified successes {'3': 0, '4+noise': 6, '5+noise': 3, '6+noise': 13, '7+noise': 19, '8+noise': 30} (informational; the gate is the 6-row demonstration buffer).
- Online: 2000 interactions; completed episodes by prefix {'3': 85, '4+noise': 98, '5+noise': 90, '6+noise': 82, '7+noise': 97, '8+noise': 89}, successes {'3': 1, '4+noise': 3, '5+noise': 1, '6+noise': 9, '7+noise': 36, '8+noise': 62}; outcomes {'drop': 335, 'palm_support': 94, 'success': 112}; handoffs terminated before the policy 11; unfinished last suffix False. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 4328; none entered replay.
- Actor BC loss first/last: 0.01466 / 9.645e-05; spread-anchor loss first/last: 0.3891 / 1.046e-06; alpha raw-equivalent last 0.1397.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 0.5706, std 0.0350, bias -0.33/-0.45; after_pretraining: L2 0.5706, std 0.0350, bias -0.10/-0.03; online_250: L2 0.0090, std 0.0100, bias -0.07/-0.09; online_500: L2 0.0190, std 0.0100, bias -0.19/-0.14; online_750: L2 0.0284, std 0.0100, bias -0.19/-0.10; online_1000: L2 0.0301, std 0.0100, bias -0.17/-0.20; online_1250: L2 0.0290, std 0.0100, bias -0.16/-0.10; online_1500: L2 0.0321, std 0.0100, bias -0.25/-0.07; online_1750: L2 0.0330, std 0.0100, bias -0.19/-0.24; online_2000: L2 0.0391, std 0.0100, bias -0.18/-0.15.

## Frozen evaluations (in-run; 10 stochastic seeds per prefix)

| Checkpoint | Prefix-3 det / stoch | Prefix-4 det / stoch | Prefix-5 det / stoch | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch | Prefix-4 noisy | Prefix-5 noisy | Prefix-6 noisy | Prefix-7 noisy | Prefix-8 noisy | Cold det / stoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pretraining_baseline | fail / 0/10 | fail / 1/10 | pass / 5/10 | pass / 10/10 | pass / 10/10 | pass / 10/10 | 0/5 | 0/5 | 0/5 | 2/5 | 3/5 | fail |
| online_250 | fail / 0/10 | fail / 1/10 | fail / 1/10 | fail / 3/10 | pass / 10/10 | pass / 10/10 | 1/5 | 0/5 | 0/5 | 1/5 | 3/5 | fail |
| online_500 | fail / 0/10 | fail / 0/10 | fail / 3/10 | fail / 2/10 | fail / 0/10 | pass / 10/10 | 0/5 | 0/5 | 0/5 | 0/5 | 3/5 | fail |
| online_750 | fail / 0/10 | fail / 2/10 | fail / 0/10 | fail / 7/10 | fail / 1/10 | pass / 10/10 | 0/5 | 0/5 | 0/5 | 1/5 | 3/5 | fail |
| online_1000 | fail / 0/10 | fail / 0/10 | fail / 2/10 | pass / 1/10 | pass / 5/10 | pass / 10/10 | 0/5 | 0/5 | 0/5 | 0/5 | 3/5 | fail |
| online_1250 | fail / 0/10 | fail / 0/10 | fail / 0/10 | fail / 1/10 | pass / 8/10 | pass / 10/10 | 0/5 | 0/5 | 1/5 | 2/5 | 3/5 | fail |
| online_1500 | fail / 0/10 | fail / 1/10 | fail / 1/10 | fail / 1/10 | pass / 2/10 | pass / 10/10 | 0/5 | 0/5 | 1/5 | 0/5 | 3/5 | fail |
| online_1750 | fail / 0/10 | fail / 0/10 | fail / 2/10 | fail / 1/10 | pass / 5/10 | pass / 10/10 | 0/5 | 0/5 | 0/5 | 0/5 | 3/5 | fail |
| online_2000 | fail / 0/10 | fail / 0/10 | fail / 0/10 | fail / 0/10 | fail / 0/10 | pass / 10/10 | 0/5 | 0/5 | 0/5 | 0/5 | 1/5 | fail / 0/5 |

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

| Checkpoint | Prefix 5 | Prefix 6 | Prefix 7 | Prefix 8 | Cold | Prefix-6 Wilson 95% |
| --- | --- | --- | --- | --- | --- | --- |
| online_250 | 8/30 | 14/30 | 30/30 D | 30/30 D | 0/30 | [0.30, 0.64] |
| online_500 | 9/30 | 9/30 | 4/30 | 30/30 D | 0/30 | [0.17, 0.48] |
| online_750 | 4/30 | 11/30 | 0/30 | 30/30 D | 0/30 | [0.22, 0.54] |
| online_1000 | 6/30 | 5/30 D | 17/30 D | 30/30 D | 0/30 | [0.07, 0.34] |
| online_1250 | 1/30 | 4/30 | 26/30 D | 30/30 D | 0/30 | [0.05, 0.30] |
| online_1500 | 3/30 | 2/30 | 11/30 D | 30/30 D | 0/30 | [0.02, 0.21] |
| online_1750 | 5/30 | 6/30 | 13/30 D | 30/30 D | 0/30 | [0.10, 0.37] |
| online_2000 | 0/30 | 0/30 | 5/30 | 30/30 D | 0/30 | [-0.00, 0.11] |

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
