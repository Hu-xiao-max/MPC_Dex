# Variant 62 `chain_p3_std005_l300_from59`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_controls": 3, "prefix_set": [3, 4, 5, 6, 7, 8], "demonstration_include_mpc": false, "demonstration_recorded_rows": [3, 9], "actor_bc_lambda": 300.0, "actor_bc_rows": "first_k", "handoff_noise_sigma": 0.015, "handoff_noise_prefixes": [4, 5, 6, 7, 8], "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "initial_from": ["bootstrap_variant_0059_chain_p3_pinned_lowlr_from57", "online_2000"], "actor_bc_first_rows": 4, "actor_lr_override": 0.0001, "eval_stochastic_trials": 10, "actor_bc_std_target": 0.005}`. Inherited checkpoint: `bootstrap_variant_0059_chain_p3_pinned_lowlr_from57/checkpoints/online_2000`. Hypothesis: Tighter pin (lambda 300, std .005) at obs3-obs6 continuing from the prefix-3 stage: deterministic prefix-3 trials pass at some checkpoints while stochastic ones fail, pointing at spread and mean precision at the knife-edge control 4.

Stage status `complete_sac_stage` (exit=0, 797 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 3000, 'online_updates': 12000}; 500 critic-pretraining and 2000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: overridden to 0.0001 for this stage (inherited 0.0001; ramp fraction unchanged). Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over the first 4 rows of each demonstration episode, added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.005)) added to the same actor step. Handoff noise: when sigma > 0 and the sampled prefix is listed, the final prefix control is executed as recorded action + N(0, sigma^2) per dimension in normalized action space, clipped to [-1, 1], from the bit-exact recorded state before it; it is a non-policy row, never appended to any buffer, and its episode is counted under key "<prefix>+noise".

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'3': 54, '4+noise': 51, '5+noise': 41, '6+noise': 30, '7+noise': 43, '8+noise': 37}, force-qualified successes {'3': 11, '4+noise': 4, '5+noise': 3, '6+noise': 15, '7+noise': 16, '8+noise': 28} (informational; the gate is the 6-row demonstration buffer).
- Online: 2000 interactions; completed episodes by prefix {'3': 85, '4+noise': 98, '5+noise': 89, '6+noise': 81, '7+noise': 96, '8+noise': 86}, successes {'3': 10, '4+noise': 3, '5+noise': 5, '6+noise': 33, '7+noise': 44, '8+noise': 59}; outcomes {'drop': 243, 'palm_support': 137, 'success': 154, 'truncated': 1}; handoffs terminated before the policy 23; unfinished last suffix False. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 4286; none entered replay.
- Actor BC loss first/last: 0.0001367 / 1.503e-05; spread-anchor loss first/last: 0.479 / 2.135e-07; alpha raw-equivalent last 0.1767.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 0.0122, std 0.0100, bias +0.06/-0.15; after_pretraining: L2 0.0122, std 0.0100, bias +0.03/+0.01; online_250: L2 0.0119, std 0.0050, bias -0.15/-0.05; online_500: L2 0.0147, std 0.0050, bias -0.06/-0.13; online_750: L2 0.0174, std 0.0050, bias -0.28/-0.17; online_1000: L2 0.0194, std 0.0050, bias -0.15/-0.08; online_1250: L2 0.0070, std 0.0050, bias -0.18/-0.12; online_1500: L2 0.0140, std 0.0050, bias -0.19/-0.22; online_1750: L2 0.0063, std 0.0050, bias -0.09/-0.10; online_2000: L2 0.0065, std 0.0050, bias -0.06/-0.08.

## Frozen evaluations (in-run; 10 stochastic seeds per prefix)

| Checkpoint | Prefix-3 det / stoch | Prefix-4 det / stoch | Prefix-5 det / stoch | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch | Prefix-4 noisy | Prefix-5 noisy | Prefix-6 noisy | Prefix-7 noisy | Prefix-8 noisy | Cold det / stoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pretraining_baseline | fail / 0/10 | pass / 3/10 | pass / 6/10 | pass / 9/10 | pass / 10/10 | pass / 10/10 | 0/5 | 1/5 | 0/5 | 4/5 | 4/5 | fail |
| online_250 | fail / 0/10 | pass / 1/10 | pass / 2/10 | fail / 1/10 | fail / 10/10 | pass / 10/10 | 0/5 | 0/5 | 0/5 | 3/5 | 4/5 | fail |
| online_500 | fail / 0/10 | pass / 3/10 | pass / 5/10 | pass / 8/10 | pass / 10/10 | pass / 10/10 | 0/5 | 0/5 | 0/5 | 4/5 | 4/5 | fail |
| online_750 | fail / 1/10 | fail / 1/10 | pass / 3/10 | fail / 5/10 | pass / 9/10 | pass / 10/10 | 0/5 | 0/5 | 1/5 | 2/5 | 4/5 | fail |
| online_1000 | fail / 1/10 | fail / 0/10 | fail / 2/10 | pass / 4/10 | pass / 9/10 | pass / 10/10 | 0/5 | 0/5 | 1/5 | 3/5 | 4/5 | fail |
| online_1250 | fail / 0/10 | pass / 5/10 | fail / 3/10 | pass / 10/10 | pass / 10/10 | pass / 10/10 | 0/5 | 1/5 | 2/5 | 2/5 | 4/5 | fail |
| online_1500 | fail / 0/10 | pass / 6/10 | pass / 5/10 | pass / 10/10 | pass / 10/10 | pass / 10/10 | 0/5 | 1/5 | 4/5 | 5/5 | 4/5 | fail |
| online_1750 | fail / 2/10 | pass / 2/10 | fail / 7/10 | pass / 10/10 | pass / 10/10 | pass / 10/10 | 0/5 | 0/5 | 3/5 | 4/5 | 4/5 | fail |
| online_2000 | fail / 1/10 | fail / 5/10 | pass / 8/10 | pass / 10/10 | pass / 10/10 | fail / 10/10 | 0/5 | 1/5 | 1/5 | 5/5 | 4/5 | fail / 0/5 |

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

| Checkpoint | Prefix 5 | Prefix 6 | Prefix 7 | Prefix 8 | Cold | Prefix-6 Wilson 95% |
| --- | --- | --- | --- | --- | --- | --- |
| online_250 | 6/30 D | 7/30 | 30/30 | 30/30 D | 0/30 | [0.12, 0.41] |
| online_500 | 9/30 D | 20/30 D | 29/30 D | 30/30 D | 0/30 | [0.49, 0.81] |
| online_750 | 8/30 D | 15/30 | 30/30 D | 30/30 D | 0/30 | [0.33, 0.67] |
| online_1000 | 11/30 | 14/30 D | 30/30 D | 30/30 D | 0/30 | [0.30, 0.64] |
| online_1250 | 13/30 | 29/30 D | 30/30 D | 30/30 D | 0/30 | [0.83, 0.99] |
| online_1500 | 15/30 D | 28/30 D | 30/30 D | 30/30 D | 0/30 | [0.79, 0.98] |
| online_1750 | 14/30 | 25/30 D | 30/30 D | 30/30 D | 0/30 | [0.66, 0.93] |
| online_2000 | 16/30 D | 30/30 D | 30/30 D | 29/30 | 0/30 | [0.89, 1.00] |

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
