# Variant 58 `chain_p4_pinned_noise01_lowlr_from51`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_controls": 4, "prefix_set": [4, 5, 6, 7, 8], "demonstration_include_mpc": false, "demonstration_recorded_rows": [4, 9], "actor_bc_lambda": 100.0, "actor_bc_rows": "first_k", "handoff_noise_sigma": 0.01, "handoff_noise_prefixes": [5, 6, 7, 8], "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "initial_from": ["bootstrap_variant_0051_chain_p5_pinned_noise01_from39", "online_1750"], "actor_bc_first_rows": 3, "actor_lr_override": 0.0001, "eval_stochastic_trials": 10, "actor_bc_std_target": 0.01}`. Inherited checkpoint: `bootstrap_variant_0051_chain_p5_pinned_noise01_from39/checkpoints/online_1750`. Hypothesis: Prefix 4 inherited from the best 30-trial prefix-5 checkpoint (0051 online_1750: prefix-5 22/30 D, prefix-6 25/30 D, prefix-7/8 30/30), sigma .01 noise on prefixes 5-8, recorded controls 5-7 pinned, smaller actor step against the disruption of later prefixes seen in 0052.

Stage status `complete_sac_stage` (exit=0, 498 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 2000, 'online_updates': 7750}; 500 critic-pretraining and 2000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: overridden to 0.0001 for this stage (inherited 0.0003; ramp fraction unchanged). Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over the first 3 rows of each demonstration episode, added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.01)) added to the same actor step. Handoff noise: when sigma > 0 and the sampled prefix is listed, the final prefix control is executed as recorded action + N(0, sigma^2) per dimension in normalized action space, clipped to [-1, 1], from the bit-exact recorded state before it; it is a non-policy row, never appended to any buffer, and its episode is counted under key "<prefix>+noise".

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'4': 63, '5+noise': 56, '6+noise': 46, '7+noise': 47, '8+noise': 44}, force-qualified successes {'4': 0, '5+noise': 7, '6+noise': 22, '7+noise': 29, '8+noise': 31} (informational; the gate is the 5-row demonstration buffer).
- Online: 2000 interactions; completed episodes by prefix {'4': 102, '5+noise': 116, '6+noise': 97, '7+noise': 112, '8+noise': 115}, successes {'4': 12, '5+noise': 9, '6+noise': 29, '7+noise': 55, '8+noise': 80}; outcomes {'drop': 301, 'palm_support': 55, 'truncated': 1, 'success': 185}; handoffs terminated before the policy 17; unfinished last suffix False. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 4763; none entered replay.
- Actor BC loss first/last: 0.01306 / 3.158e-05; spread-anchor loss first/last: 0.1302 / 4.985e-07; alpha raw-equivalent last 0.1142.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 0.4958, std 0.0179, bias -0.34/-0.42; after_pretraining: L2 0.4958, std 0.0179, bias -0.04/-0.03; online_250: L2 0.0434, std 0.0100, bias -0.04/-0.12; online_500: L2 0.0331, std 0.0100, bias +0.06/-0.16; online_750: L2 0.0165, std 0.0100, bias -0.31/-0.21; online_1000: L2 0.0111, std 0.0100, bias -0.19/-0.17; online_1250: L2 0.0105, std 0.0100, bias -0.11/-0.02; online_1500: L2 0.0147, std 0.0100, bias -0.04/-0.04; online_1750: L2 0.0133, std 0.0100, bias -0.00/+0.09; online_2000: L2 0.0254, std 0.0100, bias -0.07/-0.15.

## Frozen evaluations (in-run; 10 stochastic seeds per prefix)

| Checkpoint | Prefix-4 det / stoch | Prefix-5 det / stoch | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch | Prefix-5 noisy | Prefix-6 noisy | Prefix-7 noisy | Prefix-8 noisy | Cold det / stoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pretraining_baseline | fail / 0/10 | pass / 5/10 | pass / 8/10 | pass / 10/10 | pass / 10/10 | 0/5 | 2/5 | 3/5 | 3/5 | fail |
| online_250 | fail / 0/10 | fail / 3/10 | pass / 5/10 | pass / 10/10 | pass / 10/10 | 0/5 | 2/5 | 2/5 | 3/5 | fail |
| online_500 | fail / 1/10 | fail / 1/10 | pass / 9/10 | pass / 10/10 | pass / 10/10 | 0/5 | 1/5 | 3/5 | 2/5 | fail |
| online_750 | fail / 4/10 | fail / 4/10 | pass / 10/10 | pass / 10/10 | pass / 10/10 | 0/5 | 2/5 | 3/5 | 3/5 | fail |
| online_1000 | fail / 0/10 | fail / 2/10 | pass / 3/10 | pass / 10/10 | pass / 10/10 | 0/5 | 1/5 | 1/5 | 3/5 | fail |
| online_1250 | fail / 1/10 | fail / 1/10 | fail / 6/10 | pass / 10/10 | pass / 10/10 | 0/5 | 0/5 | 1/5 | 3/5 | fail |
| online_1500 | pass / 3/10 | fail / 1/10 | fail / 5/10 | pass / 10/10 | pass / 10/10 | 0/5 | 1/5 | 1/5 | 2/5 | fail |
| online_1750 | fail / 4/10 | pass / 1/10 | fail / 4/10 | pass / 9/10 | pass / 10/10 | 0/5 | 0/5 | 3/5 | 3/5 | fail |
| online_2000 | fail / 1/10 | pass / 5/10 | pass / 10/10 | pass / 10/10 | pass / 10/10 | 0/5 | 0/5 | 2/5 | 3/5 | fail / 0/5 |

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

| Checkpoint | Prefix 5 | Prefix 6 | Prefix 7 | Prefix 8 | Cold | Prefix-6 Wilson 95% |
| --- | --- | --- | --- | --- | --- | --- |
| online_250 | 5/30 | 16/30 D | 30/30 D | 30/30 D | 0/30 | [0.36, 0.70] |
| online_500 | 8/30 | 24/30 D | 28/30 D | 30/30 D | 0/30 | [0.63, 0.90] |
| online_750 | 11/30 | 25/30 D | 30/30 D | 30/30 D | 0/30 | [0.66, 0.93] |
| online_1000 | 5/30 | 17/30 D | 30/30 D | 30/30 D | 0/30 | [0.39, 0.73] |
| online_1250 | 4/30 | 18/30 | 30/30 D | 30/30 D | 0/30 | [0.42, 0.75] |
| online_1500 | 5/30 | 15/30 | 30/30 D | 30/30 D | 0/30 | [0.33, 0.67] |
| online_1750 | 7/30 D | 15/30 | 29/30 D | 30/30 D | 0/30 | [0.33, 0.67] |
| online_2000 | 14/30 D | 22/30 D | 30/30 D | 30/30 D | 0/30 | [0.56, 0.86] |

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
