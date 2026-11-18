# Variant 64 `full_pinned_lowlr_from57`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_controls": 0, "prefix_set": [0, 1, 2, 3, 4, 5, 6, 7, 8], "demonstration_include_mpc": false, "demonstration_recorded_rows": [0, 9], "actor_bc_lambda": 100.0, "actor_bc_rows": "first_k", "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "initial_from": ["bootstrap_variant_0057_chain_p4_pinned_lowlr_from56", "online_2000"], "actor_bc_first_rows": 7, "actor_lr_override": 0.0001, "eval_stochastic_trials": 10, "actor_bc_std_target": 0.01}`. Inherited checkpoint: `bootstrap_variant_0057_chain_p4_pinned_lowlr_from56/checkpoints/online_2000`. Hypothesis: As full_pinned_lowlr_noise_from57 without handoff noise.

Stage status `complete_sac_stage` (exit=0, 723 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 2500, 'online_updates': 10000}; 500 critic-pretraining and 2000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: overridden to 0.0001 for this stage (inherited 0.0001; ramp fraction unchanged). Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over the first 7 rows of each demonstration episode, added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.01)) added to the same actor step. Handoff noise: off.

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'0': 40, '1': 33, '2': 32, '3': 31, '4': 17, '5': 23, '6': 32, '7': 19, '8': 29}, force-qualified successes {'0': 0, '1': 0, '2': 0, '3': 0, '4': 6, '5': 15, '6': 32, '7': 19, '8': 29} (informational; the gate is the 9-row demonstration buffer).
- Online: 2000 interactions; completed episodes by prefix {'0': 51, '1': 62, '2': 63, '3': 62, '4': 44, '5': 58, '6': 50, '7': 74, '8': 48}, successes {'0': 0, '1': 0, '2': 2, '3': 4, '4': 12, '5': 33, '6': 38, '7': 72, '8': 48}; outcomes {'palm_support': 124, 'drop': 178, 'success': 209, 'truncated': 1}; handoffs terminated before the policy 0; unfinished last suffix True. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 2979; none entered replay.
- Actor BC loss first/last: 0.2995 / 6.871e-05; spread-anchor loss first/last: 1.115 / 2.175e-06; alpha raw-equivalent last 0.1429.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 1.8323, std 0.0182, bias -0.99/-0.95; after_pretraining: L2 1.8323, std 0.0182, bias -0.34/-0.32; online_250: L2 1.1085, std 0.0101, bias -0.39/-0.37; online_500: L2 0.6185, std 0.0104, bias -0.43/-0.50; online_750: L2 0.0998, std 0.0101, bias -0.40/-0.47; online_1000: L2 0.0460, std 0.0100, bias -0.40/-0.41; online_1250: L2 0.0509, std 0.0100, bias -0.51/-0.53; online_1500: L2 0.0618, std 0.0100, bias -0.49/-0.50; online_1750: L2 0.0488, std 0.0100, bias -0.60/-0.52; online_2000: L2 0.0433, std 0.0100, bias -0.49/-0.48.

## Frozen evaluations (in-run; 10 stochastic seeds per prefix)

| Checkpoint | Prefix-1 det / stoch | Prefix-2 det / stoch | Prefix-3 det / stoch | Prefix-4 det / stoch | Prefix-5 det / stoch | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch | Cold det / stoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pretraining_baseline | fail / 0/10 | fail / 0/10 | fail / 0/10 | pass / 3/10 | pass / 8/10 | pass / 10/10 | pass / 10/10 | pass / 10/10 | fail / 0/5 |
| online_250 | fail / 0/10 | fail / 0/10 | fail / 0/10 | pass / 7/10 | fail / 5/10 | pass / 10/10 | pass / 10/10 | pass / 10/10 | fail / 0/5 |
| online_500 | fail / 0/10 | fail / 0/10 | fail / 1/10 | fail / 4/10 | fail / 4/10 | pass / 10/10 | pass / 10/10 | pass / 10/10 | fail / 0/5 |
| online_750 | fail / 0/10 | fail / 2/10 | fail / 1/10 | fail / 2/10 | fail / 3/10 | pass / 8/10 | pass / 10/10 | pass / 10/10 | fail / 0/5 |
| online_1000 | fail / 0/10 | fail / 0/10 | pass / 0/10 | pass / 8/10 | pass / 5/10 | pass / 6/10 | pass / 10/10 | pass / 10/10 | fail / 0/5 |
| online_1250 | fail / 0/10 | fail / 0/10 | fail / 0/10 | fail / 5/10 | pass / 8/10 | pass / 10/10 | pass / 10/10 | pass / 10/10 | fail / 0/5 |
| online_1500 | fail / 0/10 | fail / 0/10 | fail / 2/10 | fail / 5/10 | fail / 5/10 | pass / 9/10 | pass / 10/10 | pass / 10/10 | fail / 0/5 |
| online_1750 | fail / 0/10 | fail / 0/10 | fail / 1/10 | fail / 5/10 | pass / 5/10 | pass / 10/10 | pass / 10/10 | pass / 10/10 | fail / 0/5 |
| online_2000 | fail / 0/10 | fail / 0/10 | fail / 1/10 | fail / 3/10 | fail / 2/10 | pass / 8/10 | pass / 10/10 | pass / 10/10 | fail / 0/5 |

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

| Checkpoint | Prefix 5 | Prefix 6 | Prefix 7 | Prefix 8 | Cold | Prefix-6 Wilson 95% |
| --- | --- | --- | --- | --- | --- | --- |
| online_250 | 15/30 | 28/30 D | 30/30 D | 30/30 D | 0/30 | [0.79, 0.98] |
| online_500 | 9/30 | 30/30 D | 30/30 D | 30/30 D | 0/30 | [0.89, 1.00] |
| online_750 | 10/30 | 19/30 D | 30/30 D | 30/30 D | 0/30 | [0.46, 0.78] |
| online_1000 | 21/30 D | 23/30 D | 30/30 D | 30/30 D | 0/30 | [0.59, 0.88] |
| online_1250 | 21/30 D | 28/30 D | 30/30 D | 30/30 D | 0/30 | [0.79, 0.98] |
| online_1500 | 16/30 | 27/30 D | 30/30 D | 30/30 D | 0/30 | [0.74, 0.97] |
| online_1750 | 18/30 D | 26/30 D | 30/30 D | 30/30 D | 0/30 | [0.70, 0.95] |
| online_2000 | 12/30 | 27/30 D | 30/30 D | 30/30 D | 0/30 | [0.74, 0.97] |

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
