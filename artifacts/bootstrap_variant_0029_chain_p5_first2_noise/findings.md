# Variant 29 `chain_p5_first2_noise`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_controls": 5, "prefix_set": [5, 6, 7, 8], "demonstration_include_mpc": false, "demonstration_recorded_rows": [5, 9], "actor_bc_lambda": 10.0, "actor_bc_rows": "first_k", "handoff_noise_sigma": 0.025, "handoff_noise_prefixes": [6, 7, 8], "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "initial_from": ["bootstrap_variant_0004_recorded_bc10_first", "online_2000"], "actor_bc_first_rows": 2, "eval_stochastic_trials": 10}`. Inherited checkpoint: `bootstrap_variant_0004_recorded_bc10_first/checkpoints/online_2000`. Hypothesis: chain_p5_first2 plus sigma .025 handoff noise on prefixes 6-8.

Stage status `complete_sac_stage` (exit=0, 883 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 1500, 'online_updates': 6000}; 500 critic-pretraining and 2000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: inherited 3e-4. Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over the first 2 rows of each demonstration episode, added to the single SAC actor step; spread anchor: none. Handoff noise: when sigma > 0 and the sampled prefix is listed, the final prefix control is executed as recorded action + N(0, sigma^2) per dimension in normalized action space, clipped to [-1, 1], from the bit-exact recorded state before it; it is a non-policy row, never appended to any buffer, and its episode is counted under key "<prefix>+noise".

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'5': 82, '6+noise': 64, '7+noise': 58, '8+noise': 52}, force-qualified successes {'5': 0, '6+noise': 6, '7+noise': 5, '8+noise': 27} (informational; the gate is the 4-row demonstration buffer).
- Online: 2000 interactions; completed episodes by prefix {'5': 128, '6+noise': 112, '7+noise': 96, '8+noise': 110}, successes {'5': 22, '6+noise': 17, '7+noise': 23, '8+noise': 62}; outcomes {'drop': 277, 'success': 124, 'palm_support': 45}; handoffs terminated before the policy 0; unfinished last suffix True. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 4487; none entered replay.
- Actor BC loss first/last: 0.02849 / 0.0001753; spread-anchor loss first/last: none; alpha raw-equivalent last 0.0913.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 0.6237, std 0.0487, bias -0.60/-0.59; after_pretraining: L2 0.6237, std 0.0487, bias +0.05/+0.17; online_250: L2 0.0334, std 0.0825, bias +0.06/+0.08; online_500: L2 0.0162, std 0.1098, bias +0.04/+0.09; online_750: L2 0.0276, std 0.1451, bias +0.05/+0.09; online_1000: L2 0.0296, std 0.1644, bias -0.16/-0.08; online_1250: L2 0.0308, std 0.1939, bias +0.14/+0.00; online_1500: L2 0.0626, std 0.2233, bias +0.21/+0.19; online_1750: L2 0.0420, std 0.2824, bias +0.17/+0.04; online_2000: L2 0.0418, std 0.2874, bias +0.11/+0.17.

## Frozen evaluations (in-run; 10 stochastic seeds per prefix)

| Checkpoint | Prefix-5 det / stoch | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch | Prefix-6 noisy | Prefix-7 noisy | Prefix-8 noisy | Cold det / stoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pretraining_baseline | fail / 0/10 | pass / 6/10 | pass / 10/10 | pass / 10/10 | 0/5 | 0/5 | 2/5 | fail |
| online_250 | fail / 1/10 | fail / 1/10 | fail / 2/10 | pass / 10/10 | 0/5 | 0/5 | 2/5 | fail |
| online_500 | fail / 0/10 | fail / 1/10 | pass / 7/10 | pass / 10/10 | 1/5 | 1/5 | 2/5 | fail |
| online_750 | pass / 0/10 | fail / 1/10 | fail / 2/10 | pass / 10/10 | 2/5 | 2/5 | 2/5 | fail |
| online_1000 | fail / 0/10 | fail / 1/10 | pass / 4/10 | pass / 10/10 | 1/5 | 0/5 | 2/5 | fail |
| online_1250 | fail / 0/10 | fail / 0/10 | pass / 10/10 | pass / 10/10 | 1/5 | 1/5 | 2/5 | fail |
| online_1500 | pass / 1/10 | pass / 2/10 | pass / 10/10 | pass / 10/10 | 1/5 | 2/5 | 3/5 | fail |
| online_1750 | fail / 1/10 | fail / 1/10 | pass / 8/10 | pass / 10/10 | 1/5 | 1/5 | 3/5 | fail |
| online_2000 | fail / 4/10 | pass / 0/10 | pass / 7/10 | pass / 10/10 | 0/5 | 1/5 | 2/5 | fail / 0/5 |

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

| Checkpoint | Prefix 5 | Prefix 6 | Prefix 7 | Prefix 8 | Cold | Prefix-6 Wilson 95% |
| --- | --- | --- | --- | --- | --- | --- |
| online_250 | 2/30 | 3/30 | 9/30 | 30/30 D | 0/30 | [0.03, 0.26] |
| online_500 | 2/30 | 7/30 | 15/30 D | 30/30 D | 0/30 | [0.12, 0.41] |
| online_750 | 8/30 D | 6/30 | 5/30 | 30/30 D | 0/30 | [0.10, 0.37] |
| online_1000 | 7/30 | 5/30 | 13/30 D | 30/30 D | 0/30 | [0.07, 0.34] |
| online_1250 | 6/30 | 6/30 | 26/30 D | 30/30 D | 0/30 | [0.10, 0.37] |
| online_1500 | 4/30 D | 12/30 D | 19/30 D | 30/30 D | 0/30 | [0.25, 0.58] |
| online_1750 | 10/30 | 5/30 | 21/30 D | 30/30 D | 0/30 | [0.07, 0.34] |
| online_2000 | 11/30 | 10/30 D | 21/30 D | 30/30 D | 0/30 | [0.19, 0.51] |

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
