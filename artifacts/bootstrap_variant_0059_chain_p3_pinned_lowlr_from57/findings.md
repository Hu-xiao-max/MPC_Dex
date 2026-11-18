# Variant 59 `chain_p3_pinned_lowlr_from57`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_controls": 3, "prefix_set": [3, 4, 5, 6, 7, 8], "demonstration_include_mpc": false, "demonstration_recorded_rows": [3, 9], "actor_bc_lambda": 100.0, "actor_bc_rows": "first_k", "handoff_noise_sigma": 0.015, "handoff_noise_prefixes": [4, 5, 6, 7, 8], "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "initial_from": ["bootstrap_variant_0057_chain_p4_pinned_lowlr_from56", "online_2000"], "actor_bc_first_rows": 4, "actor_lr_override": 0.0001, "eval_stochastic_trials": 10, "actor_bc_std_target": 0.01}`. Inherited checkpoint: `bootstrap_variant_0057_chain_p4_pinned_lowlr_from56/checkpoints/online_2000`. Hypothesis: Prefix 3 inherited from the prefix-4 stage (0057 online_2000 in-run: prefix-4 deterministic pass and 3/10, prefix-5 8/10, prefixes 6-8 10/10), recorded controls 4-7 pinned at obs3-obs6, sigma .015 noise on prefixes 4-8, actor step 1e-4.

Stage status `complete_sac_stage` (exit=0, 514 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 2500, 'online_updates': 10000}; 500 critic-pretraining and 2000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: overridden to 0.0001 for this stage (inherited 0.0001; ramp fraction unchanged). Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over the first 4 rows of each demonstration episode, added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.01)) added to the same actor step. Handoff noise: when sigma > 0 and the sampled prefix is listed, the final prefix control is executed as recorded action + N(0, sigma^2) per dimension in normalized action space, clipped to [-1, 1], from the bit-exact recorded state before it; it is a non-policy row, never appended to any buffer, and its episode is counted under key "<prefix>+noise".

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'3': 54, '4+noise': 51, '5+noise': 41, '6+noise': 30, '7+noise': 43, '8+noise': 37}, force-qualified successes {'3': 0, '4+noise': 3, '5+noise': 3, '6+noise': 16, '7+noise': 25, '8+noise': 28} (informational; the gate is the 6-row demonstration buffer).
- Online: 2000 interactions; completed episodes by prefix {'3': 89, '4+noise': 102, '5+noise': 98, '6+noise': 86, '7+noise': 101, '8+noise': 100}, successes {'3': 5, '4+noise': 4, '5+noise': 7, '6+noise': 31, '7+noise': 54, '8+noise': 66}; outcomes {'drop': 254, 'palm_support': 154, 'success': 167, 'truncated': 1}; handoffs terminated before the policy 24; unfinished last suffix True. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 4540; none entered replay.
- Actor BC loss first/last: 0.01211 / 0.0001364; spread-anchor loss first/last: 0.4541 / 4.288e-06; alpha raw-equivalent last 0.1431.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 0.5094, std 0.0389, bias -0.33/-0.38; after_pretraining: L2 0.5094, std 0.0389, bias +0.03/+0.05; online_250: L2 0.0679, std 0.0100, bias +0.25/+0.28; online_500: L2 0.0283, std 0.0100, bias +0.04/-0.03; online_750: L2 0.0253, std 0.0100, bias +0.07/+0.06; online_1000: L2 0.0313, std 0.0100, bias -0.14/+0.08; online_1250: L2 0.0423, std 0.0100, bias -0.05/-0.15; online_1500: L2 0.0405, std 0.0100, bias +0.01/-0.05; online_1750: L2 0.0184, std 0.0100, bias +0.22/+0.27; online_2000: L2 0.0122, std 0.0100, bias +0.06/-0.15.

## Frozen evaluations (in-run; 10 stochastic seeds per prefix)

| Checkpoint | Prefix-3 det / stoch | Prefix-4 det / stoch | Prefix-5 det / stoch | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch | Prefix-4 noisy | Prefix-5 noisy | Prefix-6 noisy | Prefix-7 noisy | Prefix-8 noisy | Cold det / stoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pretraining_baseline | fail / 0/10 | pass / 3/10 | pass / 8/10 | pass / 10/10 | pass / 10/10 | pass / 10/10 | 1/5 | 0/5 | 2/5 | 5/5 | 4/5 | fail |
| online_250 | fail / 0/10 | fail / 4/10 | fail / 3/10 | pass / 10/10 | pass / 9/10 | pass / 10/10 | 0/5 | 0/5 | 2/5 | 5/5 | 4/5 | fail |
| online_500 | pass / 0/10 | fail / 4/10 | pass / 2/10 | pass / 10/10 | pass / 10/10 | pass / 10/10 | 0/5 | 0/5 | 2/5 | 5/5 | 4/5 | fail |
| online_750 | fail / 2/10 | fail / 3/10 | pass / 6/10 | pass / 6/10 | pass / 10/10 | pass / 10/10 | 0/5 | 0/5 | 3/5 | 4/5 | 4/5 | fail |
| online_1000 | fail / 1/10 | fail / 2/10 | pass / 5/10 | fail / 5/10 | fail / 3/10 | pass / 10/10 | 0/5 | 0/5 | 2/5 | 4/5 | 4/5 | fail |
| online_1250 | fail / 0/10 | fail / 4/10 | pass / 4/10 | fail / 3/10 | fail / 2/10 | pass / 10/10 | 0/5 | 0/5 | 4/5 | 4/5 | 4/5 | fail |
| online_1500 | fail / 0/10 | fail / 3/10 | pass / 5/10 | pass / 4/10 | fail / 2/10 | pass / 10/10 | 1/5 | 0/5 | 1/5 | 4/5 | 4/5 | fail |
| online_1750 | pass / 1/10 | fail / 1/10 | fail / 3/10 | fail / 3/10 | pass / 10/10 | pass / 10/10 | 1/5 | 0/5 | 3/5 | 3/5 | 4/5 | fail |
| online_2000 | fail / 0/10 | pass / 3/10 | pass / 6/10 | pass / 9/10 | pass / 10/10 | pass / 10/10 | 0/5 | 1/5 | 0/5 | 4/5 | 4/5 | fail / 0/5 |

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

| Checkpoint | Prefix 5 | Prefix 6 | Prefix 7 | Prefix 8 | Cold | Prefix-6 Wilson 95% |
| --- | --- | --- | --- | --- | --- | --- |
| online_250 | 11/30 | 29/30 D | 29/30 D | 30/30 D | 0/30 | [0.83, 0.99] |
| online_500 | 12/30 D | 27/30 D | 30/30 D | 30/30 D | 0/30 | [0.74, 0.97] |
| online_750 | 13/30 D | 22/30 D | 30/30 D | 30/30 D | 0/30 | [0.56, 0.86] |
| online_1000 | 11/30 D | 16/30 | 15/30 | 30/30 D | 0/30 | [0.36, 0.70] |
| online_1250 | 12/30 D | 7/30 | 5/30 | 30/30 D | 0/30 | [0.12, 0.41] |
| online_1500 | 10/30 D | 21/30 D | 3/30 | 30/30 D | 0/30 | [0.52, 0.83] |
| online_1750 | 5/30 | 13/30 | 28/30 D | 30/30 D | 0/30 | [0.27, 0.61] |
| online_2000 | 22/30 D | 26/30 D | 30/30 D | 30/30 D | 0/30 | [0.70, 0.95] |

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
