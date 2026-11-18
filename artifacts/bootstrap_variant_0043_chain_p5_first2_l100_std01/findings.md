# Variant 43 `chain_p5_first2_l100_std01`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_controls": 5, "prefix_set": [5, 6, 7, 8], "demonstration_include_mpc": false, "demonstration_recorded_rows": [5, 9], "actor_bc_lambda": 100.0, "actor_bc_rows": "first_k", "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "initial_from": ["bootstrap_variant_0004_recorded_bc10_first", "online_2000"], "actor_bc_first_rows": 2, "eval_stochastic_trials": 10, "actor_bc_std_target": 0.01}`. Inherited checkpoint: `bootstrap_variant_0004_recorded_bc10_first/checkpoints/online_2000`. Hypothesis: Chained prefix-5 stage with the pinned-handoff recipe applied at both new handoffs (obs5 and obs6): strong mean pin and .01 spread anchor.

Stage status `complete_sac_stage` (exit=0, 730 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 1500, 'online_updates': 6000}; 500 critic-pretraining and 2000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: inherited 3e-4. Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over the first 2 rows of each demonstration episode, added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.01)) added to the same actor step. Handoff noise: off.

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'5': 82, '6': 64, '7': 58, '8': 52}, force-qualified successes {'5': 0, '6': 40, '7': 47, '8': 52} (informational; the gate is the 4-row demonstration buffer).
- Online: 2000 interactions; completed episodes by prefix {'5': 145, '6': 139, '7': 129, '8': 156}, successes {'5': 19, '6': 51, '7': 104, '8': 156}; outcomes {'drop': 212, 'palm_support': 27, 'success': 330}; handoffs terminated before the policy 0; unfinished last suffix True. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 5332; none entered replay.
- Actor BC loss first/last: 0.02849 / 5.642e-05; spread-anchor loss first/last: 4.301 / 7.097e-06; alpha raw-equivalent last 0.0956.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 0.6237, std 0.0487, bias -0.60/-0.59; after_pretraining: L2 0.6237, std 0.0487, bias -0.06/-0.08; online_250: L2 0.0194, std 0.0100, bias -0.56/-0.56; online_500: L2 0.0365, std 0.0100, bias -0.21/-0.31; online_750: L2 0.0417, std 0.0100, bias -0.16/-0.11; online_1000: L2 0.0169, std 0.0100, bias -0.02/-0.16; online_1250: L2 0.0264, std 0.0100, bias -0.17/-0.11; online_1500: L2 0.0259, std 0.0100, bias -0.21/-0.22; online_1750: L2 0.0323, std 0.0100, bias -0.20/-0.09; online_2000: L2 0.0104, std 0.0100, bias -0.21/-0.21.

## Frozen evaluations (in-run; 10 stochastic seeds per prefix)

| Checkpoint | Prefix-5 det / stoch | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch | Cold det / stoch |
| --- | --- | --- | --- | --- | --- |
| pretraining_baseline | fail / 0/10 | pass / 6/10 | pass / 10/10 | pass / 10/10 | fail |
| online_250 | fail / 0/10 | fail / 1/10 | fail / 0/10 | pass / 10/10 | fail |
| online_500 | fail / 2/10 | fail / 0/10 | fail / 0/10 | pass / 10/10 | fail |
| online_750 | fail / 0/10 | fail / 2/10 | pass / 10/10 | pass / 10/10 | fail |
| online_1000 | fail / 0/10 | pass / 3/10 | pass / 10/10 | pass / 10/10 | fail |
| online_1250 | fail / 5/10 | pass / 9/10 | pass / 10/10 | pass / 10/10 | fail |
| online_1500 | fail / 0/10 | fail / 5/10 | pass / 10/10 | pass / 10/10 | fail |
| online_1750 | fail / 0/10 | fail / 0/10 | fail / 3/10 | pass / 10/10 | fail |
| online_2000 | fail / 3/10 | fail / 3/10 | pass / 9/10 | pass / 10/10 | fail / 0/5 |

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

| Checkpoint | Prefix 5 | Prefix 6 | Prefix 7 | Prefix 8 | Cold | Prefix-6 Wilson 95% |
| --- | --- | --- | --- | --- | --- | --- |
| online_250 | 1/30 | 4/30 | 1/30 | 30/30 D | 0/30 | [0.05, 0.30] |
| online_500 | 4/30 | 0/30 | 2/30 | 30/30 D | 0/30 | [-0.00, 0.11] |
| online_750 | 1/30 | 7/30 | 30/30 D | 30/30 D | 0/30 | [0.12, 0.41] |
| online_1000 | 5/30 | 5/30 D | 29/30 D | 30/30 D | 0/30 | [0.07, 0.34] |
| online_1250 | 7/30 | 24/30 D | 30/30 D | 30/30 D | 0/30 | [0.63, 0.90] |
| online_1500 | 2/30 | 14/30 | 30/30 D | 30/30 D | 0/30 | [0.30, 0.64] |
| online_1750 | 1/30 | 2/30 | 12/30 | 30/30 D | 0/30 | [0.02, 0.21] |
| online_2000 | 4/30 | 14/30 | 29/30 D | 30/30 D | 0/30 | [0.30, 0.64] |

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
