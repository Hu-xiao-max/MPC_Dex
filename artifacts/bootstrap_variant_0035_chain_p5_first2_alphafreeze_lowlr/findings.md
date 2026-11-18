# Variant 35 `chain_p5_first2_alphafreeze_lowlr`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_controls": 5, "prefix_set": [5, 6, 7, 8], "demonstration_include_mpc": false, "demonstration_recorded_rows": [5, 9], "actor_bc_lambda": 10.0, "actor_bc_rows": "first_k", "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "initial_from": ["bootstrap_variant_0004_recorded_bc10_first", "online_2000"], "actor_bc_first_rows": 2, "actor_lr_override": 0.0001, "eval_stochastic_trials": 10, "alpha_freeze": true}`. Inherited checkpoint: `bootstrap_variant_0004_recorded_bc10_first/checkpoints/online_2000`. Hypothesis: Chained prefix-5 stage with both stabilizers.

Stage status `complete_sac_stage` (exit=0, 602 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 1500, 'online_updates': 6000}; 500 critic-pretraining and 2000 online updates as prescribed. Temperature: frozen at the inherited value for this stage; alpha loss logged only, alpha optimizer untouched. Actor step: overridden to 0.0001 for this stage (ramp fraction unchanged). Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over the first 2 rows of each demonstration episode, added to the single SAC actor step; spread anchor: none. Handoff noise: off.

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'5': 82, '6': 64, '7': 58, '8': 52}, force-qualified successes {'5': 0, '6': 40, '7': 47, '8': 52} (informational; the gate is the 4-row demonstration buffer).
- Online: 2000 interactions; completed episodes by prefix {'5': 141, '6': 131, '7': 124, '8': 141}, successes {'5': 4, '6': 29, '7': 74, '8': 141}; outcomes {'drop': 243, 'palm_support': 45, 'success': 248, 'truncated': 1}; handoffs terminated before the policy 0; unfinished last suffix True. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 5109; none entered replay.
- Actor BC loss first/last: 0.02849 / 6.042e-05; spread-anchor loss first/last: none; alpha raw-equivalent last 0.0754.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 0.6237, std 0.0487, bias -0.60/-0.59; after_pretraining: L2 0.6237, std 0.0487, bias -0.06/-0.08; online_250: L2 0.1124, std 0.0568, bias -0.05/-0.05; online_500: L2 0.0922, std 0.0618, bias -0.12/-0.11; online_750: L2 0.0138, std 0.0675, bias -0.18/-0.15; online_1000: L2 0.0325, std 0.0755, bias -0.18/-0.11; online_1250: L2 0.0384, std 0.0816, bias -0.14/-0.11; online_1500: L2 0.0351, std 0.0869, bias -0.11/-0.09; online_1750: L2 0.0141, std 0.0964, bias -0.18/-0.11; online_2000: L2 0.0146, std 0.1011, bias -0.09/-0.18.

## Frozen evaluations (in-run; 10 stochastic seeds per prefix)

| Checkpoint | Prefix-5 det / stoch | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch | Cold det / stoch |
| --- | --- | --- | --- | --- | --- |
| pretraining_baseline | fail / 0/10 | pass / 6/10 | pass / 10/10 | pass / 10/10 | fail |
| online_250 | fail / 0/10 | pass / 3/10 | pass / 7/10 | pass / 10/10 | fail |
| online_500 | fail / 1/10 | pass / 6/10 | pass / 6/10 | pass / 10/10 | fail |
| online_750 | fail / 0/10 | fail / 1/10 | fail / 5/10 | pass / 10/10 | fail |
| online_1000 | fail / 0/10 | fail / 1/10 | pass / 5/10 | pass / 10/10 | fail |
| online_1250 | fail / 0/10 | fail / 2/10 | pass / 7/10 | pass / 10/10 | fail |
| online_1500 | fail / 1/10 | fail / 1/10 | fail / 5/10 | pass / 10/10 | fail |
| online_1750 | fail / 0/10 | pass / 3/10 | pass / 9/10 | pass / 10/10 | fail |
| online_2000 | fail / 1/10 | pass / 4/10 | fail / 7/10 | pass / 10/10 | fail / 0/5 |

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

| Checkpoint | Prefix 5 | Prefix 6 | Prefix 7 | Prefix 8 | Cold | Prefix-6 Wilson 95% |
| --- | --- | --- | --- | --- | --- | --- |
| online_250 | 1/30 | 13/30 D | 22/30 D | 30/30 D | 0/30 | [0.27, 0.61] |
| online_500 | 2/30 | 10/30 D | 24/30 D | 30/30 D | 0/30 | [0.19, 0.51] |
| online_750 | 3/30 | 3/30 | 11/30 | 30/30 D | 0/30 | [0.03, 0.26] |
| online_1000 | 2/30 | 1/30 | 23/30 D | 30/30 D | 0/30 | [0.01, 0.17] |
| online_1250 | 2/30 | 12/30 | 24/30 D | 30/30 D | 0/30 | [0.25, 0.58] |
| online_1500 | 2/30 | 7/30 | 17/30 | 30/30 D | 0/30 | [0.12, 0.41] |
| online_1750 | 5/30 | 16/30 D | 26/30 D | 30/30 D | 0/30 | [0.36, 0.70] |
| online_2000 | 4/30 | 16/30 D | 24/30 | 30/30 D | 0/30 | [0.36, 0.70] |

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
