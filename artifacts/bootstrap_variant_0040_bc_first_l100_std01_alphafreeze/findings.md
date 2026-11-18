# Variant 40 `bc_first_l100_std01_alphafreeze`: auto-generated findings

**Settings differing from SAC0004:** `{"demonstration_include_mpc": false, "actor_bc_lambda": 100.0, "actor_bc_rows": "first", "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "eval_stochastic_trials": 10, "alpha_freeze": true, "actor_bc_std_target": 0.01}`. Inherited checkpoint: `bootstrap_suffix_sac_0001/checkpoints/online_2000`. Hypothesis: Strong mean pin, tight spread anchor and frozen temperature.

Stage status `complete_sac_stage` (exit=0, 679 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 1000, 'online_updates': 4000}; 500 critic-pretraining and 2000 online updates as prescribed. Temperature: frozen at the inherited value for this stage; alpha loss logged only, alpha optimizer untouched. Actor step: inherited 3e-4. Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over the first row of each demonstration episode, added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.01)) added to the same actor step. Handoff noise: off.

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'6': 256}, force-qualified successes {'6': 0} (informational; the gate is the 3-row demonstration buffer).
- Online: 2000 interactions; completed episodes by prefix {'6': 484}, successes {'6': 195}; outcomes {'drop': 214, 'palm_support': 74, 'success': 195, 'truncated': 1}; handoffs terminated before the policy 0; unfinished last suffix True. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 4446; none entered replay.
- Actor BC loss first/last: 0.04802 / 0.0002482; spread-anchor loss first/last: 3.61 / 1.625e-07; alpha raw-equivalent last 0.0612.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 0.4126, std 0.0674, bias -0.68/-0.62; after_pretraining: L2 0.4126, std 0.0674, bias -0.00/+0.01; online_250: L2 0.0146, std 0.0100, bias -0.05/-0.13; online_500: L2 0.0138, std 0.0100, bias -0.19/-0.19; online_750: L2 0.0116, std 0.0100, bias -0.03/-0.16; online_1000: L2 0.0059, std 0.0100, bias -0.09/-0.06; online_1250: L2 0.0132, std 0.0100, bias -0.03/-0.12; online_1500: L2 0.0105, std 0.0100, bias -0.15/-0.10; online_1750: L2 0.0200, std 0.0100, bias -0.21/-0.10; online_2000: L2 0.0513, std 0.0100, bias -0.21/-0.10.

## Frozen evaluations (in-run; 10 stochastic seeds per prefix)

| Checkpoint | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch | Cold det / stoch |
| --- | --- | --- | --- | --- |
| pretraining_baseline | fail / 0/10 | pass / 10/10 | pass / 10/10 | fail |
| online_250 | fail / 0/10 | fail / 0/10 | pass / 10/10 | fail |
| online_500 | fail / 0/10 | fail / 0/10 | pass / 10/10 | fail |
| online_750 | fail / 3/10 | fail / 6/10 | pass / 10/10 | fail |
| online_1000 | pass / 6/10 | pass / 10/10 | pass / 10/10 | fail |
| online_1250 | fail / 5/10 | pass / 10/10 | pass / 10/10 | fail |
| online_1500 | pass / 7/10 | pass / 10/10 | pass / 10/10 | fail |
| online_1750 | fail / 3/10 | pass / 10/10 | pass / 10/10 | fail |
| online_2000 | fail / 0/10 | pass / 7/10 | pass / 10/10 | fail / 0/5 |

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

| Checkpoint | Prefix 5 | Prefix 6 | Prefix 7 | Prefix 8 | Cold | Prefix-6 Wilson 95% |
| --- | --- | --- | --- | --- | --- | --- |
| online_250 | 0/30 | 0/30 | 0/30 | 30/30 D | 0/30 | [-0.00, 0.11] |
| online_500 | 0/30 | 3/30 | 1/30 | 30/30 D | 0/30 | [0.03, 0.26] |
| online_750 | 0/30 | 16/30 | 24/30 | 30/30 D | 0/30 | [0.36, 0.70] |
| online_1000 | 0/30 | 16/30 D | 29/30 D | 30/30 D | 0/30 | [0.36, 0.70] |
| online_1250 | 1/30 | 17/30 | 25/30 D | 30/30 D | 0/30 | [0.39, 0.73] |
| online_1500 | 1/30 | 22/30 D | 30/30 D | 30/30 D | 0/30 | [0.56, 0.86] |
| online_1750 | 1/30 | 18/30 | 30/30 D | 30/30 D | 0/30 | [0.42, 0.75] |
| online_2000 | 0/30 | 0/30 | 15/30 D | 30/30 D | 0/30 | [-0.00, 0.11] |

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
