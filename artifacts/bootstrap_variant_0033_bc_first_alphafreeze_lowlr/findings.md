# Variant 33 `bc_first_alphafreeze_lowlr`: auto-generated findings

**Settings differing from SAC0004:** `{"demonstration_include_mpc": false, "actor_bc_lambda": 10.0, "actor_bc_rows": "first", "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "actor_lr_override": 0.0001, "eval_stochastic_trials": 10, "alpha_freeze": true}`. Inherited checkpoint: `bootstrap_suffix_sac_0001/checkpoints/online_2000`. Hypothesis: Both stabilizers: frozen temperature and the smaller actor step.

Stage status `complete_sac_stage` (exit=0, 488 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 1000, 'online_updates': 4000}; 500 critic-pretraining and 2000 online updates as prescribed. Temperature: frozen at the inherited value for this stage; alpha loss logged only, alpha optimizer untouched. Actor step: overridden to 0.0001 for this stage (ramp fraction unchanged). Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over the first row of each demonstration episode, added to the single SAC actor step; spread anchor: none. Handoff noise: off.

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'6': 256}, force-qualified successes {'6': 0} (informational; the gate is the 3-row demonstration buffer).
- Online: 2000 interactions; completed episodes by prefix {'6': 435}, successes {'6': 81}; outcomes {'drop': 269, 'success': 81, 'palm_support': 85}; handoffs terminated before the policy 0; unfinished last suffix True. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 4152; none entered replay.
- Actor BC loss first/last: 0.04802 / 4.273e-05; spread-anchor loss first/last: none; alpha raw-equivalent last 0.0612.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 0.4126, std 0.0674, bias -0.68/-0.62; after_pretraining: L2 0.4126, std 0.0674, bias -0.00/+0.01; online_250: L2 0.0565, std 0.0634, bias -0.17/-0.01; online_500: L2 0.0179, std 0.0634, bias -0.28/-0.22; online_750: L2 0.0257, std 0.0669, bias -0.14/-0.00; online_1000: L2 0.0225, std 0.0716, bias -0.05/-0.10; online_1250: L2 0.0190, std 0.0790, bias -0.06/-0.19; online_1500: L2 0.0207, std 0.0887, bias -0.03/-0.14; online_1750: L2 0.0156, std 0.0985, bias -0.08/-0.05; online_2000: L2 0.0125, std 0.1073, bias -0.13/-0.09.

## Frozen evaluations (in-run; 10 stochastic seeds per prefix)

| Checkpoint | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch | Cold det / stoch |
| --- | --- | --- | --- | --- |
| pretraining_baseline | fail / 0/10 | pass / 10/10 | pass / 10/10 | fail |
| online_250 | fail / 0/10 | fail / 1/10 | pass / 10/10 | fail |
| online_500 | pass / 0/10 | fail / 5/10 | pass / 10/10 | fail |
| online_750 | pass / 6/10 | pass / 7/10 | pass / 10/10 | fail |
| online_1000 | pass / 3/10 | pass / 6/10 | pass / 10/10 | fail |
| online_1250 | fail / 1/10 | pass / 5/10 | pass / 10/10 | fail |
| online_1500 | fail / 3/10 | pass / 5/10 | pass / 10/10 | fail |
| online_1750 | pass / 3/10 | fail / 6/10 | pass / 10/10 | fail |
| online_2000 | pass / 5/10 | pass / 8/10 | pass / 10/10 | fail / 0/5 |

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

| Checkpoint | Prefix 5 | Prefix 6 | Prefix 7 | Prefix 8 | Cold | Prefix-6 Wilson 95% |
| --- | --- | --- | --- | --- | --- | --- |
| online_250 | 0/30 | 2/30 | 5/30 | 30/30 D | 0/30 | [0.02, 0.21] |
| online_500 | 0/30 | 4/30 D | 7/30 | 30/30 D | 0/30 | [0.05, 0.30] |
| online_750 | 0/30 | 11/30 D | 24/30 D | 30/30 D | 0/30 | [0.22, 0.54] |
| online_1000 | 0/30 | 10/30 D | 27/30 D | 30/30 D | 0/30 | [0.19, 0.51] |
| online_1250 | 0/30 | 3/30 | 20/30 D | 30/30 D | 0/30 | [0.03, 0.26] |
| online_1500 | 0/30 | 7/30 | 24/30 D | 30/30 D | 0/30 | [0.12, 0.41] |
| online_1750 | 0/30 | 9/30 D | 22/30 | 30/30 D | 0/30 | [0.17, 0.48] |
| online_2000 | 0/30 | 11/30 D | 20/30 D | 30/30 D | 0/30 | [0.22, 0.54] |

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
