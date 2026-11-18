# Variant 37 `bc_first_std02`: auto-generated findings

**Settings differing from SAC0004:** `{"demonstration_include_mpc": false, "actor_bc_lambda": 10.0, "actor_bc_rows": "first", "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "eval_stochastic_trials": 10, "actor_bc_std_target": 0.02}`. Inherited checkpoint: `bootstrap_suffix_sac_0001/checkpoints/online_2000`. Hypothesis: Spread anchor at the handoff row (latent std toward .02 per dimension, L2 about .08): the handoff std grew from .067 to .13-.17 in every BC run while the basin is about .025-.05 wide, capping stochastic success near two thirds.

Stage status `complete_sac_stage` (exit=0, 692 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 1000, 'online_updates': 4000}; 500 critic-pretraining and 2000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: inherited 3e-4. Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over the first row of each demonstration episode, added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.02)) added to the same actor step. Handoff noise: off.

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'6': 256}, force-qualified successes {'6': 0} (informational; the gate is the 3-row demonstration buffer).
- Online: 2000 interactions; completed episodes by prefix {'6': 439}, successes {'6': 85}; outcomes {'drop': 270, 'palm_support': 83, 'success': 85, 'truncated': 1}; handoffs terminated before the policy 0; unfinished last suffix True. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 4176; none entered replay.
- Actor BC loss first/last: 0.04802 / 9.863e-05; spread-anchor loss first/last: 1.468 / 9.915e-06; alpha raw-equivalent last 0.0769.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 0.4126, std 0.0674, bias -0.68/-0.62; after_pretraining: L2 0.4126, std 0.0674, bias -0.00/+0.01; online_250: L2 0.0524, std 0.0200, bias -0.07/-0.01; online_500: L2 0.0882, std 0.0200, bias -0.32/-0.33; online_750: L2 0.0339, std 0.0200, bias -0.27/-0.25; online_1000: L2 0.0614, std 0.0200, bias -0.18/-0.09; online_1250: L2 0.0447, std 0.0200, bias -0.16/-0.05; online_1500: L2 0.0424, std 0.0200, bias -0.13/-0.22; online_1750: L2 0.0469, std 0.0200, bias -0.16/-0.30; online_2000: L2 0.0215, std 0.0200, bias -0.21/-0.16.

## Frozen evaluations (in-run; 10 stochastic seeds per prefix)

| Checkpoint | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch | Cold det / stoch |
| --- | --- | --- | --- | --- |
| pretraining_baseline | fail / 0/10 | pass / 10/10 | pass / 10/10 | fail |
| online_250 | fail / 0/10 | fail / 0/10 | pass / 10/10 | fail |
| online_500 | fail / 0/10 | fail / 0/10 | pass / 10/10 | fail |
| online_750 | pass / 4/10 | pass / 6/10 | pass / 10/10 | fail |
| online_1000 | fail / 3/10 | pass / 10/10 | pass / 10/10 | fail |
| online_1250 | fail / 3/10 | pass / 10/10 | pass / 10/10 | fail |
| online_1500 | fail / 0/10 | pass / 7/10 | pass / 10/10 | fail |
| online_1750 | fail / 1/10 | fail / 3/10 | pass / 10/10 | fail |
| online_2000 | fail / 4/10 | pass / 9/10 | pass / 10/10 | fail / 0/5 |

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

| Checkpoint | Prefix 5 | Prefix 6 | Prefix 7 | Prefix 8 | Cold | Prefix-6 Wilson 95% |
| --- | --- | --- | --- | --- | --- | --- |
| online_250 | 0/30 | 0/30 | 0/30 | 30/30 D | 0/30 | [-0.00, 0.11] |
| online_500 | 0/30 | 6/30 | 0/30 | 30/30 D | 0/30 | [0.10, 0.37] |
| online_750 | 0/30 | 11/30 D | 28/30 D | 30/30 D | 0/30 | [0.22, 0.54] |
| online_1000 | 0/30 | 7/30 | 24/30 D | 30/30 D | 0/30 | [0.12, 0.41] |
| online_1250 | 0/30 | 8/30 | 30/30 D | 30/30 D | 0/30 | [0.14, 0.44] |
| online_1500 | 0/30 | 3/30 | 22/30 D | 30/30 D | 0/30 | [0.03, 0.26] |
| online_1750 | 0/30 | 5/30 | 7/30 | 30/30 D | 0/30 | [0.07, 0.34] |
| online_2000 | 0/30 | 6/30 | 29/30 D | 30/30 D | 0/30 | [0.10, 0.37] |

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
