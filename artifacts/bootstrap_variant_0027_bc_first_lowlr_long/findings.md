# Variant 27 `bc_first_lowlr_long`: auto-generated findings

**Settings differing from SAC0004:** `{"demonstration_include_mpc": false, "actor_bc_lambda": 10.0, "actor_bc_rows": "first", "online_policy_interactions": 4000, "checkpoints": [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], "actor_lr_override": 0.0001, "eval_stochastic_trials": 10}`. Inherited checkpoint: `bootstrap_suffix_sac_0001/checkpoints/online_2000`. Hypothesis: bc_first_lowlr with twice the budget, since a smaller step may need more updates to reach the same handoff residual.

Stage status `complete_sac_stage` (exit=0, 786 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 1000, 'online_updates': 6000}; 500 critic-pretraining and 4000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: overridden to 0.0001 for this stage (ramp fraction unchanged). Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over the first row of each demonstration episode, added to the single SAC actor step; spread anchor: none. Handoff noise: off.

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'6': 256}, force-qualified successes {'6': 0} (informational; the gate is the 3-row demonstration buffer).
- Online: 4000 interactions; completed episodes by prefix {'6': 873}, successes {'6': 305}; outcomes {'drop': 455, 'success': 305, 'palm_support': 108, 'truncated': 5}; handoffs terminated before the policy 0; unfinished last suffix True. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 6780; none entered replay.
- Actor BC loss first/last: 0.04802 / 5.281e-05; spread-anchor loss first/last: none; alpha raw-equivalent last 0.0921.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 0.4126, std 0.0674, bias -0.68/-0.62; after_pretraining: L2 0.4126, std 0.0674, bias -0.00/+0.01; online_500: L2 0.0250, std 0.0657, bias -0.18/-0.21; online_1000: L2 0.0377, std 0.0790, bias -0.06/-0.09; online_1500: L2 0.0201, std 0.1052, bias -0.14/-0.15; online_2000: L2 0.0214, std 0.1255, bias -0.15/-0.06; online_2500: L2 0.0243, std 0.1349, bias -0.25/-0.02; online_3000: L2 0.0311, std 0.1309, bias -0.15/-0.13; online_3500: L2 0.0259, std 0.1228, bias -0.10/-0.05; online_4000: L2 0.0186, std 0.1485, bias -0.15/-0.15.

## Frozen evaluations (in-run; 10 stochastic seeds per prefix)

| Checkpoint | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch | Cold det / stoch |
| --- | --- | --- | --- | --- |
| pretraining_baseline | fail / 0/10 | pass / 10/10 | pass / 10/10 | fail |
| online_500 | fail / 0/10 | fail / 1/10 | pass / 10/10 | fail |
| online_1000 | fail / 0/10 | pass / 8/10 | pass / 10/10 | fail |
| online_1500 | fail / 2/10 | pass / 9/10 | pass / 10/10 | fail |
| online_2000 | pass / 3/10 | pass / 8/10 | pass / 10/10 | fail |
| online_2500 | fail / 1/10 | pass / 7/10 | pass / 10/10 | fail |
| online_3000 | pass / 4/10 | pass / 10/10 | pass / 10/10 | fail |
| online_3500 | fail / 0/10 | fail / 7/10 | pass / 10/10 | fail |
| online_4000 | pass / 1/10 | fail / 4/10 | pass / 10/10 | fail / 0/5 |

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

| Checkpoint | Prefix 5 | Prefix 6 | Prefix 7 | Prefix 8 | Cold | Prefix-6 Wilson 95% |
| --- | --- | --- | --- | --- | --- | --- |
| online_500 | 0/30 | 3/30 | 2/30 | 30/30 D | 0/30 | [0.03, 0.26] |
| online_1000 | 1/30 | 2/30 | 29/30 D | 30/30 D | 0/30 | [0.02, 0.21] |
| online_1500 | 0/30 | 5/30 | 27/30 D | 30/30 D | 0/30 | [0.07, 0.34] |
| online_2000 | 0/30 | 8/30 D | 23/30 D | 30/30 D | 0/30 | [0.14, 0.44] |
| online_2500 | 0/30 | 7/30 | 25/30 D | 30/30 D | 0/30 | [0.12, 0.41] |
| online_3000 | 0/30 | 19/30 D | 30/30 D | 30/30 D | 0/30 | [0.46, 0.78] |
| online_3500 | 0/30 | 1/30 | 20/30 | 30/30 D | 0/30 | [0.01, 0.17] |
| online_4000 | 0/30 | 5/30 D | 16/30 | 30/30 D | 0/30 | [0.07, 0.34] |

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
