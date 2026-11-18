# Variant 46 `bc_first_l100_std01_long`: auto-generated findings

**Settings differing from SAC0004:** `{"demonstration_include_mpc": false, "actor_bc_lambda": 100.0, "actor_bc_rows": "first", "online_policy_interactions": 4000, "checkpoints": [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], "eval_stochastic_trials": 10, "actor_bc_std_target": 0.01}`. Inherited checkpoint: `bootstrap_suffix_sac_0001/checkpoints/online_2000`. Hypothesis: bc_first_l100_std01 with twice the budget: does the pinned handoff stay stable beyond 2,000 updates (bc_first_long decayed from 20/30 to 6/30 by 4,000)?

Stage status `complete_sac_stage` (exit=0, 947 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 1000, 'online_updates': 6000}; 500 critic-pretraining and 4000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: inherited 3e-4. Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over the first row of each demonstration episode, added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.01)) added to the same actor step. Handoff noise: off.

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'6': 256}, force-qualified successes {'6': 0} (informational; the gate is the 3-row demonstration buffer).
- Online: 4000 interactions; completed episodes by prefix {'6': 1107}, successes {'6': 713}; outcomes {'drop': 327, 'palm_support': 64, 'success': 713, 'truncated': 3}; handoffs terminated before the policy 0; unfinished last suffix False. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 8178; none entered replay.
- Actor BC loss first/last: 0.04802 / 0.0005584; spread-anchor loss first/last: 3.61 / 3.211e-05; alpha raw-equivalent last 0.0950.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 0.4126, std 0.0674, bias -0.68/-0.62; after_pretraining: L2 0.4126, std 0.0674, bias -0.00/+0.01; online_500: L2 0.0120, std 0.0100, bias -0.19/-0.20; online_1000: L2 0.0076, std 0.0100, bias -0.08/-0.14; online_1500: L2 0.0355, std 0.0100, bias -0.17/-0.12; online_2000: L2 0.0166, std 0.0100, bias -0.07/-0.12; online_2500: L2 0.0135, std 0.0100, bias -0.11/-0.15; online_3000: L2 0.0090, std 0.0100, bias -0.13/-0.18; online_3500: L2 0.0292, std 0.0100, bias -0.17/-0.23; online_4000: L2 0.0324, std 0.0100, bias -0.14/-0.20.

## Frozen evaluations (in-run; 10 stochastic seeds per prefix)

| Checkpoint | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch | Cold det / stoch |
| --- | --- | --- | --- | --- |
| pretraining_baseline | fail / 0/10 | pass / 10/10 | pass / 10/10 | fail |
| online_500 | fail / 0/10 | fail / 0/10 | pass / 10/10 | fail |
| online_1000 | pass / 3/10 | pass / 8/10 | pass / 10/10 | fail |
| online_1500 | fail / 3/10 | pass / 10/10 | pass / 10/10 | fail |
| online_2000 | pass / 9/10 | pass / 10/10 | pass / 10/10 | fail |
| online_2500 | pass / 7/10 | fail / 10/10 | pass / 10/10 | fail |
| online_3000 | pass / 7/10 | pass / 10/10 | pass / 10/10 | fail |
| online_3500 | pass / 8/10 | pass / 9/10 | pass / 10/10 | fail |
| online_4000 | pass / 9/10 | pass / 10/10 | pass / 10/10 | fail / 0/5 |

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

| Checkpoint | Prefix 5 | Prefix 6 | Prefix 7 | Prefix 8 | Cold | Prefix-6 Wilson 95% |
| --- | --- | --- | --- | --- | --- | --- |
| online_500 | 1/30 | 3/30 | 2/30 | 30/30 D | 0/30 | [0.03, 0.26] |
| online_1000 | 3/30 | 12/30 D | 25/30 D | 30/30 D | 0/30 | [0.25, 0.58] |
| online_1500 | 1/30 | 16/30 | 30/30 D | 30/30 D | 0/30 | [0.36, 0.70] |
| online_2000 | 0/30 | 26/30 D | 30/30 D | 30/30 D | 0/30 | [0.70, 0.95] |
| online_2500 | 5/30 | 28/30 D | 26/30 | 30/30 D | 0/30 | [0.79, 0.98] |
| online_3000 | 14/30 | 21/30 D | 30/30 D | 30/30 D | 0/30 | [0.52, 0.83] |
| online_3500 | 2/30 | 24/30 D | 29/30 D | 30/30 D | 0/30 | [0.63, 0.90] |
| online_4000 | 5/30 | 28/30 D | 30/30 D | 30/30 D | 0/30 | [0.79, 0.98] |

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
