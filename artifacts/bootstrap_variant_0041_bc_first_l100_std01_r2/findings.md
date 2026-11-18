# Variant 41 `bc_first_l100_std01_r2`: auto-generated findings

**Settings differing from SAC0004:** `{"demonstration_include_mpc": false, "actor_bc_lambda": 100.0, "actor_bc_rows": "first", "seed_matched": false, "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "eval_stochastic_trials": 10, "actor_bc_std_target": 0.01}`. Inherited checkpoint: `bootstrap_suffix_sac_0001/checkpoints/online_2000`. Hypothesis: Seed replicate of bc_first_l100_std01 (prefix-6 8/10 at online 1,750 with prefix-7/8 10/10).

Stage status `complete_sac_stage` (exit=0, 612 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 1000, 'online_updates': 4000}; 500 critic-pretraining and 2000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: inherited 3e-4. Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over the first row of each demonstration episode, added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.01)) added to the same actor step. Handoff noise: off.

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'6': 256}, force-qualified successes {'6': 0} (informational; the gate is the 3-row demonstration buffer).
- Online: 2000 interactions; completed episodes by prefix {'6': 511}, successes {'6': 283}; outcomes {'drop': 191, 'success': 283, 'palm_support': 37}; handoffs terminated before the policy 0; unfinished last suffix True. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 4608; none entered replay.
- Actor BC loss first/last: 0.04802 / 9.571e-05; spread-anchor loss first/last: 3.61 / 2.36e-07; alpha raw-equivalent last 0.0776.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 0.4126, std 0.0674, bias -0.68/-0.62; after_pretraining: L2 0.4126, std 0.0674, bias +0.01/+0.01; online_250: L2 0.0246, std 0.0100, bias -0.09/-0.03; online_500: L2 0.0213, std 0.0100, bias -0.19/-0.15; online_750: L2 0.0233, std 0.0100, bias -0.17/-0.24; online_1000: L2 0.0180, std 0.0100, bias -0.07/-0.12; online_1250: L2 0.0105, std 0.0100, bias -0.06/-0.10; online_1500: L2 0.0066, std 0.0100, bias -0.09/+0.05; online_1750: L2 0.0068, std 0.0100, bias -0.13/-0.09; online_2000: L2 0.0227, std 0.0100, bias -0.10/-0.03.

## Frozen evaluations (in-run; 10 stochastic seeds per prefix)

| Checkpoint | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch | Cold det / stoch |
| --- | --- | --- | --- | --- |
| pretraining_baseline | fail / 0/10 | pass / 10/10 | pass / 10/10 | fail |
| online_250 | fail / 0/10 | fail / 0/10 | pass / 10/10 | fail |
| online_500 | fail / 0/10 | fail / 0/10 | pass / 10/10 | fail |
| online_750 | fail / 4/10 | pass / 8/10 | pass / 10/10 | fail |
| online_1000 | fail / 2/10 | pass / 10/10 | pass / 10/10 | fail |
| online_1250 | pass / 10/10 | pass / 10/10 | pass / 10/10 | fail |
| online_1500 | pass / 9/10 | pass / 10/10 | pass / 10/10 | fail |
| online_1750 | fail / 9/10 | pass / 10/10 | pass / 10/10 | fail |
| online_2000 | pass / 10/10 | pass / 8/10 | pass / 10/10 | fail / 0/5 |

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

| Checkpoint | Prefix 5 | Prefix 6 | Prefix 7 | Prefix 8 | Cold | Prefix-6 Wilson 95% |
| --- | --- | --- | --- | --- | --- | --- |
| online_250 | 0/30 | 4/30 | 0/30 | 30/30 D | 0/30 | [0.05, 0.30] |
| online_500 | 0/30 | 0/30 | 0/30 | 30/30 D | 0/30 | [-0.00, 0.11] |
| online_750 | 1/30 | 14/30 | 26/30 D | 30/30 D | 0/30 | [0.30, 0.64] |
| online_1000 | 0/30 | 17/30 | 30/30 D | 30/30 D | 0/30 | [0.39, 0.73] |
| online_1250 | 0/30 | 25/30 D | 30/30 D | 30/30 D | 0/30 | [0.66, 0.93] |
| online_1500 | 0/30 | 26/30 D | 30/30 D | 30/30 D | 0/30 | [0.70, 0.95] |
| online_1750 | 1/30 | 23/30 | 30/30 D | 30/30 D | 0/30 | [0.59, 0.88] |
| online_2000 | 2/30 | 27/30 D | 28/30 D | 30/30 D | 0/30 | [0.74, 0.97] |

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
