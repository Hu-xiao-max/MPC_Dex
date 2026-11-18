# Variant 26 `bc_first_lowlr`: auto-generated findings

**Settings differing from SAC0004:** `{"demonstration_include_mpc": false, "actor_bc_lambda": 10.0, "actor_bc_rows": "first", "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "actor_lr_override": 0.0001, "eval_stochastic_trials": 10}`. Inherited checkpoint: `bootstrap_suffix_sac_0001/checkpoints/online_2000`. Hypothesis: recorded_bc10_first with a three times smaller actor step: prefix-6 competence flickered between checkpoints in every BC-first run while training success kept rising, consistent with actor drift; ten stochastic trials per prefix reduce evaluation noise.

Stage status `complete_sac_stage` (exit=0, 499 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 1000, 'online_updates': 4000}; 500 critic-pretraining and 2000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: overridden to 0.0001 for this stage (ramp fraction unchanged). Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over the first row of each demonstration episode, added to the single SAC actor step; spread anchor: none. Handoff noise: off.

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'6': 256}, force-qualified successes {'6': 0} (informational; the gate is the 3-row demonstration buffer).
- Online: 2000 interactions; completed episodes by prefix {'6': 444}, successes {'6': 155}; outcomes {'drop': 223, 'success': 155, 'palm_support': 65, 'truncated': 1}; handoffs terminated before the policy 0; unfinished last suffix True. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 4206; none entered replay.
- Actor BC loss first/last: 0.04802 / 7.602e-05; spread-anchor loss first/last: none; alpha raw-equivalent last 0.0755.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 0.4126, std 0.0674, bias -0.68/-0.62; after_pretraining: L2 0.4126, std 0.0674, bias -0.00/+0.01; online_250: L2 0.0563, std 0.0661, bias -0.06/-0.08; online_500: L2 0.0250, std 0.0657, bias -0.18/-0.21; online_750: L2 0.0245, std 0.0705, bias -0.02/-0.09; online_1000: L2 0.0377, std 0.0790, bias -0.06/-0.09; online_1250: L2 0.0222, std 0.0886, bias -0.08/-0.05; online_1500: L2 0.0201, std 0.1052, bias -0.14/-0.15; online_1750: L2 0.0271, std 0.1225, bias +0.01/-0.06; online_2000: L2 0.0214, std 0.1255, bias -0.15/-0.06.

## Frozen evaluations (in-run; 10 stochastic seeds per prefix)

| Checkpoint | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch | Cold det / stoch |
| --- | --- | --- | --- | --- |
| pretraining_baseline | fail / 0/10 | pass / 10/10 | pass / 10/10 | fail |
| online_250 | fail / 0/10 | fail / 0/10 | pass / 10/10 | fail |
| online_500 | fail / 0/10 | fail / 1/10 | pass / 10/10 | fail |
| online_750 | pass / 3/10 | pass / 7/10 | pass / 10/10 | fail |
| online_1000 | fail / 0/10 | pass / 8/10 | pass / 10/10 | fail |
| online_1250 | pass / 5/10 | pass / 9/10 | pass / 10/10 | fail |
| online_1500 | fail / 2/10 | pass / 9/10 | pass / 10/10 | fail |
| online_1750 | fail / 2/10 | pass / 9/10 | pass / 10/10 | fail |
| online_2000 | pass / 3/10 | pass / 8/10 | pass / 10/10 | fail / 0/5 |

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

| Checkpoint | Prefix 5 | Prefix 6 | Prefix 7 | Prefix 8 | Cold | Prefix-6 Wilson 95% |
| --- | --- | --- | --- | --- | --- | --- |
| online_250 | 0/30 | 3/30 | 4/30 | 30/30 D | 0/30 | [0.03, 0.26] |
| online_500 | 0/30 | 3/30 | 2/30 | 30/30 D | 0/30 | [0.03, 0.26] |
| online_750 | 0/30 | 17/30 D | 20/30 D | 30/30 D | 0/30 | [0.39, 0.73] |
| online_1000 | 1/30 | 2/30 | 29/30 D | 30/30 D | 0/30 | [0.02, 0.21] |
| online_1250 | 0/30 | 20/30 D | 29/30 D | 30/30 D | 0/30 | [0.49, 0.81] |
| online_1500 | 0/30 | 5/30 | 27/30 D | 30/30 D | 0/30 | [0.07, 0.34] |
| online_1750 | 0/30 | 10/30 | 21/30 D | 30/30 D | 0/30 | [0.19, 0.51] |
| online_2000 | 0/30 | 8/30 D | 23/30 D | 30/30 D | 0/30 | [0.14, 0.44] |

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
