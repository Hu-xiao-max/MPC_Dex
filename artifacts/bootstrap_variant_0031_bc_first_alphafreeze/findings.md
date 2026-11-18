# Variant 31 `bc_first_alphafreeze`: auto-generated findings

**Settings differing from SAC0004:** `{"demonstration_include_mpc": false, "actor_bc_lambda": 10.0, "actor_bc_rows": "first", "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "eval_stochastic_trials": 10, "alpha_freeze": true}`. Inherited checkpoint: `bootstrap_suffix_sac_0001/checkpoints/online_2000`. Hypothesis: recorded_bc10_first with the temperature frozen at its inherited value (.000306 normalized): alpha rose monotonically in every run because the policy entropy never approaches the -16 target, inflating the handoff std from .067 to .13-.17 while the mean stayed on the label; 30-trial rates drifted 20/30 -> 6/30 with more updates.

Stage status `complete_sac_stage` (exit=0, 511 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 1000, 'online_updates': 4000}; 500 critic-pretraining and 2000 online updates as prescribed. Temperature: frozen at the inherited value for this stage; alpha loss logged only, alpha optimizer untouched. Actor step: inherited 3e-4. Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over the first row of each demonstration episode, added to the single SAC actor step; spread anchor: none. Handoff noise: off.

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'6': 256}, force-qualified successes {'6': 0} (informational; the gate is the 3-row demonstration buffer).
- Online: 2000 interactions; completed episodes by prefix {'6': 432}, successes {'6': 78}; outcomes {'drop': 281, 'palm_support': 71, 'success': 78, 'truncated': 2}; handoffs terminated before the policy 0; unfinished last suffix True. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 4134; none entered replay.
- Actor BC loss first/last: 0.04802 / 4.667e-05; spread-anchor loss first/last: none; alpha raw-equivalent last 0.0612.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 0.4126, std 0.0674, bias -0.68/-0.62; after_pretraining: L2 0.4126, std 0.0674, bias -0.00/+0.01; online_250: L2 0.0338, std 0.0716, bias -0.13/-0.13; online_500: L2 0.0450, std 0.0968, bias -0.38/-0.34; online_750: L2 0.0144, std 0.1072, bias -0.21/-0.14; online_1000: L2 0.0176, std 0.1236, bias -0.14/-0.17; online_1250: L2 0.0189, std 0.1329, bias -0.05/-0.17; online_1500: L2 0.0504, std 0.1322, bias -0.13/-0.22; online_1750: L2 0.0223, std 0.1283, bias -0.14/-0.19; online_2000: L2 0.0127, std 0.1419, bias -0.09/-0.09.

## Frozen evaluations (in-run; 10 stochastic seeds per prefix)

| Checkpoint | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch | Cold det / stoch |
| --- | --- | --- | --- | --- |
| pretraining_baseline | fail / 0/10 | pass / 10/10 | pass / 10/10 | fail |
| online_250 | fail / 0/10 | fail / 1/10 | pass / 10/10 | fail |
| online_500 | pass / 0/10 | fail / 1/10 | pass / 10/10 | fail |
| online_750 | pass / 5/10 | pass / 10/10 | pass / 10/10 | fail |
| online_1000 | fail / 0/10 | pass / 8/10 | pass / 10/10 | fail |
| online_1250 | pass / 0/10 | pass / 4/10 | pass / 10/10 | fail |
| online_1500 | fail / 3/10 | pass / 6/10 | pass / 10/10 | fail |
| online_1750 | fail / 1/10 | pass / 5/10 | pass / 10/10 | fail |
| online_2000 | pass / 4/10 | pass / 6/10 | pass / 10/10 | fail / 0/5 |

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

| Checkpoint | Prefix 5 | Prefix 6 | Prefix 7 | Prefix 8 | Cold | Prefix-6 Wilson 95% |
| --- | --- | --- | --- | --- | --- | --- |
| online_250 | 0/30 | 0/30 | 3/30 | 30/30 D | 0/30 | [-0.00, 0.11] |
| online_500 | 0/30 | 14/30 D | 4/30 | 30/30 D | 0/30 | [0.30, 0.64] |
| online_750 | 0/30 | 12/30 D | 23/30 D | 30/30 D | 0/30 | [0.25, 0.58] |
| online_1000 | 0/30 | 8/30 | 29/30 D | 30/30 D | 0/30 | [0.14, 0.44] |
| online_1250 | 0/30 | 9/30 D | 22/30 D | 30/30 D | 0/30 | [0.17, 0.48] |
| online_1500 | 0/30 | 10/30 | 23/30 D | 30/30 D | 0/30 | [0.19, 0.51] |
| online_1750 | 0/30 | 3/30 | 19/30 D | 30/30 D | 0/30 | [0.03, 0.26] |
| online_2000 | 0/30 | 12/30 D | 11/30 D | 30/30 D | 0/30 | [0.25, 0.58] |

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
