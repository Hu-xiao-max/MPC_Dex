# Variant 36 `bc_first_l100`: auto-generated findings

**Settings differing from SAC0004:** `{"demonstration_include_mpc": false, "actor_bc_lambda": 100.0, "actor_bc_rows": "first", "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "eval_stochastic_trials": 10}`. Inherited checkpoint: `bootstrap_suffix_sac_0001/checkpoints/online_2000`. Hypothesis: Ten times stronger first-row supervision: 30-trial collapses coincide with the handoff mean drifting to L2 .09 from the label (0019 online_2000) in a basin about .025-.05 wide.

Stage status `complete_sac_stage` (exit=0, 642 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 1000, 'online_updates': 4000}; 500 critic-pretraining and 2000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: inherited 3e-4. Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over the first row of each demonstration episode, added to the single SAC actor step; spread anchor: none. Handoff noise: off.

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'6': 256}, force-qualified successes {'6': 0} (informational; the gate is the 3-row demonstration buffer).
- Online: 2000 interactions; completed episodes by prefix {'6': 428}, successes {'6': 165}; outcomes {'drop': 222, 'palm_support': 40, 'success': 165, 'truncated': 1}; handoffs terminated before the policy 0; unfinished last suffix True. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 4110; none entered replay.
- Actor BC loss first/last: 0.04802 / 3.02e-05; spread-anchor loss first/last: none; alpha raw-equivalent last 0.0752.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 0.4126, std 0.0674, bias -0.68/-0.62; after_pretraining: L2 0.4126, std 0.0674, bias -0.00/+0.01; online_250: L2 0.0125, std 0.1386, bias -0.03/-0.08; online_500: L2 0.0052, std 0.0902, bias -0.25/-0.25; online_750: L2 0.0089, std 0.1219, bias -0.16/-0.06; online_1000: L2 0.0115, std 0.1278, bias -0.26/-0.20; online_1250: L2 0.0256, std 0.1375, bias -0.11/-0.13; online_1500: L2 0.0186, std 0.1340, bias -0.15/-0.17; online_1750: L2 0.0146, std 0.1286, bias -0.16/-0.18; online_2000: L2 0.0091, std 0.1335, bias -0.22/-0.13.

## Frozen evaluations (in-run; 10 stochastic seeds per prefix)

| Checkpoint | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch | Cold det / stoch |
| --- | --- | --- | --- | --- |
| pretraining_baseline | fail / 0/10 | pass / 10/10 | pass / 10/10 | fail |
| online_250 | fail / 2/10 | fail / 0/10 | pass / 10/10 | fail |
| online_500 | pass / 3/10 | fail / 3/10 | pass / 10/10 | fail |
| online_750 | pass / 7/10 | pass / 10/10 | pass / 10/10 | fail |
| online_1000 | pass / 6/10 | pass / 8/10 | pass / 10/10 | fail |
| online_1250 | fail / 5/10 | pass / 10/10 | pass / 10/10 | fail |
| online_1500 | fail / 3/10 | fail / 5/10 | pass / 10/10 | fail |
| online_1750 | fail / 4/10 | pass / 10/10 | pass / 10/10 | fail |
| online_2000 | fail / 2/10 | pass / 8/10 | pass / 10/10 | fail / 0/5 |

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

| Checkpoint | Prefix 5 | Prefix 6 | Prefix 7 | Prefix 8 | Cold | Prefix-6 Wilson 95% |
| --- | --- | --- | --- | --- | --- | --- |
| online_250 | 0/30 | 0/30 | 0/30 | 30/30 D | 0/30 | [-0.00, 0.11] |
| online_500 | 0/30 | 11/30 D | 14/30 | 30/30 D | 0/30 | [0.22, 0.54] |
| online_750 | 0/30 | 19/30 D | 27/30 D | 30/30 D | 0/30 | [0.46, 0.78] |
| online_1000 | 5/30 | 16/30 D | 29/30 D | 30/30 D | 0/30 | [0.36, 0.70] |
| online_1250 | 5/30 | 12/30 | 30/30 D | 30/30 D | 0/30 | [0.25, 0.58] |
| online_1500 | 3/30 | 8/30 | 18/30 | 30/30 D | 0/30 | [0.14, 0.44] |
| online_1750 | 1/30 | 19/30 | 24/30 D | 30/30 D | 0/30 | [0.46, 0.78] |
| online_2000 | 3/30 | 12/30 | 20/30 D | 30/30 D | 0/30 | [0.25, 0.58] |

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
