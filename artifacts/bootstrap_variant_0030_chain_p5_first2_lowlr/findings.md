# Variant 30 `chain_p5_first2_lowlr`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_controls": 5, "prefix_set": [5, 6, 7, 8], "demonstration_include_mpc": false, "demonstration_recorded_rows": [5, 9], "actor_bc_lambda": 10.0, "actor_bc_rows": "first_k", "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "initial_from": ["bootstrap_variant_0004_recorded_bc10_first", "online_2000"], "actor_bc_first_rows": 2, "actor_lr_override": 0.0001, "eval_stochastic_trials": 10}`. Inherited checkpoint: `bootstrap_variant_0004_recorded_bc10_first/checkpoints/online_2000`. Hypothesis: chain_p5_first2 with the smaller actor step.

Stage status `complete_sac_stage` (exit=0, 576 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 1500, 'online_updates': 6000}; 500 critic-pretraining and 2000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: overridden to 0.0001 for this stage (ramp fraction unchanged). Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over the first 2 rows of each demonstration episode, added to the single SAC actor step; spread anchor: none. Handoff noise: off.

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'5': 82, '6': 64, '7': 58, '8': 52}, force-qualified successes {'5': 0, '6': 40, '7': 47, '8': 52} (informational; the gate is the 4-row demonstration buffer).
- Online: 2000 interactions; completed episodes by prefix {'5': 142, '6': 133, '7': 127, '8': 148}, successes {'5': 5, '6': 23, '7': 81, '8': 148}; outcomes {'drop': 245, 'palm_support': 48, 'success': 257}; handoffs terminated before the policy 0; unfinished last suffix True. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 5202; none entered replay.
- Actor BC loss first/last: 0.02849 / 0.0001006; spread-anchor loss first/last: none; alpha raw-equivalent last 0.0925.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 0.6237, std 0.0487, bias -0.60/-0.59; after_pretraining: L2 0.6237, std 0.0487, bias -0.06/-0.08; online_250: L2 0.1215, std 0.0578, bias -0.09/-0.08; online_500: L2 0.0736, std 0.0640, bias -0.16/-0.12; online_750: L2 0.0024, std 0.0706, bias -0.11/-0.08; online_1000: L2 0.0154, std 0.0770, bias -0.11/-0.09; online_1250: L2 0.0153, std 0.0869, bias -0.26/-0.23; online_1500: L2 0.0278, std 0.0933, bias -0.17/-0.18; online_1750: L2 0.0492, std 0.1040, bias -0.10/-0.09; online_2000: L2 0.0182, std 0.1102, bias -0.25/-0.25.

## Frozen evaluations (in-run; 10 stochastic seeds per prefix)

| Checkpoint | Prefix-5 det / stoch | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch | Cold det / stoch |
| --- | --- | --- | --- | --- | --- |
| pretraining_baseline | fail / 0/10 | pass / 6/10 | pass / 10/10 | pass / 10/10 | fail |
| online_250 | fail / 0/10 | fail / 1/10 | fail / 8/10 | pass / 10/10 | fail |
| online_500 | fail / 1/10 | fail / 2/10 | pass / 9/10 | pass / 10/10 | fail |
| online_750 | fail / 0/10 | fail / 3/10 | pass / 7/10 | pass / 10/10 | fail |
| online_1000 | fail / 2/10 | fail / 1/10 | pass / 8/10 | pass / 10/10 | fail |
| online_1250 | fail / 1/10 | fail / 3/10 | pass / 3/10 | pass / 10/10 | fail |
| online_1500 | fail / 0/10 | fail / 0/10 | pass / 7/10 | pass / 10/10 | fail |
| online_1750 | fail / 1/10 | pass / 3/10 | pass / 5/10 | pass / 10/10 | fail |
| online_2000 | fail / 0/10 | pass / 2/10 | pass / 9/10 | pass / 10/10 | fail / 0/5 |

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

| Checkpoint | Prefix 5 | Prefix 6 | Prefix 7 | Prefix 8 | Cold | Prefix-6 Wilson 95% |
| --- | --- | --- | --- | --- | --- | --- |
| online_250 | 0/30 | 2/30 | 20/30 | 30/30 D | 0/30 | [0.02, 0.21] |
| online_500 | 1/30 | 12/30 | 27/30 D | 30/30 D | 0/30 | [0.25, 0.58] |
| online_750 | 3/30 | 13/30 | 21/30 D | 30/30 D | 0/30 | [0.27, 0.61] |
| online_1000 | 3/30 | 3/30 | 22/30 D | 30/30 D | 0/30 | [0.03, 0.26] |
| online_1250 | 4/30 | 2/30 | 21/30 D | 30/30 D | 0/30 | [0.02, 0.21] |
| online_1500 | 0/30 | 7/30 | 16/30 D | 30/30 D | 0/30 | [0.12, 0.41] |
| online_1750 | 2/30 | 4/30 D | 18/30 D | 30/30 D | 0/30 | [0.05, 0.30] |
| online_2000 | 6/30 | 16/30 D | 24/30 D | 30/30 D | 0/30 | [0.36, 0.70] |

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
