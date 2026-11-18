# Variant 28 `chain_p5_first2`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_controls": 5, "prefix_set": [5, 6, 7, 8], "demonstration_include_mpc": false, "demonstration_recorded_rows": [5, 9], "actor_bc_lambda": 10.0, "actor_bc_rows": "first_k", "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "initial_from": ["bootstrap_variant_0004_recorded_bc10_first", "online_2000"], "actor_bc_first_rows": 2, "eval_stochastic_trials": 10}`. Inherited checkpoint: `bootstrap_variant_0004_recorded_bc10_first/checkpoints/online_2000`. Hypothesis: Chained prefix-5 stage keeping supervised labels at both new handoffs (obs5 -> recorded control 6 and obs6 -> recorded control 7): chain_p5_from_bc_first lost prefix-6 competence within 500 updates when only obs5 was supervised.

Stage status `complete_sac_stage` (exit=0, 811 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 1500, 'online_updates': 6000}; 500 critic-pretraining and 2000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: inherited 3e-4. Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over the first 2 rows of each demonstration episode, added to the single SAC actor step; spread anchor: none. Handoff noise: off.

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'5': 82, '6': 64, '7': 58, '8': 52}, force-qualified successes {'5': 0, '6': 40, '7': 47, '8': 52} (informational; the gate is the 4-row demonstration buffer).
- Online: 2000 interactions; completed episodes by prefix {'5': 144, '6': 135, '7': 128, '8': 153}, successes {'5': 3, '6': 25, '7': 77, '8': 153}; outcomes {'drop': 271, 'palm_support': 31, 'success': 258}; handoffs terminated before the policy 0; unfinished last suffix True. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 5272; none entered replay.
- Actor BC loss first/last: 0.02849 / 7.879e-05; spread-anchor loss first/last: none; alpha raw-equivalent last 0.0920.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 0.6237, std 0.0487, bias -0.60/-0.59; after_pretraining: L2 0.6237, std 0.0487, bias -0.06/-0.08; online_250: L2 0.0609, std 0.0764, bias -0.19/-0.17; online_500: L2 0.0921, std 0.1098, bias -0.11/-0.17; online_750: L2 0.0443, std 0.1368, bias -0.20/-0.23; online_1000: L2 0.0795, std 0.1583, bias -0.19/-0.19; online_1250: L2 0.0665, std 0.1683, bias -0.41/-0.35; online_1500: L2 0.0197, std 0.1990, bias -0.16/-0.13; online_1750: L2 0.0278, std 0.1909, bias -0.39/-0.30; online_2000: L2 0.0204, std 0.2171, bias -0.20/-0.24.

## Frozen evaluations (in-run; 10 stochastic seeds per prefix)

| Checkpoint | Prefix-5 det / stoch | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch | Cold det / stoch |
| --- | --- | --- | --- | --- | --- |
| pretraining_baseline | fail / 0/10 | pass / 6/10 | pass / 10/10 | pass / 10/10 | fail |
| online_250 | fail / 0/10 | fail / 2/10 | fail / 4/10 | pass / 10/10 | fail |
| online_500 | fail / 0/10 | pass / 2/10 | pass / 10/10 | pass / 10/10 | fail |
| online_750 | fail / 1/10 | fail / 0/10 | pass / 9/10 | pass / 10/10 | fail |
| online_1000 | fail / 0/10 | fail / 6/10 | pass / 8/10 | pass / 10/10 | fail |
| online_1250 | fail / 1/10 | fail / 2/10 | pass / 10/10 | pass / 10/10 | fail |
| online_1500 | fail / 1/10 | fail / 2/10 | pass / 7/10 | pass / 10/10 | fail |
| online_1750 | fail / 0/10 | fail / 0/10 | fail / 0/10 | pass / 10/10 | fail |
| online_2000 | fail / 0/10 | pass / 2/10 | pass / 6/10 | pass / 10/10 | fail / 0/5 |

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

| Checkpoint | Prefix 5 | Prefix 6 | Prefix 7 | Prefix 8 | Cold | Prefix-6 Wilson 95% |
| --- | --- | --- | --- | --- | --- | --- |
| online_250 | 0/30 | 5/30 | 21/30 | 30/30 D | 0/30 | [0.07, 0.34] |
| online_500 | 0/30 | 2/30 D | 26/30 D | 30/30 D | 0/30 | [0.02, 0.21] |
| online_750 | 6/30 | 4/30 | 25/30 D | 30/30 D | 0/30 | [0.05, 0.30] |
| online_1000 | 0/30 | 11/30 | 23/30 D | 30/30 D | 0/30 | [0.22, 0.54] |
| online_1250 | 0/30 | 3/30 | 23/30 D | 30/30 D | 0/30 | [0.03, 0.26] |
| online_1500 | 2/30 | 5/30 | 17/30 D | 30/30 D | 0/30 | [0.07, 0.34] |
| online_1750 | 0/30 | 0/30 | 0/30 | 30/30 D | 0/30 | [-0.00, 0.11] |
| online_2000 | 0/30 | 11/30 D | 21/30 D | 30/30 D | 0/30 | [0.22, 0.54] |

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
