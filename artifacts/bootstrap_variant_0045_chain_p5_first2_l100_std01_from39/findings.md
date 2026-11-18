# Variant 45 `chain_p5_first2_l100_std01_from39`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_controls": 5, "prefix_set": [5, 6, 7, 8], "demonstration_include_mpc": false, "demonstration_recorded_rows": [5, 9], "actor_bc_lambda": 100.0, "actor_bc_rows": "first_k", "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "initial_from": ["bootstrap_variant_0039_bc_first_l100_std01", "online_2000"], "actor_bc_first_rows": 2, "eval_stochastic_trials": 10, "actor_bc_std_target": 0.01}`. Inherited checkpoint: `bootstrap_variant_0039_bc_first_l100_std01/checkpoints/online_2000`. Hypothesis: Chained prefix-5 stage inheriting the pinned-handoff checkpoint (bc_first_l100_std01 online_2000: prefix-6 deterministic pass and 9/10 stochastic, prefix-7/8 10/10, handoff std .01) with the same recipe at obs5 and obs6.

Stage status `complete_sac_stage` (exit=0, 733 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 1500, 'online_updates': 6000}; 500 critic-pretraining and 2000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: inherited 3e-4. Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over the first 2 rows of each demonstration episode, added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.01)) added to the same actor step. Handoff noise: off.

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'5': 82, '6': 64, '7': 58, '8': 52}, force-qualified successes {'5': 3, '6': 54, '7': 57, '8': 52} (informational; the gate is the 4-row demonstration buffer).
- Online: 2000 interactions; completed episodes by prefix {'5': 170, '6': 160, '7': 147, '8': 178}, successes {'5': 29, '6': 100, '7': 121, '8': 178}; outcomes {'drop': 198, 'success': 428, 'palm_support': 27, 'truncated': 2}; handoffs terminated before the policy 0; unfinished last suffix True. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 5885; none entered replay.
- Actor BC loss first/last: 0.03805 / 5.53e-05; spread-anchor loss first/last: 0.3313 / 1.463e-06; alpha raw-equivalent last 0.0955.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 0.4777, std 0.0225, bias -0.46/-0.43; after_pretraining: L2 0.4777, std 0.0225, bias -0.06/+0.04; online_250: L2 0.0138, std 0.0100, bias -0.06/-0.13; online_500: L2 0.0088, std 0.0100, bias -0.20/-0.22; online_750: L2 0.0280, std 0.0100, bias -0.14/-0.06; online_1000: L2 0.0527, std 0.0100, bias -0.24/-0.18; online_1250: L2 0.0275, std 0.0100, bias -0.13/-0.19; online_1500: L2 0.0143, std 0.0100, bias -0.15/-0.17; online_1750: L2 0.0351, std 0.0100, bias -0.24/-0.20; online_2000: L2 0.0123, std 0.0100, bias -0.18/-0.12.

## Frozen evaluations (in-run; 10 stochastic seeds per prefix)

| Checkpoint | Prefix-5 det / stoch | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch | Cold det / stoch |
| --- | --- | --- | --- | --- | --- |
| pretraining_baseline | fail / 0/10 | pass / 9/10 | pass / 10/10 | pass / 10/10 | fail |
| online_250 | fail / 3/10 | pass / 8/10 | pass / 10/10 | pass / 10/10 | fail |
| online_500 | fail / 4/10 | pass / 7/10 | pass / 6/10 | pass / 10/10 | fail |
| online_750 | fail / 0/10 | fail / 3/10 | fail / 1/10 | pass / 10/10 | fail |
| online_1000 | fail / 2/10 | fail / 3/10 | fail / 4/10 | pass / 10/10 | fail |
| online_1250 | pass / 4/10 | pass / 9/10 | pass / 10/10 | pass / 10/10 | fail |
| online_1500 | fail / 2/10 | pass / 10/10 | pass / 10/10 | pass / 10/10 | fail |
| online_1750 | fail / 1/10 | pass / 9/10 | pass / 7/10 | pass / 10/10 | fail |
| online_2000 | fail / 4/10 | fail / 5/10 | pass / 10/10 | pass / 10/10 | fail / 0/5 |

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

| Checkpoint | Prefix 5 | Prefix 6 | Prefix 7 | Prefix 8 | Cold | Prefix-6 Wilson 95% |
| --- | --- | --- | --- | --- | --- | --- |
| online_250 | 6/30 | 20/30 D | 30/30 D | 30/30 D | 0/30 | [0.49, 0.81] |
| online_500 | 5/30 | 16/30 D | 23/30 D | 30/30 D | 0/30 | [0.36, 0.70] |
| online_750 | 6/30 | 7/30 | 6/30 | 30/30 D | 0/30 | [0.12, 0.41] |
| online_1000 | 8/30 | 11/30 | 15/30 | 30/30 D | 0/30 | [0.22, 0.54] |
| online_1250 | 13/30 D | 26/30 D | 30/30 D | 30/30 D | 0/30 | [0.70, 0.95] |
| online_1500 | 11/30 | 23/30 D | 30/30 D | 30/30 D | 0/30 | [0.59, 0.88] |
| online_1750 | 7/30 | 21/30 D | 20/30 D | 30/30 D | 0/30 | [0.52, 0.83] |
| online_2000 | 12/30 | 14/30 | 30/30 D | 30/30 D | 0/30 | [0.30, 0.64] |

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
