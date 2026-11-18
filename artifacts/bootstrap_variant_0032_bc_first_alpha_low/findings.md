# Variant 32 `bc_first_alpha_low`: auto-generated findings

**Settings differing from SAC0004:** `{"demonstration_include_mpc": false, "actor_bc_lambda": 10.0, "actor_bc_rows": "first", "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "eval_stochastic_trials": 10, "alpha_freeze": true, "alpha_override": 0.0001}`. Inherited checkpoint: `bootstrap_suffix_sac_0001/checkpoints/online_2000`. Hypothesis: Temperature frozen at a third of the inherited value (raw-equivalent .02), weakening the entropy bonus so SAC can shrink the std where failures occur.

Stage status `complete_sac_stage` (exit=0, 493 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 1000, 'online_updates': 4000}; 500 critic-pretraining and 2000 online updates as prescribed. Temperature: frozen at the override value for this stage; alpha loss logged only, alpha optimizer untouched. Actor step: inherited 3e-4. Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over the first row of each demonstration episode, added to the single SAC actor step; spread anchor: none. Handoff noise: off.

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'6': 256}, force-qualified successes {'6': 0} (informational; the gate is the 3-row demonstration buffer).
- Online: 2000 interactions; completed episodes by prefix {'6': 400}, successes {'6': 80}; outcomes {'drop': 228, 'success': 80, 'palm_support': 90, 'truncated': 2}; handoffs terminated before the policy 0; unfinished last suffix False. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 3936; none entered replay.
- Actor BC loss first/last: 0.04802 / 0.0003075; spread-anchor loss first/last: none; alpha raw-equivalent last 0.0200.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 0.4126, std 0.0674, bias -0.68/-0.62; after_pretraining: L2 0.4126, std 0.0674, bias -0.00/+0.01; online_250: L2 0.0474, std 0.0586, bias -0.10/-0.09; online_500: L2 0.0381, std 0.0604, bias -0.35/-0.34; online_750: L2 0.0218, std 0.0670, bias -0.41/-0.41; online_1000: L2 0.0388, std 0.0726, bias -0.34/-0.43; online_1250: L2 0.0110, std 0.0864, bias -0.30/-0.34; online_1500: L2 0.0392, std 0.0807, bias -0.13/-0.27; online_1750: L2 0.0777, std 0.0833, bias -0.11/-0.08; online_2000: L2 0.0576, std 0.0835, bias -0.04/-0.07.

## Frozen evaluations (in-run; 10 stochastic seeds per prefix)

| Checkpoint | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch | Cold det / stoch |
| --- | --- | --- | --- | --- |
| pretraining_baseline | fail / 0/10 | pass / 10/10 | pass / 10/10 | fail |
| online_250 | fail / 0/10 | fail / 0/10 | pass / 10/10 | fail |
| online_500 | fail / 1/10 | pass / 5/10 | pass / 10/10 | fail |
| online_750 | pass / 1/10 | fail / 0/10 | pass / 10/10 | fail |
| online_1000 | fail / 2/10 | pass / 6/10 | pass / 10/10 | fail |
| online_1250 | fail / 1/10 | fail / 3/10 | pass / 10/10 | fail |
| online_1500 | fail / 1/10 | fail / 3/10 | pass / 10/10 | fail |
| online_1750 | fail / 0/10 | fail / 2/10 | pass / 10/10 | fail |
| online_2000 | fail / 2/10 | pass / 3/10 | pass / 10/10 | fail / 0/5 |

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

| Checkpoint | Prefix 5 | Prefix 6 | Prefix 7 | Prefix 8 | Cold | Prefix-6 Wilson 95% |
| --- | --- | --- | --- | --- | --- | --- |
| online_250 | 0/30 | 1/30 | 4/30 | 30/30 D | 0/30 | [0.01, 0.17] |
| online_500 | 0/30 | 3/30 | 15/30 D | 30/30 D | 0/30 | [0.03, 0.26] |
| online_750 | 0/30 | 0/30 D | 1/30 | 30/30 D | 0/30 | [-0.00, 0.11] |
| online_1000 | 0/30 | 7/30 | 24/30 D | 30/30 D | 0/30 | [0.12, 0.41] |
| online_1250 | 0/30 | 7/30 | 13/30 | 30/30 D | 0/30 | [0.12, 0.41] |
| online_1500 | 0/30 | 1/30 | 12/30 | 30/30 D | 0/30 | [0.01, 0.17] |
| online_1750 | 0/30 | 2/30 | 12/30 | 30/30 D | 0/30 | [0.02, 0.21] |
| online_2000 | 0/30 | 2/30 | 14/30 D | 30/30 D | 0/30 | [0.02, 0.21] |

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
