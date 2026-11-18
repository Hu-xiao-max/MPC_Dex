# Variant 69 `ms_suffix123_noise_from61`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_set": [6, 7, 8], "demonstration_include_mpc": false, "actor_bc_lambda": 100.0, "handoff_noise_sigma": 0.015, "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "initial_from": ["bootstrap_variant_0061_chain_p3_cont_from59", "online_1750"], "actor_lr_override": 0.0001, "actor_bc_std_target": 0.01, "prefix_sources": ["mpc_cold_demonstrations_0001", 15], "suffix_set": [1, 2, 3], "demonstration_suffix_rows": 3, "handoff_noise_suffixes": [2, 3]}`. Inherited checkpoint: `bootstrap_variant_0061_chain_p3_cont_from59/checkpoints/online_1750`. Hypothesis: As ms_suffix123_from61 with sigma .015 noise on the handoff control for suffixes 2 and 3.

Stage status `complete_sac_stage` (exit=0, 1474 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 3500, 'online_updates': 13750}; 500 critic-pretraining and 2000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: overridden to 0.0001 for this stage (inherited 0.0001; ramp fraction unchanged). Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over all demonstration rows, added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.01)) added to the same actor step. Handoff noise: off.

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'s1': 99, 's2+noise': 72, 's3+noise': 85}, force-qualified successes {'s1': 39, 's2+noise': 6, 's3+noise': 8} (informational; the gate is the 45-row demonstration buffer).
- Online: 2000 interactions; completed episodes by prefix {'s1': 125, 's2+noise': 121, 's3+noise': 119}, successes {'s1': 98, 's2+noise': 44, 's3+noise': 16}; outcomes {'drop': 124, 'success': 158, 'truncated': 9, 'palm_support': 74}; handoffs terminated before the policy 3; unfinished last suffix True. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 10539; none entered replay.
- Actor BC loss first/last: 3.882 / 0.004801; spread-anchor loss first/last: 0.636 / 5.534e-05; alpha raw-equivalent last 0.2114.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 4.2693, std 0.0257, bias -1.51/-1.71; after_pretraining: L2 4.2693, std 0.0257, bias +0.03/+0.03; online_250: L2 1.7437, std 0.0085, bias -0.19/-0.21; online_500: L2 0.6595, std 0.0105, bias -0.17/-0.18; online_750: L2 0.3521, std 0.0104, bias -0.23/-0.24; online_1000: L2 0.2456, std 0.0103, bias -0.13/-0.16; online_1250: L2 0.0870, std 0.0103, bias -0.12/-0.13; online_1500: L2 0.0552, std 0.0101, bias -0.22/-0.23; online_1750: L2 0.0256, std 0.0100, bias -0.19/-0.20; online_2000: L2 0.0129, std 0.0100, bias -0.20/-0.23.

## Frozen evaluations (in-run; 5 stochastic seeds per prefix)

| Checkpoint | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch | Cold det / stoch |
| --- | --- | --- | --- | --- |
| pretraining_baseline | pass / 4/5 | pass / 5/5 | pass / 5/5 | fail |
| online_250 | fail / 0/5 | fail / 0/5 | pass / 5/5 | fail |
| online_500 | fail / 0/5 | fail / 0/5 | pass / 5/5 | fail |
| online_750 | fail / 0/5 | fail / 0/5 | pass / 5/5 | fail |
| online_1000 | fail / 0/5 | fail / 0/5 | pass / 5/5 | fail |
| online_1250 | fail / 0/5 | fail / 0/5 | pass / 5/5 | fail |
| online_1500 | fail / 0/5 | fail / 0/5 | pass / 5/5 | fail |
| online_1750 | fail / 0/5 | fail / 0/5 | pass / 5/5 | fail |
| online_2000 | fail / 0/5 | fail / 0/5 | pass / 5/5 | fail / 0/5 |

The cold column counts only the cold-reset trials (an earlier rendering of this table had added the multi-source cells to it; corrected 2026-10-03).

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

No large-sample result for this stage at the time of writing.

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
