# Variant 85 `ms_spine193_nosarsa_from82`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_set": [8], "demonstration_include_mpc": false, "actor_bc_lambda": 100.0, "actor_bc_rows": "per_item", "online_policy_interactions": 4000, "checkpoints": [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], "initial_from": ["bootstrap_variant_0082_ms_shared_prefix_from81", "online_4000"], "actor_lr_override": 0.0001, "eval_stochastic_trials": 3, "actor_bc_std_target": 0.01, "prefix_sources": ["mpc_cold_demonstrations_0001", 120], "suffix_set": [1, 2, 3, 4, 5, 6, 7, 8], "demonstration_suffix_rows": 40, "suffix_weights": [2, 2, 2, 2, 1, 1, 1, 1], "holdout_source_names": ["seed_197"], "eval_source_names": ["seed_100"], "start_prefix_set": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], "start_prefix_share": 0.5, "actor_bc_last_rows": 8, "spine_source": "seed_193", "actor_bc_shared_lambda": 50.0}`. Inherited checkpoint: `bootstrap_variant_0082_ms_shared_prefix_from81/checkpoints/online_4000`. Hypothesis: Ablation of index 84 without the online demonstration backup (standard SAC targets on demonstration rows, as in every earlier stage): isolates the contribution of the value ridge along the demonstration paths from the spine curriculum and the stronger pins. Caveat: the backup differs from the standard one in two ways at once, the recorded successor instead of a policy sample and no entropy term (alpha * log pi is about 0.3-0.5 normalized units per step at the inherited temperature and spread), so this pair attributes their joint effect; stochastic in-run cells use index-derived seeds and only the 30-trial frozen evaluations on fixed seeds support stochastic comparisons.

Stage status `complete_sac_stage` (exit=0 (inferred from summary.json and all eight checkpoints being written; the launcher did not capture the exit status), 2277 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 6500, 'online_updates': 35750}; 500 critic-pretraining and 4000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: overridden to 0.0001 for this stage (inherited 0.0001; ramp fraction unchanged). Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over the last 8 rows of each demonstration episode plus its shared cold-start prefix rows; additionally, per distinct shared state, lambda_shared 50.0 * (MSE of the actor mean to the label + MSE of the log-std to log(std target)) summed over the distinct shared states (one term per state, however many sources sit on it), added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.01)) added to the same actor step. Handoff noise: off.

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'s1': 22, 's2': 13, 's3': 16, 's4': 13, 's5': 11, 's6': 11, 's7': 12, 's8': 16, 'k0': 16, 'k1': 8, 'k2': 8, 'k3': 11, 'k4': 9, 'k5': 8, 'k6': 7, 'k7': 10, 'k8': 14, 'k9': 9, 'k10': 6, 'k11': 20, 'k12': 5, 'k13': 11}, force-qualified successes {'s1': 22, 's2': 9, 's3': 8, 's4': 3, 's5': 1, 's6': 0, 's7': 0, 's8': 1, 'k0': 0, 'k1': 0, 'k2': 0, 'k3': 0, 'k4': 0, 'k5': 0, 'k6': 0, 'k7': 0, 'k8': 0, 'k9': 1, 'k10': 5, 'k11': 20, 'k12': 5, 'k13': 11} (informational; the gate is the 1081-row demonstration buffer).
- Online: 4000 interactions; completed episodes by prefix {'s1': 74, 's2': 81, 's3': 80, 's4': 75, 's5': 44, 's6': 31, 's7': 49, 's8': 42, 'k0': 42, 'k1': 34, 'k2': 37, 'k3': 33, 'k4': 37, 'k5': 37, 'k6': 30, 'k7': 28, 'k8': 36, 'k9': 25, 'k10': 38, 'k11': 29, 'k12': 39, 'k13': 40}, successes {'s1': 71, 's2': 55, 's3': 29, 's4': 13, 's5': 4, 's6': 0, 's7': 4, 's8': 1, 'k0': 0, 'k1': 0, 'k2': 0, 'k3': 0, 'k4': 0, 'k5': 0, 'k6': 0, 'k7': 0, 'k8': 0, 'k9': 2, 'k10': 38, 'k11': 29, 'k12': 39, 'k13': 40}; outcomes {'success': 325, 'drop': 397, 'palm_support': 227, 'truncated': 12}; handoffs terminated before the policy 0; unfinished last suffix True. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 11758; none entered replay.
- Actor BC loss first/last: 0.01563 / 0.008444; spread-anchor loss first/last: 8.267e-06 / 4.1e-05; alpha raw-equivalent last 1.9141.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 1.3397, std 0.0100, bias -3.64/-3.65; after_pretraining: L2 1.3397, std 0.0100, bias -3.08/-3.07; online_500: L2 1.4534, std 0.0100, bias -3.60/-3.65; online_1000: L2 1.3294, std 0.0100, bias -4.29/-4.30; online_1500: L2 1.2712, std 0.0100, bias -4.50/-4.47; online_2000: L2 1.2794, std 0.0100, bias -4.86/-4.85; online_2500: L2 1.2760, std 0.0100, bias -5.00/-5.03; online_3000: L2 1.2787, std 0.0100, bias -5.29/-5.30; online_3500: L2 1.2451, std 0.0100, bias -5.38/-5.44; online_4000: L2 1.3823, std 0.0100, bias -5.49/-5.49.

## Frozen evaluations (in-run; 3 stochastic seeds per prefix)

| Checkpoint | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch | Cold det / stoch |
| --- | --- | --- | --- | --- |
| pretraining_baseline | fail / 0/3 | fail / 0/3 | pass / 3/3 | fail / 0/5 |
| online_500 | fail / 0/3 | fail / 0/3 | pass / 3/3 | fail / 0/5 |
| online_1000 | fail / 0/3 | fail / 0/3 | pass / 3/3 | fail / 0/5 |
| online_1500 | fail / 0/3 | fail / 0/3 | pass / 3/3 | fail / 0/5 |
| online_2000 | fail / 0/3 | fail / 0/3 | pass / 3/3 | fail / 0/5 |
| online_2500 | fail / 0/3 | fail / 0/3 | pass / 3/3 | fail / 0/5 |
| online_3000 | fail / 0/3 | fail / 0/3 | pass / 3/3 | fail / 0/5 |
| online_3500 | fail / 0/3 | fail / 0/3 | pass / 3/3 | fail / 0/5 |
| online_4000 | fail / 0/3 | fail / 0/3 | pass / 3/3 | fail / 0/5 |

## Multi-source evaluation (in-run; a cell is the deterministic trial P/F followed by stochastic successes/trials; per source the cells run over the suffix lengths; shared k = the state after k shared cold-start controls)

| Checkpoint | Cold | Shared k1 | Shared k2 | Shared k3 | Shared k4 | Shared k5 | Shared k6 | Shared k7 | Shared k8 | Shared k9 | Shared k10 | Shared k11 | Shared k12 | Shared k13 | seed_100 s1..s8 | seed_197 s1..s8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pretraining_baseline | F0/5 | F0/3 | F0/3 | F0/3 | F0/3 | F0/3 | F0/3 | F0/3 | F0/3 | F0/3 | P3/3 | P3/3 | P3/3 | P3/3 | P3/3 P3/3 F2/3 F1/3 F1/3 F1/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F1/3 F0/3 F0/3 F0/3 F0/3 |
| online_500 | F0/5 | F0/3 | F0/3 | F0/3 | F0/3 | F0/3 | F0/3 | F0/3 | F0/3 | F0/3 | P3/3 | P3/3 | P3/3 | P3/3 | P3/3 P3/3 P3/3 F1/3 F2/3 F1/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_1000 | F0/5 | F0/3 | F0/3 | F0/3 | F0/3 | F0/3 | F0/3 | F0/3 | F0/3 | F0/3 | P3/3 | P3/3 | P3/3 | P3/3 | P3/3 F2/3 P3/3 F2/3 F1/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_1500 | F0/5 | F0/3 | F0/3 | F0/3 | F0/3 | F0/3 | F0/3 | F0/3 | F0/3 | F0/3 | P3/3 | P3/3 | P3/3 | P3/3 | P3/3 F0/3 P3/3 F1/3 F1/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_2000 | F0/5 | F0/3 | F0/3 | F0/3 | F0/3 | F0/3 | F0/3 | F0/3 | F0/3 | F0/3 | P3/3 | P3/3 | P3/3 | P3/3 | P3/3 P0/3 P0/3 F1/3 F2/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_2500 | F0/5 | F0/3 | F0/3 | F0/3 | F0/3 | F0/3 | F0/3 | F0/3 | F0/3 | F0/3 | P3/3 | P3/3 | P3/3 | P3/3 | P3/3 P3/3 F1/3 F1/3 F2/3 F0/3 F0/3 F1/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_3000 | F0/5 | F0/3 | F0/3 | F0/3 | F0/3 | F0/3 | F0/3 | F0/3 | F1/3 | P1/3 | P3/3 | P3/3 | P3/3 | P3/3 | P3/3 P3/3 F2/3 F0/3 F1/3 F0/3 F0/3 F1/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_3500 | F0/5 | F0/3 | F0/3 | F0/3 | F0/3 | F0/3 | F0/3 | F0/3 | F0/3 | F0/3 | P3/3 | P3/3 | P3/3 | P3/3 | P3/3 P3/3 F1/3 P2/3 F1/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_4000 | F0/5 | F0/3 | F0/3 | F0/3 | F0/3 | F0/3 | F0/3 | F0/3 | F0/3 | P0/3 | P3/3 | P3/3 | P3/3 | P3/3 | P3/3 P3/3 P2/3 P1/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

No large-sample result for this stage at the time of writing.

## Spine states, cold reset and the learning-signal repairs: what was measured

The spine is seed_193 (14 controls). Half of the episodes handed off at its prefixes k = 0..13 (k = 0 the cold reset; k = 1-3 states shared with 33, 7 and 3 other sources), the other half were weighted end-suffix episodes over all 63 sources. 30-trial frozen evaluation of online_4000 at the cold reset and at every spine prefix (`frozen_checkpoint_evaluation_0001/results_shared/bootstrap_variant_0085_ms_spine193_nosarsa_from82/online_4000.json`; 1 deterministic + 30 stochastic fresh-seed trials per cell, `D` = deterministic pass):

| checkpoint | k0 (cold) | k1 | k2 | k3 | k4 | k5 | k6 | k7 | k8 | k9 | k10 | k11 | k12 | k13 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `0082` online_4000 (inherited; probe, 3 trials) | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 3/3 D | 3/3 D | 3/3 D | 3/3 D |
| `0084` online_4000 (spine + online demonstration backup + per-state pins) | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 1/30 D | 30/30 D | 30/30 D | 30/30 D | 30/30 D |
| `0085` online_4000 (without the online demonstration backup) | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 0/30 | 1/30 D | 30/30 D | 30/30 D | 30/30 D | 30/30 D |
| `0086` online_4000 (without the spine; shared groups k = 0-3) | 0/30 | 0/30 | 0/30 | 0/30 | – | – | – | – | – | – | – | – | – | – |

Online training episodes from spine prefixes k = 0-8: 0 successes in 305 (`0084`) and 314 (`0085`); k9 4/25 and 2/25; k10-k13 137/142 and 146/146. `0086`: 0 successes in 622 shared-state episodes. The frontier moved by one control in 4,000 updates (from k10 to k9, the deterministic trial), as in the recorded-trajectory chain of stages `0057`-`0067`.

**Diagnostics.**

- The per-state pins work where no on-policy data contradicts them: at every spine state from row 4 on, the actor mean lies within 0.00-0.05 of the recorded control (latent std .010) from online_500 onward, in both `0084` and `0085`. At rows 1-3 (the shared states, where the stream puts hundreds of on-policy failures) the mean sits 1.3-1.5 away throughout training and at the cold reset 0.1-0.5: the SAC actor term outweighs a weight-50 pin where the critic's action gradient is steep.
- The value ridge did not form. Under the inherited critic the recorded path had Q(s, a_rec) of -0.9 (rows 0-2) and -3.8 to -1.8 (rows 3-9) against a return of +1.57; after 4,000 online updates it is -2.7 and -5.6 to -4.0 in `0084` and -2.3 and -6.1 to -4.0 in `0085`: lower, not higher, with and without the online demonstration backup. The on-policy episodes from the spine prefixes emit controls within 0.00-0.05 of the recorded ones and still fail, so the critic is asked to separate (s_k, a_k) -> success from (s_k, a_k + 0.02) -> failure at every spine state; it fits the far more numerous failures.
- The reason is the open-loop structure of the demonstrations ([spine probe](../frozen_checkpoint_evaluation_0001/spine_probe/README.md)): replaying seed_193 after perturbing one control by L2 0.003 fails at controls 0-7 in every trial and 0.0003 at control 0 fails as well (the state diverges only after 9-10 controls at a contact event); tolerance appears at control 9 (0.003: 3/4) and is wide from control 10 (0.03: 4/4). The policy's success along the spine coincides exactly with that open-loop tolerance (controls 10-13 at 30/30, control 9 at 1/30 with a deterministic pass, nothing before), so what the chain stages learn is the open-loop tolerance of the last few controls, not feedback. The MPC completes turns from any of these states because it re-plans from the true state every control.

**Reading.** Following one continuation with consistent supervision, the per-state pins and the online demonstration backup move the frontier by one control per stage, the pace the recorded-trajectory chain had, and the chaos of the demonstrations explains why: no neighbourhood of the recorded controls succeeds open-loop, so the value of the demonstration path cannot be represented next to the policy's near-identical failures, and the actor has nothing to climb. A policy that completes the turn from the cold reset must be a feedback controller; the supervision for it is the MPC's correction at states near a successful path, which this family's data does not contain (`mpc_spine_tube_0001`, `mpc_spine_dagger_0001` collect it). No cold-reset trial succeeded; nothing here is primitive-eligible.

This ablation reproduces `0084` cell for cell at 30 trials; the online demonstration backup is therefore neither the cause of nor the remedy for the stalled frontier.

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
