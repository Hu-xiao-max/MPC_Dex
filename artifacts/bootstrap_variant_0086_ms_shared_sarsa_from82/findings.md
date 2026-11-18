# Variant 86 `ms_shared_sarsa_from82`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_set": [8], "demonstration_include_mpc": false, "actor_bc_lambda": 100.0, "actor_bc_rows": "per_item", "online_policy_interactions": 4000, "checkpoints": [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], "initial_from": ["bootstrap_variant_0082_ms_shared_prefix_from81", "online_4000"], "actor_lr_override": 0.0001, "eval_stochastic_trials": 3, "actor_bc_std_target": 0.01, "prefix_sources": ["mpc_cold_demonstrations_0001", 120], "suffix_set": [1, 2, 3, 4, 5, 6, 7, 8], "demonstration_suffix_rows": 40, "suffix_weights": [2, 2, 2, 2, 1, 1, 1, 1], "holdout_source_names": ["seed_197"], "eval_source_names": ["seed_100"], "start_prefix_set": [0, 1, 2, 3], "start_prefix_weights": [1, 2, 1, 1], "start_prefix_share": 0.5, "actor_bc_last_rows": 8, "actor_bc_shared_lambda": 50.0, "online_demo_sarsa": true}`. Inherited checkpoint: `bootstrap_variant_0082_ms_shared_prefix_from81/checkpoints/online_4000`. Hypothesis: Ablation of index 84 without the spine (the 0082 shared-state stream, k = 0..3 over the shared groups) but with the online demonstration backup and the stronger shared-row pins: isolates the contribution of following one continuation from the two learning-signal repairs.

Stage status `complete_sac_stage` (exit=0 (inferred from summary.json and all eight checkpoints being written; the launcher did not capture the exit status), 1683 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 6500, 'online_updates': 35750}; 500 critic-pretraining and 4000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: overridden to 0.0001 for this stage (inherited 0.0001; ramp fraction unchanged). Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over the last 8 rows of each demonstration episode plus its shared cold-start prefix rows; additionally, per distinct shared state, lambda_shared 50.0 * (MSE of the actor mean to the label + MSE of the log-std to log(std target)) summed over the distinct shared states (one term per state, however many sources sit on it), added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.01)) added to the same actor step. Handoff noise: off.

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'s1': 20, 's2': 23, 's3': 22, 's4': 18, 's5': 11, 's6': 11, 's7': 11, 's8': 11, 'k0': 23, 'k1': 54, 'k2': 25, 'k3': 27}, force-qualified successes {'s1': 20, 's2': 19, 's3': 7, 's4': 3, 's5': 0, 's6': 1, 's7': 1, 's8': 2, 'k0': 0, 'k1': 0, 'k2': 0, 'k3': 0} (informational; the gate is the 1081-row demonstration buffer).
- Online: 4000 interactions; completed episodes by prefix {'s1': 118, 's2': 108, 's3': 117, 's4': 116, 's5': 42, 's6': 50, 's7': 50, 's8': 68, 'k0': 143, 'k1': 223, 'k2': 119, 'k3': 137}, successes {'s1': 111, 's2': 79, 's3': 59, 's4': 35, 's5': 4, 's6': 4, 's7': 1, 's8': 2, 'k0': 0, 'k1': 0, 'k2': 0, 'k3': 0}; outcomes {'palm_support': 674, 'drop': 301, 'success': 295, 'truncated': 21}; handoffs terminated before the policy 0; unfinished last suffix True. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 11749; none entered replay.
- Actor BC loss first/last: 0.000566 / 0.0005081; spread-anchor loss first/last: 6.556e-06 / 6.83e-06; alpha raw-equivalent last 1.8964.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 1.3397, std 0.0100, bias -3.64/-3.65; after_pretraining: L2 1.3397, std 0.0100, bias -3.09/-3.13; online_500: L2 1.2936, std 0.0100, bias -3.49/-3.53; online_1000: L2 1.3172, std 0.0100, bias -3.57/-3.62; online_1500: L2 1.2711, std 0.0100, bias -3.72/-3.77; online_2000: L2 1.3166, std 0.0100, bias -3.82/-3.82; online_2500: L2 1.2999, std 0.0100, bias -4.14/-4.14; online_3000: L2 1.3018, std 0.0100, bias -4.43/-4.40; online_3500: L2 1.2894, std 0.0100, bias -4.73/-4.73; online_4000: L2 1.2945, std 0.0100, bias -4.72/-4.88.

## Frozen evaluations (in-run; 3 stochastic seeds per prefix)

| Checkpoint | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch | Cold det / stoch |
| --- | --- | --- | --- | --- |
| pretraining_baseline | fail / 0/3 | fail / 0/3 | pass / 3/3 | fail / 0/5 |
| online_500 | fail / 0/3 | fail / 0/3 | pass / 3/3 | fail / 0/5 |
| online_1000 | fail / 0/3 | fail / 0/3 | pass / 3/3 | fail / 0/5 |
| online_1500 | fail / 0/3 | fail / 0/3 | pass / 3/3 | fail / 0/5 |
| online_2000 | fail / 0/3 | fail / 0/3 | pass / 3/3 | fail / 0/5 |
| online_2500 | fail / 0/3 | fail / 0/3 | pass / 3/3 | fail / 0/5 |
| online_3000 | fail / 0/3 | fail / 0/3 | fail / 3/3 | fail / 0/5 |
| online_3500 | fail / 0/3 | fail / 0/3 | fail / 0/3 | fail / 0/5 |
| online_4000 | fail / 0/3 | fail / 0/3 | fail / 2/3 | fail / 0/5 |

## Multi-source evaluation (in-run; a cell is the deterministic trial P/F followed by stochastic successes/trials; per source the cells run over the suffix lengths; shared k = the state after k shared cold-start controls)

| Checkpoint | Cold | Shared k1 | Shared k2 | Shared k3 | seed_100 s1..s8 | seed_197 s1..s8 |
| --- | --- | --- | --- | --- | --- | --- |
| pretraining_baseline | F0/5 | F0/3 | F0/3 | F0/3 | P3/3 P3/3 F2/3 F2/3 F1/3 F1/3 F0/3 F0/3 | F0/3 F0/3 F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_500 | F0/5 | F0/3 | F0/3 | F0/3 | P3/3 F0/3 F0/3 F1/3 F1/3 F0/3 F0/3 F0/3 | F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_1000 | F0/5 | F0/3 | F0/3 | F0/3 | P3/3 F0/3 P2/3 F1/3 F1/3 P2/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_1500 | F0/5 | F0/3 | F0/3 | F0/3 | P3/3 P3/3 F1/3 F1/3 P2/3 F2/3 F0/3 P0/3 | F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_2000 | F0/5 | F0/3 | F0/3 | F0/3 | P3/3 P3/3 P1/3 P1/3 P1/3 F0/3 F0/3 F0/3 | P2/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_2500 | F0/5 | F0/3 | F0/3 | F0/3 | P3/3 P3/3 P1/3 P1/3 F1/3 F0/3 F1/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_3000 | F0/5 | F0/3 | F0/3 | F0/3 | P3/3 P3/3 F1/3 F0/3 F2/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 P0/3 F0/3 F0/3 F0/3 |
| online_3500 | F0/5 | F0/3 | F0/3 | F0/3 | P3/3 P3/3 P2/3 F1/3 F1/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_4000 | F0/5 | F0/3 | F0/3 | F0/3 | P3/3 P0/3 F3/3 F1/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

No large-sample result for this stage at the time of writing.

## Spine states, cold reset and the learning-signal repairs: what was measured

The spine is seed_193 (14 controls). Half of the episodes handed off at its prefixes k = 0..13 (k = 0 the cold reset; k = 1-3 states shared with 33, 7 and 3 other sources), the other half were weighted end-suffix episodes over all 63 sources. 30-trial frozen evaluation of online_4000 at the cold reset and at every spine prefix (`frozen_checkpoint_evaluation_0001/results_shared/bootstrap_variant_0086_ms_shared_sarsa_from82/online_4000.json`; 1 deterministic + 30 stochastic fresh-seed trials per cell, `D` = deterministic pass):

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

Without the spine (the `0082` shared-state stream with the repaired learning signals) the result is the `0082` result: 0/30 at the cold reset and at the three shared states, 0 successes in 622 shared-state training episodes; the training source seed_100 keeps lengths 1-2 and the held-out seed_197 its last control (P3 at online_4000).

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
