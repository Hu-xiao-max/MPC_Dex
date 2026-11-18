# Variant 83 `ms_shared_prefix_nosharedbc_from81`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_set": [8], "demonstration_include_mpc": false, "actor_bc_lambda": 100.0, "actor_bc_rows": "per_item", "online_policy_interactions": 4000, "checkpoints": [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], "initial_from": ["bootstrap_variant_0081_ms_add_heldout_demos_from79", "online_4000"], "actor_lr_override": 0.0001, "eval_stochastic_trials": 3, "actor_bc_std_target": 0.01, "prefix_sources": ["mpc_cold_demonstrations_0001", 120], "suffix_set": [1, 2, 3, 4, 5, 6, 7, 8], "demonstration_suffix_rows": 40, "suffix_weights": [2, 2, 2, 2, 1, 1, 1, 1], "holdout_source_names": ["seed_197"], "eval_source_names": ["seed_100", "seed_101"], "start_prefix_set": [0, 1, 2, 3], "start_prefix_weights": [1, 2, 1, 1], "start_prefix_share": 0.5, "actor_bc_last_rows": 8, "actor_bc_shared_prefix_rows": false}`. Inherited checkpoint: `bootstrap_variant_0081_ms_add_heldout_demos_from79/checkpoints/online_4000`. Hypothesis: Ablation of the shared-prefix stage (index 82): identical episode stream and demonstration buffer, but the actor BC term excludes the shared prefix rows (BC on the last 8 rows only, with the same exclusion of tail rows at shared states), so the BC term never touches the cold reset or the shared states; any cold-start or shared-state gain in 82 that is absent here is attributable to supervision toward the shared first controls rather than to the long-suffix episodes and the whole-trajectory critic data.

Stage status `complete_sac_stage` (exit=0 (inferred from summary.json and all eight checkpoints being written; the launcher did not capture the exit status), 1231 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 6000, 'online_updates': 31750}; 500 critic-pretraining and 4000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: overridden to 0.0001 for this stage (inherited 0.0001; ramp fraction unchanged). Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over the last 8 rows of each demonstration episode plus its shared cold-start prefix rows, added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.01)) added to the same actor step. Handoff noise: off.

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'s1': 20, 's2': 23, 's3': 22, 's4': 18, 's5': 11, 's6': 11, 's7': 11, 's8': 11, 'k0': 23, 'k1': 54, 'k2': 25, 'k3': 27}, force-qualified successes {'s1': 20, 's2': 20, 's3': 12, 's4': 4, 's5': 1, 's6': 0, 's7': 0, 's8': 0, 'k0': 0, 'k1': 0, 'k2': 0, 'k3': 0} (informational; the gate is the 1081-row demonstration buffer).
- Online: 4000 interactions; completed episodes by prefix {'s1': 127, 's2': 115, 's3': 124, 's4': 118, 's5': 44, 's6': 53, 's7': 51, 's8': 70, 'k0': 147, 'k1': 241, 'k2': 130, 'k3': 141}, successes {'s1': 124, 's2': 94, 's3': 76, 's4': 39, 's5': 6, 's6': 4, 's7': 3, 's8': 5, 'k0': 0, 'k1': 0, 'k2': 0, 'k3': 0}; outcomes {'drop': 690, 'palm_support': 304, 'success': 351, 'truncated': 16}; handoffs terminated before the policy 0; unfinished last suffix False. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 12257; none entered replay.
- Actor BC loss first/last: 0.0006761 / 0.0003948; spread-anchor loss first/last: 9.571e-06 / 1.163e-05; alpha raw-equivalent last 1.2793.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 1.7032, std 0.0100, bias -2.73/-2.41; after_pretraining: L2 1.7032, std 0.0100, bias -2.56/-2.60; online_500: L2 1.7359, std 0.0105, bias -3.15/-3.12; online_1000: L2 2.8945, std 0.0110, bias -3.35/-3.36; online_1500: L2 3.0160, std 0.0111, bias -3.31/-3.32; online_2000: L2 2.7539, std 0.0116, bias -3.67/-3.66; online_2500: L2 2.5138, std 0.0123, bias -3.77/-3.72; online_3000: L2 2.7439, std 0.0128, bias -3.89/-3.89; online_3500: L2 2.6750, std 0.0134, bias -3.81/-3.86; online_4000: L2 2.6900, std 0.0142, bias -3.84/-3.89.

## Frozen evaluations (in-run; 3 stochastic seeds per prefix)

| Checkpoint | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch | Cold det / stoch |
| --- | --- | --- | --- | --- |
| pretraining_baseline | fail / 0/3 | fail / 0/3 | pass / 3/3 | fail / 0/5 |
| online_500 | fail / 0/3 | fail / 0/3 | pass / 3/3 | fail / 0/5 |
| online_1000 | fail / 0/3 | fail / 0/3 | pass / 3/3 | fail / 0/5 |
| online_1500 | fail / 0/3 | fail / 0/3 | fail / 3/3 | fail / 0/5 |
| online_2000 | fail / 0/3 | fail / 0/3 | pass / 2/3 | fail / 0/5 |
| online_2500 | fail / 0/3 | fail / 0/3 | pass / 3/3 | fail / 0/5 |
| online_3000 | fail / 0/3 | fail / 0/3 | pass / 1/3 | fail / 0/5 |
| online_3500 | fail / 0/3 | fail / 0/3 | fail / 0/3 | fail / 0/5 |
| online_4000 | fail / 0/3 | fail / 0/3 | fail / 3/3 | fail / 0/5 |

## Multi-source evaluation (in-run; a cell is the deterministic trial P/F followed by stochastic successes/trials; per source the cells run over the suffix lengths; shared k = the state after k shared cold-start controls)

| Checkpoint | Cold | Shared k1 | Shared k2 | Shared k3 | seed_100 s1..s8 | seed_101 s1..s8 | seed_197 s1..s8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pretraining_baseline | F0/5 | F0/3 | F0/3 | F0/3 | P3/3 P3/3 F1/3 F1/3 F0/3 F1/3 F0/3 F1/3 | P3/3 F0/3 F0/3 F1/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_500 | F0/5 | F0/3 | F0/3 | F0/3 | P3/3 P3/3 P3/3 P1/3 P1/3 F0/3 F0/3 F0/3 | P3/3 P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F2/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_1000 | F0/5 | F0/3 | F0/3 | F0/3 | P3/3 P3/3 P3/3 P2/3 P1/3 F0/3 F0/3 F0/3 | P3/3 P3/3 F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_1500 | F0/5 | F0/3 | F0/3 | F0/3 | P3/3 P3/3 P3/3 F2/3 P1/3 F0/3 F0/3 F0/3 | P3/3 P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_2000 | F0/5 | F0/3 | F0/3 | F0/3 | P3/3 P3/3 P3/3 P1/3 P2/3 F1/3 F1/3 F0/3 | P3/3 P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F1/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_2500 | F0/5 | F0/3 | F0/3 | F0/3 | P3/3 P3/3 P3/3 P1/3 F3/3 F1/3 F0/3 F0/3 | P3/3 P3/3 F0/3 F1/3 F0/3 F0/3 F0/3 F0/3 | F2/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_3000 | F0/5 | F0/3 | F0/3 | F0/3 | P3/3 P2/3 P3/3 P2/3 P2/3 F0/3 F0/3 F1/3 | P3/3 P3/3 P0/3 F0/3 F1/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_3500 | F0/5 | F0/3 | F0/3 | F0/3 | P3/3 P3/3 P3/3 P2/3 P1/3 F0/3 F0/3 F0/3 | P3/3 P3/3 F0/3 F1/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_4000 | F0/5 | F0/3 | F0/3 | F0/3 | P3/3 F1/3 P2/3 P1/3 F2/3 F1/3 F0/3 F0/3 | P3/3 P3/3 P0/3 F1/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |

## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

No large-sample result for this stage at the time of writing.

## Shared cold-start states: what was measured

Episode stream (both stages, identical seeds): half of the episodes hand off at the shared states (k = 0 cold reset, k = 1 the state after the first control shared by 34 training sources, k = 2 by 8, k = 3 by 4; weights 1:2:1:1) and the policy has to complete the whole remaining turn; the other half are the weighted end-suffix episodes (lengths 1–8). Whole demonstrations (1,081 rows, 63 episodes) form the demonstration buffer. Evaluations at every checkpoint: cold deterministic + 5 stochastic trials, shared k1–k3 deterministic + 3 stochastic, and the suffix cells of two training sources (seeds 100, 101) and the held-out seed_197 (table above). 30-trial frozen evaluations (`frozen_checkpoint_evaluation_0001/results_shared/<stage>/online_{2000,4000}.json`, fresh seeds, 1 deterministic + 30 stochastic per cell):

| checkpoint | cold (k0) | k1 (34 continuations) | k2 (8) | k3 (4) |
| --- | --- | --- | --- | --- |
| `0081` online_4000 (inherited) | 0/30 | – | – | – |
| `0082` online_2000 | 0/30 | 0/30 | 0/30 | 0/30 |
| `0082` online_4000 | 0/30 | 0/30 | 0/30 | 0/30 |
| `0083` online_2000 | 0/30 | 0/30 | 0/30 | 0/30 |
| `0083` online_4000 | 0/30 | 0/30 | 0/30 | 0/30 |

No deterministic trial passed anywhere.

Online training episodes from the shared states: 659 (k0 147, k1 241, k2 130, k3 141), 0 force-qualified successes; end-suffix successes 124/127, 94/115, 76/124, 39/118, 6/44, 4/53, 3/51, 5/70 for lengths 1–8.

**Diagnostics.** Without BC at the shared rows the actor moved away from the shared first control: its L2 distance at the cold reset rose from 1.600 to 2.6–3.0 from online_1000 on (latent std .010 → .014). Episodes from the shared states shortened over training: in the last third every k0, k1 and k2 episode ended by a drop after exactly one policy control (55/55, 86/86, 43/43) and k3 episodes after a median of one (47/47 drops); the median final force-qualified angle was negative throughout (−0.3 to −0.7 rad). At the shared state after one control the online_4000 actor is 2.94 from the nearest recorded second control. The in-run cells show the same collapse for the held-out seed_197 (last control P3 at baseline, F0 from online_1000) while seeds 100 and 101 keep their short suffixes.

**Reading.** This ablation isolates what the SAC objective does at the shared states when no supervision anchors them: the policy's own continuations rotate the object the wrong way, the shaping return is negative, and the critic comes to prefer the shortest episode, so the actor converges on dropping the object at once. The shared-row BC of `0082` prevents that collapse at the first control (L2 0.16 instead of 2.9) but not the failure two controls later, so neither the long-suffix episodes nor the whole-trajectory buffer supplies a usable gradient toward completion from the shared states. No cold-reset trial succeeded; nothing here is primitive-eligible.

### Training-set comparison (fixed eight demonstrations, 30 trials per cell)

Seeds 100, 101, 110, 120, 136, 139, 157 and 158 (`frozen_checkpoint_evaluation_0001/results_multisource/bootstrap_variant_0083_ms_shared_prefix_nosharedbc_from81/online_4000__seed_100_..._seed_158.json`):

| checkpoint | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `0079` online_4000 | 240/240 | 182/240 | 101/240 | 59/240 | 69/240 | 39/240 | 19/240 | 15/240 |
| `0081` online_4000 (inherited) | 219/240 det 7/8 | 168/240 det 6/8 | 113/240 det 4/8 | 70/240 det 2/8 | 53/240 det 1/8 | 21/240 det 1/8 | 14/240 det 0/8 | 7/240 det 0/8 |
| `0083` online_4000 | 214/240 det 7/8 | 168/240 det 6/8 | 126/240 det 6/8 | 67/240 det 3/8 | 65/240 det 3/8 | 26/240 det 1/8 | 23/240 det 0/8 | 7/240 det 0/8 |
| `0082` online_4000 | 238/240 det 8/8 | 199/240 det 6/8 | 111/240 det 5/8 | 69/240 det 5/8 | 46/240 det 2/8 | 27/240 det 0/8 | 15/240 det 0/8 | 7/240 det 1/8 |

The ablation keeps the training-set competence at the inherited level (214, 168, 126 against 219, 168, 113 at lengths 1–3) while its policy collapses at the shared states, so the collapse is confined to the states the BC term does not reach.

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
