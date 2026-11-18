# Variant 82 `ms_shared_prefix_from81`: auto-generated findings

**Settings differing from SAC0004:** `{"prefix_set": [8], "demonstration_include_mpc": false, "actor_bc_lambda": 100.0, "actor_bc_rows": "per_item", "online_policy_interactions": 4000, "checkpoints": [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], "initial_from": ["bootstrap_variant_0081_ms_add_heldout_demos_from79", "online_4000"], "actor_lr_override": 0.0001, "eval_stochastic_trials": 3, "actor_bc_std_target": 0.01, "prefix_sources": ["mpc_cold_demonstrations_0001", 120], "suffix_set": [1, 2, 3, 4, 5, 6, 7, 8], "demonstration_suffix_rows": 40, "suffix_weights": [2, 2, 2, 2, 1, 1, 1, 1], "holdout_source_names": ["seed_197"], "eval_source_names": ["seed_100", "seed_101"], "start_prefix_set": [0, 1, 2, 3], "start_prefix_weights": [1, 2, 1, 1], "start_prefix_share": 0.5, "actor_bc_last_rows": 8}`. Inherited checkpoint: `bootstrap_variant_0081_ms_add_heldout_demos_from79/checkpoints/online_4000`. Hypothesis: Longer-suffix stage with shared cold-start prefixes. The demonstrations form a tree from the common cold reset: 34 of the 63 training sources share a bit-identical first control (35 of the 64 in the collection; the 35th is the held-out seed_197), 8 of those the first two and 4 the first three. Half of the episodes hand off at those shared states (k = 0 cold reset, 1, 2, 3 controls from the reset, weights 1:2:1:1) and the policy completes the whole remaining turn (8-32 controls), so one handoff state carries many verified continuations instead of one; the other half keep the weighted end-suffix episodes (lengths 1-8). Whole demonstrations enter the demonstration buffer (entropy-free critic targets along every continuation during critic pretraining; the online batches carry the same demonstration rows under the standard SAC backup, as in every stage of this family). The actor BC term covers the last 8 rows of each demonstration plus its shared prefix rows, whose labels agree across the group; a tail row that lies at a state another source also visits with a different control (the two 8-control and one 9-control demonstrations) is never a BC target, so no supervised state carries conflicting labels. Continues 0081 online_4000 (63 training sources incl. 194-196; seed_197 held out).

Stage status `complete_sac_stage` (exit=0 (inferred from summary.json and all eight checkpoints being written; the launcher did not capture the exit status), 1194 s loop); final learner counters {'bc_updates': 12000, 'pretrain_updates': 6000, 'online_updates': 31750}; 500 critic-pretraining and 4000 online updates as prescribed. Temperature: learned from the inherited value. Actor step: overridden to 0.0001 for this stage (inherited 0.0001; ramp fraction unchanged). Supervision: lambda * MSE(actor mean(obs), atanh(clip(action, +-(1 - .02)))) over the last 8 rows of each demonstration episode plus its shared cold-start prefix rows, added to the single SAC actor step; spread anchor: lambda_std * MSE(actor log-std at the supervised rows, log(0.01)) added to the same actor step. Handoff noise: off.

## Actual execution

- Warm-up: 256 frozen-actor episodes by prefix {'s1': 20, 's2': 23, 's3': 22, 's4': 18, 's5': 11, 's6': 11, 's7': 11, 's8': 11, 'k0': 23, 'k1': 54, 'k2': 25, 'k3': 27}, force-qualified successes {'s1': 20, 's2': 20, 's3': 12, 's4': 4, 's5': 1, 's6': 0, 's7': 0, 's8': 0, 'k0': 0, 'k1': 0, 'k2': 0, 'k3': 0} (informational; the gate is the 1081-row demonstration buffer).
- Online: 4000 interactions; completed episodes by prefix {'s1': 118, 's2': 106, 's3': 116, 's4': 114, 's5': 42, 's6': 49, 's7': 49, 's8': 66, 'k0': 143, 'k1': 221, 'k2': 116, 'k3': 136}, successes {'s1': 111, 's2': 88, 's3': 68, 's4': 34, 's5': 12, 's6': 2, 's7': 1, 's8': 2, 'k0': 0, 'k1': 0, 'k2': 0, 'k3': 0}; outcomes {'drop': 412, 'palm_support': 525, 'success': 318, 'truncated': 21}; handoffs terminated before the policy 0; unfinished last suffix True. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: 11619; none entered replay.
- Actor BC loss first/last: 0.02175 / 0.0005464; spread-anchor loss first/last: 8.807e-06 / 5.578e-06; alpha raw-equivalent last 1.2746.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): before: L2 1.7032, std 0.0100, bias -2.73/-2.41; after_pretraining: L2 1.7032, std 0.0100, bias -2.56/-2.60; online_500: L2 1.3630, std 0.0100, bias -3.21/-3.23; online_1000: L2 1.3669, std 0.0100, bias -3.31/-3.34; online_1500: L2 1.3629, std 0.0100, bias -3.34/-3.30; online_2000: L2 1.3979, std 0.0100, bias -3.46/-3.50; online_2500: L2 1.3720, std 0.0100, bias -3.38/-3.37; online_3000: L2 1.3673, std 0.0100, bias -3.44/-3.49; online_3500: L2 1.3381, std 0.0100, bias -3.58/-3.59; online_4000: L2 1.3397, std 0.0100, bias -3.64/-3.65.

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

| Checkpoint | Cold | Shared k1 | Shared k2 | Shared k3 | seed_100 s1..s8 | seed_101 s1..s8 | seed_197 s1..s8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pretraining_baseline | F0/5 | F0/3 | F0/3 | F0/3 | P3/3 P3/3 F1/3 F1/3 F1/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F1/3 F1/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_500 | F0/5 | F0/3 | F0/3 | F0/3 | P3/3 P3/3 P3/3 F1/3 F1/3 F0/3 F0/3 F0/3 | P3/3 P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_1000 | F0/5 | F0/3 | F0/3 | F0/3 | P3/3 P3/3 P3/3 P1/3 F0/3 F0/3 F0/3 F0/3 | P3/3 P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_1500 | F0/5 | F0/3 | F0/3 | F0/3 | P3/3 P3/3 P3/3 F1/3 F2/3 P0/3 F1/3 F0/3 | P3/3 P3/3 F0/3 F1/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_2000 | F0/5 | F0/3 | F0/3 | F0/3 | P3/3 P3/3 P3/3 P1/3 F0/3 F2/3 F1/3 F0/3 | P3/3 P2/3 P0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_2500 | F0/5 | F0/3 | F0/3 | F0/3 | P3/3 P3/3 P3/3 P1/3 F3/3 F0/3 F0/3 F1/3 | P3/3 P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_3000 | F0/5 | F0/3 | F0/3 | F0/3 | P3/3 P3/3 F3/3 F3/3 F2/3 P0/3 P0/3 F0/3 | P3/3 F3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_3500 | F0/5 | F0/3 | F0/3 | F0/3 | P3/3 P3/3 P1/3 F3/3 F2/3 F0/3 F0/3 F1/3 | P3/3 P3/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |
| online_4000 | F0/5 | F0/3 | F0/3 | F0/3 | P3/3 P3/3 F2/3 F1/3 F2/3 F0/3 F0/3 F0/3 | P3/3 P3/3 P1/3 F0/3 F0/3 F0/3 F0/3 F0/3 | F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 F0/3 |

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

Online training episodes from the shared states: 616 (k0 143, k1 221, k2 116, k3 136), 0 force-qualified successes; end-suffix episodes 660 with successes 111/118, 88/106, 68/116, 34/114, 12/42, 2/49, 1/49, 2/66 for lengths 1–8, so the known competence is retained while the policy learns nothing from the shared states. The best shared-state episode of the stage reached 5.61 rad of force-qualified rotation (one k1 episode in the last third of training; a turn is 6.28 rad).

**Diagnostics.**

- The actor adopted the shared first control: its deterministic L2 distance to that control at the cold reset fell from 1.600 (inherited) to 0.12–0.20 at every online checkpoint (latent std .010), i.e. the shared-row BC term works as designed and the labels there are consistent (0 conflicting BC states among 504; 46 shared-prefix rows).
- It still fails two controls later. Over the three thirds of training the cold-start episodes last a median of 2 policy controls (maximum 5–6), ending by palm support in 14/43, 38/53 and 46/54 of them; k1–k3 episodes likewise last a median of 2 controls. The median final force-qualified angle is negative in every third (−2.26, −0.66, −0.71 rad at k0; −1.2 to −1.3 rad at k1): the policy rotates the object the wrong way before losing it.
- At the shared state after one control the online_4000 actor lies 0.204 from the nearest of the 34 recorded second controls and 1.43 from their atanh-space mean; the 34 continuations are mutually incompatible (pairwise L2 median 2.40). The actor therefore selects one continuation rather than averaging them, but misses it by about four times the width of the success basin around a recorded control (.02–.05), and the next control has no consistent label at all.
- Held-out seed_197 lost its last-control competence (P3 at baseline, F0 from online_1000 on) in this stage as in the ablation.

**Reading.** Shared handoff states concentrate many verified continuations on one physical state, and consistent BC at those states does pin the first control, but completion from them did not emerge in 4,000 updates: the continuation after the shared prefix is multimodal with basins far narrower than what the SAC gradient and the 547-row BC average deliver, and the wrong-way rotation that follows yields negative shaping returns, from which the critic learns that ending the episode early is preferable (see the ablation `0083`, where without the shared-row BC the actor drifts to dropping at the first control). Whole-demonstration critic data does not change this because Q(s, a) is fitted at the demonstrations' exact actions, which the policy never emits. Hypothesis refuted in this form; no cold-reset trial succeeded; nothing here is primitive-eligible. The retained end-suffix cells (seeds 100, 101) are at the inherited level; a 30-trial fixed-set comparison is recorded under `frozen_checkpoint_evaluation_0001/results_multisource/` when available.

### Training-set comparison (fixed eight demonstrations, 30 trials per cell)

Seeds 100, 101, 110, 120, 136, 139, 157 and 158 (`frozen_checkpoint_evaluation_0001/results_multisource/bootstrap_variant_0082_ms_shared_prefix_from81/online_4000__seed_100_..._seed_158.json`):

| checkpoint | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `0079` online_4000 | 240/240 | 182/240 | 101/240 | 59/240 | 69/240 | 39/240 | 19/240 | 15/240 |
| `0081` online_4000 (inherited) | 219/240 det 7/8 | 168/240 det 6/8 | 113/240 det 4/8 | 70/240 det 2/8 | 53/240 det 1/8 | 21/240 det 1/8 | 14/240 det 0/8 | 7/240 det 0/8 |
| `0082` online_4000 | 238/240 det 8/8 | 199/240 det 6/8 | 111/240 det 5/8 | 69/240 det 5/8 | 46/240 det 2/8 | 27/240 det 0/8 | 15/240 det 0/8 | 7/240 det 1/8 |
| `0083` online_4000 | 214/240 det 7/8 | 168/240 det 6/8 | 126/240 det 6/8 | 67/240 det 3/8 | 65/240 det 3/8 | 26/240 det 1/8 | 23/240 det 0/8 | 7/240 det 0/8 |

The shared-prefix stage did not cost end-suffix competence on the training set: it is above its inherited checkpoint at lengths 1–3 and 7 (238, 199, 111 against 219, 168, 113) and level elsewhere, and matches `0079` at length 1 while exceeding it at lengths 2, 3 and 4. The long-suffix episodes and the whole-trajectory buffer therefore act as a mild regulariser for the short suffixes; they bring nothing at the shared states.

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
