# Qualified policy-only first turn; continued policy drops

The final numerically refined actor completed a strict policy-only full turn in **9 controls / 0.9 seconds**, reaching raw angle **6.823360886 rad** and independently force-qualified angle **6.651565369 rad**, without palm or external support. This is a successful first-turn task completion from the tested fixed reset. When allowed to continue selecting its own commands, the same actor **dropped at control 12 / 1.2 seconds**. It did not pass the separate two-second continuation requirement. These are distinct outcomes; this is not a successful SAC policy, a production primitive, or evidence of robustness across resets.

## Dataset and physical provenance

`supervised_pairs.npz` contains exactly 12 actual observation/action pairs: original bounded teacher rows 1–6, nominal common-continuation trajectory rows 7–9, and recovery trajectory rows 7–9. The last two groups have bitwise identical action labels at their respective actual observations. Original teacher future rows 7–22 and all six preceding actor actions are excluded from supervision. The existing epsilon 0.02 float32 Torch clamp changes zero scalars. No fabricated next actions or replay episodes were created.

Both source branches completed qualified turns in nine controls and survived two-second fixed-target holds. Before training, all four source traces were independently re-executed with the archived current numerical code: nominal strict, nominal hold, recovery strict, and recovery hold. Available transition fields and every 2 ms physics sample matched bitwise. Exact input traces, commands, reports, source frames including initial joint state, and source manifests are preserved under `inputs/`; fresh replay proofs are in `verified_*`. See `dataset_validation.json` for all comparisons.

## Prescribed BC and one head correction

A fresh SACAgent was initialized with seed 17001; no previous actor or optimizer was loaded. Training performed exactly 12,000 mean-only BC Adam updates at learning rate 0.003, batch size 256, epsilon 0.02, with uniform replacement sampling of the 12 pairs. Architecture is 60/16 with 256-by-256 hidden layers. The two unused diagnostic critics do not affect the BC loss; the paper uses ten critics. CUDA, one Torch/inter-op CPU thread, and disabled TF32 were used. The paper settings source, exact sampled batches, learner configuration, frozen current source, runtime/package versions, and all checkpoints are archived. Critics, targets, and temperature remained bitwise unchanged. There were zero critic-pretraining or online SAC updates.

| BC updates | Exact singleton action MSE | Maximum action fit error | Strict actual policy result | Peak force-qualified angle |
|---:|---:|---:|---|---:|
| 3,000 | 0.0103872 | 0.757547 | Drop at 0.9 s | 1.920238 rad |
| 6,000 | 0.00942368 | 0.714867 | Palm support at 0.2 s | 0.504550 rad |
| 12,000 | 5.49862e-7 | 0.00543860 | Drop at 1.0 s | 3.895180 rad |

Because final maximum fit error exceeded 1e-6, the one authorized frozen-feature head residual solve was executed. The 12-by-257 design had full row rank 12 and condition number 18,595.30. One float64 minimum-norm correction to the existing float32 atanh targets was cast to float32 and added once. Its Frobenius norm was 1.973973, maximum coefficient 0.382264, and head weight norm increased from 4.04315 to 4.50468. The ideal float64 residual MSE was 2.358e-32. Actual singleton Torch action MSE became **5.698e-15**, with maximum error **4.768e-7**. Exactly the mean-head weight and bias changed; every other checkpoint payload value remained bitwise unchanged from the preserved final-BC baseline. No optimizer sweep followed.

## Strict success and separate continuation failure

`head_polish/independent_audit/` records the nine actual learned commands, observations, next observations, rewards, terminal flags, qpos/qvel, targets, initial states, and full physics. The actor first achieved per-physics-step qualification at 0.858 s and remained qualified at the 0.9 s control boundary. The strict success checkpoint is `head_polish/polished.pt`; its metadata is `head_polish/polished.pt.metadata.json`.

`head_polish/independent_audit_continuous_policy_2s/` replays the strict prefix bitwise and then continues the actual deterministic actor, with success termination disabled and all failure checks retained. The actor dropped at 1.2 s, only 0.342 s after first physics qualification / 0.3 s after the successful control boundary. Final force-qualified angle fell to 1.893146467 rad. No fixed command was substituted for policy continuation. The first post-success command differed from the successful teacher fixed-hold command by 1.85957 in maximum normalized action component. Post-success observations have no supervised continuation labels in this 12-pair dataset; this limits the claim to the measured first-turn success.

The common corrections remove the previously conflicting late labels. They do not eliminate every steep mapping in the dataset: the inherited first two teacher observations differ by only 0.00541487 in L2 distance while their actions differ by 1.77027. This is a possible conditioning issue, not a proven sole cause of any observed failure. `trajectory_analysis.json` reports exact pair distances and actual closed-loop error growth.

## Checkpoint state and scope

Final checkpoint SHA-256: `bc00ed2358e2d50e33d9634da22e8bb350bda149b56644c4dfad268bfbddd77b`.

Metadata SHA-256: `99d63bc4931501dba277364d898a43f18b2fbf4cc2f0066a2f1f57412c2e20eb`.

The true counters are **bc_updates=12000, pretrain_updates=0, online_updates=0**. `std_initialized` is true. All 16 reset log standard deviations equal **-1.2039728164672852**, corresponding to standard deviation approximately 0.3. The stage is `diagnostic-common-continuation-head-refinement`; critics remain at random initialization. `checkpoint_state.json` preserves the full learner configuration and reset actor outputs. Any future SAC initialization requires an explicit import preserving these truthful counters and state; this artifact itself is not a main-stage checkpoint.

The bounded protocol is complete: three BC checkpoint evaluations, one permitted head solve/evaluation, and one conditional continued-policy evaluation, plus the four source-verification replays. No further numerical experiment was run. `manifest.json` freezes the complete artifact; `inputs_manifest.json` verifies all archived inputs remained unchanged throughout.
