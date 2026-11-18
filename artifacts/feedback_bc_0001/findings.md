# Findings: supervised feedback learner and the learnability of the tube's corrections

## Learner runs (`train.py`, frozen splits in `runs/<tag>/manifest.json`)

| run | data | train fit (distinct rows, L2) | held-out regression (L2 median / signal) | spine det. prefixes | informative handoffs (j <= 9) train / held-out | cold |
| --- | --- | --- | --- | --- | --- | --- |
| v2_fo_h256 (256x256) | coarse tube (floor .1), 121 / 41 branches | 0.015 | 1.34 / ~0.5 | k9-13 | 1/20 / 1/19 | 0 |
| v2_fo_h1024x3 | same | 0.0001 | 0.80 / ~0.5 | k3-5, k7-13 | 7/20 / 1/19 | 0 |
| dagger_r1_h1024x3 | coarse tube + 2nd order + DAgger round 1, 198 / 57 | 0.015 (spine row 2 at 0.93) | 0.92 / ~0.5 | k9-13 | 1/22 / 1/20 | 0 |
| fine_partial_h1024x3 | fine tube (floor .005), 94 / 17 | 0.005 | 0.096 / 0.03 | k9-10, 12-13 | 2/16 / 0/11 | 0 |

A larger actor memorises the rows and completes more training handoffs (7/20 against 1/20) but generalises to held-out directions no better than three times the label magnitude at either planner scale; DAgger rows on the coarse data made the fit worse rather than better; no run reaches the cold reset.

## Local linear law (`linear_law.py`, fine tube, direction 5 held out, 196 branches)

Per spine time index t, ridge least squares of the correction (branch action minus spine action) on the state deviation (branch observation minus spine observation, 60 dims) over every correction row at t (15-154 rows): the held-out residual is 0.93-1.0 times the correction magnitude at every t for the best ridge, i.e. no better than predicting no correction, while small ridges overfit in-sample (residual 0.004-0.04). Closed loop with the law fitted on all directions: the 14 spine prefixes pass (in-sample states), fresh-direction basins fail at every control up to 9 (0-2 of 3 per cell; 3/3 only from control 10, the open-loop tolerance region), and held-out branches hand off successfully in 0 of 23 cells at j <= 9 (8 of 9 at j >= 10).

## Why: the labels are sampling noise (`mpc_spine_tube_0001/noise_probe/`)

Planning once from the exact spine state s_t, where the required correction is zero, the fine tracking planner (floor .005) returns controls 0.048-0.068 from the spine control over six seeds; from an r = .01 perturbed state the seed-to-seed spread of the returned control (0.031-0.038) is as large as the mean correction (0.014-0.036; signal-to-noise 0.4-0.95). The tube's corrections are therefore CEM sampling noise around the reference, not a function of the state: the MPC completes the turn from perturbed states by re-planning every control and rejecting infeasible candidates, with a 0.5 s horizon that cannot see a divergence surfacing 9-10 controls later, so it has no gradient toward an early correction. No learner can extract a feedback law from such labels, whatever its size or the number of branches; the earlier coarse corrections (0.2-0.7 at floor .1) were the same noise at a larger scale. Nothing here is primitive-eligible; no cold-reset trial succeeded.

## Seed-averaged long-horizon labels (`tracking_lh`, 42 branches, 1.0 s horizon, 4 seeds averaged on the first 3 corrections; gate run `lh_partial42`, direction 5 held out)

| measure | result |
| --- | --- |
| training fit (323 distinct rows) | L2 0.0005 (memorised) |
| held-out regression (42 rows) | L2 median 0.091, p90 0.33 (correction magnitude 0.02-0.03) |
| spine prefixes, deterministic | pass from k7 (previous learners: from k9) |
| spine prefixes, stochastic (3 each) | k4 1, k6 1, k8 2, k9-13 3 |
| corrected-state handoffs, training, j <= 9 | 6/24 (j6 1/7, j8 4/8, j9 1/1) |
| corrected-state handoffs, held-out, j <= 9 | 2/5 (j6 1/2, j8 1/1) |
| cold reset | fail (17 controls, rotated the wrong way to -4.2 rad) |
| per-time linear law, held-out residual | 0.95-1.7 x the correction magnitude at every t (no generalisation); 0/5 held-out handoffs |

Averaging seeds helped the closed loop modestly (the first held-out handoffs before control 9 in this family, and deterministic passes from spine prefix 7) but the held-out regression error is still three to four times the label magnitude. The noise probes explain the ceiling: at an early perturbed state the required correction within the horizon is zero, so an averaged label there is still noise around the reference, and the real corrections occur later, where the per-state sample count is small. The deterministic labeller (tracking penalty, `tracking_tp`) removes the noise component; its gate is the next measurement.

## Deterministic labels (`tracking_tp`, 41 branches, 1.5 s horizon, action-tracking penalty 10,000; gate run `tp_partial41`, direction 5 held out)

| measure | result |
| --- | --- |
| training fit (287 distinct rows) | L2 0.0026 |
| held-out regression (40 rows) | L2 median 0.027, p90 0.12 (label magnitude 0.007-0.013) |
| spine prefixes, deterministic | pass from k9 |
| spine prefixes, stochastic (3 each) | k5 1, k8 2, k9-13 3 |
| corrected-state handoffs, training, j <= 9 | 3/24 (all at j8) |
| corrected-state handoffs, held-out, j <= 9 | 1/5 (j8) |
| cold reset | fail (6 controls) |
| per-time linear law | 0/5 held-out handoffs |

The labels are now as reproducible as the planner allows (seed spread 0.007-0.009 at control 6; more CEM iterations do not tighten it, `mpc_spine_tube_0001/noise_probe/penalty_iterations_probe.log`), yet the closed loop did not improve over the seed-averaged labels: the policy's off-sample error (0.027) exceeds the feasible slack the labels reveal (0.007-0.02 per step), and under the open-loop chaos any step outside the slack ends the turn. Random-direction branches do not reduce that error in the directions the policy actually errs; DAgger rounds with the deterministic labeller at the policy's own states (`mpc_spine_tube_0001/dagger_tp_round*`) and a width comparison (256x256, 512x512 on the same split) are the next measurements.

### Width comparison on the deterministic split (`tp41_h256x256`, `tp41_h512x512`, `tp_partial41` = 1024x3; same manifest)

| actor | training fit (L2) | held-out regression (L2 median / p90) | spine deterministic from | informative handoffs train / held-out |
| --- | --- | --- | --- | --- |
| 256x256 | 0.0066 | 0.036 / 0.19 | k9 | 1/24 / 0/5 |
| 512x512 | 0.0042 | 0.042 / 0.18 | k8 (not k9) | 0/24 / 1/5 |
| 1024x3 | 0.0026 | 0.027 / 0.12 | k9 | 3/24 / 1/5 |

The largest actor generalises best, as on the coarse data; width is not the lever that closes the gap between the off-sample error (0.027 at best) and the per-step slack (0.007-0.02). DAgger rounds at the policy's own states continue with 1024x3.

### DAgger round 1 with the deterministic labeller (`dagger_tp_round_r1_h1024x1024x1024`: 63 random + 24 DAgger + 8 later random branches = 95 training, 13 held out)

| measure | 41 random branches (`tp_partial41`) | 95 branches incl. 24 DAgger |
| --- | --- | --- |
| training fit | 0.0026 | 0.0021 |
| held-out regression (L2 median / p90) | 0.027 / 0.12 | 0.053 / 0.16 |
| spine deterministic from | k9 | k10 |
| informative handoffs train / held-out | 3/24 / 1/5 | 1/23 / 0/13 |
| cold reset | fail (6 controls) | fail (3 controls) |

Adding branches at the policy's own states made the held-out error worse and the closed loop no better, as the coarse-label DAgger round had. The density learner on 74 random branches (`tp74_h1024x3`) separates the effect of more data from the effect of DAgger data.

### Smoothness test: the law is locally consistent, the regressor is not (`scratchpad spine/smoothness.py`, 2026-10-06)

Held-out correction rows binned by the observation distance to their nearest training row (learner `tp_partial41`, 41 branches; `tp74_h1024x3`, 74 branches):

| nearest-neighbour distance | learner error (41 / 74) | nearest training row's label difference (41 / 74) | error of outputting the spine action (41 / 74) |
| --- | --- | --- | --- |
| closest quartile | 0.015 / 0.030 | 0.006 / 0.014 | 0.006 / 0.007 |
| second | 0.025 / 0.051 | 0.014 / 0.018 | 0.008 / 0.014 |
| third | 0.026 / 0.065 | 0.015 / 0.021 | 0.015 / 0.017 |
| farthest | 0.062 / 0.064 | 0.014 / 0.022 | 0.008 / 0.015 |
| first corrected states only | 0.011 / 0.033 | 0.005 / 0.018 | 0.005 / 0.009 |

In every bin the trained actor's error exceeds both the nearest training row's label difference and the trivial spine-action baseline by a factor of two to four, and it rises with more training data at the same distances. The labels therefore vary smoothly enough across neighbouring states (a one-nearest-neighbour predictor would already halve the error); the 1024x3 actor memorises the rows and oscillates between them. The earlier "rough law" reading was wrong: the obstacle is the regressor's interpolation, which standard regularisation (input-noise augmentation, weight decay, early stopping on the held-out error) and non-parametric regression address; both are being tested on the same split.

### Regularised learners and nearest-neighbour policies on the 41-branch split (2026-10-06)

| policy | held-out regression (L2 median / p90) | spine rows fit | spine deterministic from | informative handoffs train / held-out |
| --- | --- | --- | --- | --- |
| 1024x3, plain (`tp_partial41`) | 0.027 / 0.12 | 0.001-0.008 | k9 | 3/24 / 1/5 |
| + input noise 0.02 (`tp41_noise02`) | 0.018 / 0.064 | rows 0-3 broken (1.1-1.6) | k10 | 0/24 / 0/5 |
| + weight decay 1e-3 (`tp41_wd1e3`) | 0.032 / 0.14 | rows 0-3 broken (1.0-1.4) | k9 | 5/24 / 0/5 |
| + early stop on held-out (`tp41_early`, best at 20k steps) | 0.020 / 0.081 | 0.013-0.022 | k6, k9 | 4/24 / 0/5 |
| 1-nearest-neighbour (`knn_k1`) | 0.013 / 0.027 | exact | all (exact replay) | 30/30 (exact replay) / 0/5 |
| 5-nearest-neighbour, scaled (`knn_k5_scale`) | 0.013 / 0.028 | – | k9 | 9/30 / 0/5 |

The nearest-neighbour policy's passes from the cold reset, the spine prefixes and the training branches are exact replays (the query state is a training row at distance zero), not feedback; its held-out handoffs are 0/5 like every learner. Its held-out regression error (0.013) is the interpolation floor of this data: it is set by the label difference between neighbouring branches in a 16-dimensional perturbation space, not by the regressor, and the best regularised MLP (0.018-0.020) approaches it without reaching the per-step slack (0.005-0.01). Noise augmentation and weight decay also destroy the fit at the spine's first four rows, which lie 0.002-0.005 apart in observation space. A second split (direction 2 held out, 74 branches: `tp_hold2_h1024x3`) gives 0.074, so the held-out error also varies strongly with the held-out direction.

### Nearest-neighbour policy under fresh perturbations (`runs/knn_k1_basins.log`): the interpolation floor is not feedback

| control j perturbed by r, then the 1-NN policy continues | r = .003 | r = .01 | r = .03 |
| --- | --- | --- | --- |
| 0-2 | 0/3 each | 0/3 | 0/3 |
| 3 | 1/3 | 0/3 | 0/3 |
| 4-6 | 0/3 each | 0/3 | 0/3 |
| 7 | 1/3 | 1/3 | 1/3 |
| 8 | 0/3 | 0/3 | 2/3 |
| 9 | 1/3 | 1/3 | 0/3 |
| 10-12 | 3/3 each | 2-3/3 | 2-3/3 |

This is the open-loop tolerance pattern of the spine itself (`frozen_checkpoint_evaluation_0001/spine_probe/basin_193.log`: open-loop replay fails at controls 0-8 for r >= .003 and passes from control 10), reproduced by the policy that interpolates the deterministic corrections most faithfully (held-out error 0.013). The corrections sampled at 6-18 branches per control do not define a feedback law at the precision the dynamics demand (0.005-0.01 per step); a per-time linear law on the same labels generalises no better than 0.73-1.0 of the signal at any t.

## Conclusion of route (a)

Second-order branches, a larger actor, deterministic long-horizon labels with a tracking penalty, DAgger at the policy's own states, regularisation and non-parametric regression were all tried on one demonstration. The labels became reproducible (seed spread 0.005-0.009) and locally consistent, the regressors reached the data's interpolation floor (0.013-0.020 held-out), and no policy gained competence before spine control 9 on held-out directions or from the cold reset: the per-step slack of the dynamics (0.005-0.01) is below the floor set by sampling a 16-dimensional perturbation space with O(100) branches, and density cannot be bought in that dimension. The MPC succeeds through re-planning with lookahead, not through a reproducible local law. Nothing in this artifact is primitive-eligible; no cold-reset trial of a learned policy succeeded (the nearest-neighbour policy's cold pass is a bit-exact replay of the spine from a distance-zero query, not feedback).
