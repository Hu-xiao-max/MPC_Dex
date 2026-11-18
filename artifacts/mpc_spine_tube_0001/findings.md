# Findings: tube of MPC corrections around the spine seed_193

Spine: `mpc_cold_demonstrations_0001/episodes/seed_193` (14 controls). Branches: one control j in 0..13 replaced by a_j + r u (u a unit vector, 4 directions per radius, r in {.01, .03, .1}), then the planner completes the turn; strict force-qualified success re-executed from a fresh cold reset and stored only on a bit-exact match.

## Collections

| planner | branches attempted | strict successes | stored | notes |
| --- | --- | --- | --- | --- |
| re-planning CEM (archived recipe; `episodes/`) | 72 (stopped early) | 64 | 64 | 48 of the 64 at controls 10-13; 7 budget exhaustions (25 corrections) at controls 0-6 |
| reference-tracking CEM (`tracking_cem.py`, floor scale .1; `tracking/episodes/`) | 168 | 166 | 166 | 2 failures (j1 r.03 d2, j2 r.01 d1: boundary reached without strict success); 12 branches at control 13 end the turn with the perturbed control itself and carry no correction rows |

## Label coherence (first correction after the perturbed control; L2 in normalized action space)

| j | n | perturbed-state obs L2 from the spine (median) | re-planning: first correction to spine a_{j+1} / between directions | tracking: first correction to spine a_{j+1} / between directions | tracking: whole segment to the time-aligned spine controls | tracking: controls to finish vs spine |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 12 | 0.70 | 0.03 / 1.58 (4 branches) | 0.23 / 0.65 | 0.70 | 10 vs 13 |
| 1 | 11 | 0.29 | – | 0.64 / 0.94 | 0.68 | 10 vs 12 |
| 2 | 11 | 0.68 | – | 0.65 / 0.82 | 0.70 | 11 vs 11 |
| 3 | 12 | 0.29 | – | 0.55 / 0.74 | 0.65 | 10 vs 10 |
| 4 | 12 | 0.27 | 2.83 / 1.56 (6) | 0.58 / 0.63 | 0.62 | 9 vs 9 |
| 5 | 12 | 0.20 | – | 0.16 / 0.42 | 0.54 | 8 vs 8 |
| 6 | 12 | 0.22 | – | 0.14 / 0.25 | 0.52 | 7 vs 7 |
| 7 | 12 | 0.17 | 3.46 / 3.10 (6) | 0.21 / 0.41 | 0.54 | 5 vs 6 |
| 8 | 12 | 0.11 | – | 0.29 / 0.41 | 0.57 | 4 vs 5 |
| 9 | 12 | 0.22 | – | 0.80 / 0.91 | 0.73 | 4 vs 4 |
| 10 | 12 | 0.24 | 3.35 / 1.93 (12) | 0.36 / 0.51 | 0.54 | 3 vs 3 |
| 11 | 12 | 0.19 | 2.21 / 1.28 (12) | 0.69 / 0.80 | 0.69 | 2 vs 2 |
| 12 | 12 | 0.15 | 3.07 / 1.46 (12) | 0.70 / 0.59 | 0.70 | 1 vs 1 |

The re-planning CEM returns a different full solution from every perturbed state (corrections 2.2-3.5 from the spine control at states 0.06-0.26 away); the tracking CEM returns corrections 0.14-0.80 from the spine control, 0.25-0.94 between directions, and finishes in the spine's own number of controls. Corrections of 0.2-0.7 for perturbations of 0.01-0.1 mean the feedback a policy must learn is strongly amplifying (consistent with the chaos of the open-loop sequences), but it is coherent across neighbouring states.

## Status

The tracking collection is the input of `bootstrap_variant_0087_ms_spine193_tube_from84` and its ablation `0088`. Every stored episode is a verified cold-reset trajectory; the perturbed control is never a learning target; nothing here is primitive-eligible. `coherence_test/` holds the 12-branch pilot that selected the tracking planner and is not used as a source.

## Search granularity of the tracking planner decides the gain of the law (2026-10-06)

Coherence test at controls 4 and 7 (radii .01/.03, 3 directions each; `coherence_test/` at floor scale .1, `coherence_fine/` at .02):

| floor scale | successes | first correction to spine a_{j+1} (median; j4 / j7) | between directions (j4 / j7) | whole segment to the time-aligned spine controls (j4 / j7) |
| --- | --- | --- | --- | --- |
| .1 (collections `tracking/`, `tracking2/`, `dagger_round*/`) | 12/12 | 0.58 / 0.21 | 0.63 / 0.41 | 0.62 / 0.54 |
| .02 | 12/12 | 0.10 / 0.10 | 0.14 / 0.11 | 0.13 / 0.12 |

At floor scale .1 the planner's sampling granularity (about 0.1 rad on the residual) is itself larger than the correction a perturbation of 0.01-0.03 needs, so every label is a 0.2-0.7 move away from the spine and the feedback law a learner has to represent has a gain of roughly 25 (0.5 of action per 0.02 of state); the first-order supervised learners memorised such labels but could not interpolate them (held-out regression error 0.8-1.3, spine rows mis-fitted once DAgger rows were added). At floor scale .02 the corrections are five times smaller with the same success rate, i.e. a law of gain about 3-10. The coarse collections are therefore superseded for the feedback-learning purpose; a finer floor (.005) is being tested and the tube will be recollected at the chosen scale.

Floor scale .005 (`coherence_finer/`): 12/12 successes, first correction 0.026 / 0.033 from the spine control (j4 / j7), 0.032 / 0.038 between directions, whole segment 0.033 / 0.035: corrections of the size of the perturbation itself (gain about 1-3). The first-order tube is being recollected at this scale (`tracking_fine/`: controls 0-13, radii .003/.01/.03, 6 fresh directions each, 252 branches), followed by DAgger rounds with the same planner.

## The labels are sampling noise (2026-10-06, `noise_probe/planner_noise.py`, floor .005, 6 seeds; 5 of 14 controls before the probe's time limit)

| t | spine state: deviation of the returned control from a_t, median / max (required correction 0) | perturbed state r = .01: mean correction / seed spread | signal-to-noise |
| --- | --- | --- | --- |
| 0 | 0.048 / 0.068 | 0.036 / 0.038 | 0.95 |
| 1 | 0.051 / 0.060 | 0.014 / 0.031 | 0.44 |
| 2 | 0.041 / 0.052 | 0.022 / 0.029 | 0.77 |
| 3 | 0.032 / 0.054 | 0.015 / 0.026 | 0.59 |
| 4 | 0.031 / 0.057 | 0.022 / 0.018 | 1.23 |

The planner's deviation from the reference at a state where no correction is needed (0.03-0.05) equals the size of the "corrections" in the fine tube (0.03), and at perturbed states the seed spread equals the mean correction. The tube's labels are therefore CEM sampling noise around the reference, not a function of the state; the planner's 0.5 s horizon leaves 942-1008 of 1,050 candidates feasible (`noise_probe/horizon_noise.py`, control 4), so the cost does not discriminate among them, and the turn is completed by re-planning from the true state every control rather than by any correction a policy could reproduce. This explains every learner result above: a linear law or an MLP fitted to these rows generalises no better than the reference action alone. The question moved to the labeller: does a longer horizon (1.0-1.5 s) make the corrections seed-consistent (probe running)?

## A longer planning horizon makes the labeller deterministic on the spine (2026-10-06, `noise_probe/horizon_noise.py`, control 4, floor .005, 6 seeds, r = .01)

| horizon | spine state: deviation from a_4 (median over seeds) | perturbed state: mean correction / seed spread | signal-to-noise | feasible candidates of 1,050 (spine, perturbed) |
| --- | --- | --- | --- | --- |
| 0.5 s (archived) | 0.031 | 0.027 / 0.020 | 1.4 | 942, 1008 |
| 1.0 s | 0.003 | 0.012 / 0.018 | 0.7 | 489, 447 |
| 1.5 s | 0.000 | 0.015 / 0.018 | 0.8 | 380, 421 |

With a 1.0-1.5 s horizon the feasibility constraint removes half to two thirds of the candidates and the reference wins exactly at the spine state, so the labeller is deterministic where no correction is needed. At a perturbed state the returned corrections still scatter by 0.018 around a mean of 0.012-0.015: the feasible-and-cheap set has a diameter of a few hundredths and one CEM run lands at a random point of it. The label of a feedback law is therefore the centre of that set, estimated by averaging independent seeds while the branch is off the spine (the collector's `--avg-seeds 4 --avg-steps 3`); the gate collection `tracking_lh/` (controls 0-9, radii .01/.03, 6 fresh directions, horizon 1.0 s) tests whether such labels are learnable.

`tracking_fine/` was stopped at 244 of 252 branches (2026-10-06) once the noise probe showed its single-seed 0.5 s-horizon labels to be sampling noise; the stored branches remain verified cold-reset trajectories.

## More CEM iterations make the labeller worse: the task cost prefers leaving the reference (2026-10-06, `noise_probe/iterations_probe.log`, 1.0 s, control 4)

| iterations | spine state: deviation from a_4 | perturbed state: mean / spread | feasible candidates |
| --- | --- | --- | --- |
| 3 | 0.003 | 0.012 / 0.018 | 489 of 1,050 |
| 8 | 0.035 | 0.016 / 0.021 | 1,375 of 2,800 |

With more refinement the search leaves the warm start and finds feasible candidates of lower task cost away from the reference: the archived cost (rotation progress, pose terms) has a gradient away from the demonstration, and the reference survives only because three iterations barely move. The tracking planner's proposal distribution therefore does not make it a tracking controller; its "corrections" are the task cost's preference for a different rotation, sampled stochastically. A tracking penalty on the cost itself (`tracking_cem.TrackingPenaltyEvaluator`: weight times the per-step mean squared deviation from the reference, added to every candidate) makes the reference the unique optimum where it is feasible and the nearest feasible path the optimum elsewhere; its weights are being probed (`noise_probe/tracking_cost_noise.py`).

## A tracking penalty makes the labeller deterministic (2026-10-06, `noise_probe/tracking_cost_noise.py`, 1.0 s, control 4, 6 seeds, r = .01)

| penalty weight | spine state: deviation from a_4 | perturbed state: mean correction / seed spread | feasible (spine, perturbed) |
| --- | --- | --- | --- |
| 0 | 0.003 | 0.012 / 0.018 | 489, 447 |
| 100 | 0.007 | 0.012 / 0.018 | 496, 447 |
| 1,000 | 0.003 | 0.007 / 0.011 | 492, 442 |
| 10,000 | 0.000 | 0.004 / 0.005 | 502, 428 |

With weight 10,000 (a 0.01 deviation per actuator costs about as much as the whole task cost of 21) the reference wins exactly where it is feasible and the planner's output at a perturbed state is reproducible to the sampling resolution (0.005). The mean correction at the early perturbed state goes to zero with the penalty: the failure caused by the perturbation lies beyond the horizon, so the correction is computed later, at the step where the predicted failure enters the horizon. The deterministic collection `tracking_tp/` (horizon 1.5 s, weight 10,000, no seed averaging; controls 0-9, radii .003/.01/.03, 6 fresh directions) replaces `tracking_lh/` (stopped at 42 of 120 after its gate evaluation).

First deterministic branches (`tracking_tp`, r = .003, 2026-10-06): every correction stays within 0.005-0.045 of the time-aligned spine control while the observation drifts 0.4-4 away from the spine's within five to eight controls, and the turn is completed in the spine's own number of controls. With the penalty on actuator targets the planner keeps the actions near the reference rather than the state; the branch states leave the spine's neighbourhood and the small adjustments at those states are what decide success. The learnability gate on these labels (`feedback_bc_0001/gate_eval.sh tracking_tp ...`) is the measurement that matters; if the adjustments are not a learnable function of the state, the next labeller variant is a penalty on the state trajectory (object pose and joint positions against the spine's recorded states), which would keep branches in the spine's neighbourhood.

## A state-trajectory penalty does not keep branches near the spine (2026-10-06, `noise_probe/state_penalty_smoke*.log`; control 6, r = .003, 1.5 s, action penalty 10,000, two seeds per weight)

| state weight | first corrections (two seeds) | observation distance from the spine after 1 / 5 / 7 controls (two seeds) | success |
| --- | --- | --- | --- |
| 0 | 0.023, 0.008 | 0.02 / 1.85 / 2.47 and 0.02 / 0.87 / 2.34 | 2/2 |
| 100 | 0.030, 0.003 | 0.02 / 2.00 / 2.87 and 0.02 / 0.25 / 1.67 | 2/2 |
| 1,000 | 0.029, 0.020 | 0.02 / 1.05 / 3.70 and 0.02 / 0.53 / 0.75 | 2/2 |

The penalty on the rolled-out hand-joint and object-pose state (`StateTrackingPenaltyEvaluator`) neither reduces the drift (which is seed-dependent and reaches 1-4 within seven controls at every weight) nor changes the outcome; the drift lives in components the planner cannot pull back without losing feasibility, and a feedback policy must operate at those states. The same rows show the labeller's reproducibility at this state: first corrections differ by 0.015-0.027 between seeds at every weight, i.e. the reference is infeasible within the horizon here and the set of equally cheap feasible deviations is about 0.02 wide. Whether more CEM iterations converge inside that set is being probed (`noise_probe/penalty_iterations_probe.log`).

## DAgger with the deterministic labeller (2026-10-06, `dagger_tp_round1/`, policy = `feedback_bc_0001/runs/tp_partial41`)

From spine prefixes 0-9 the policy survives 4-23 controls (one success, from prefix 9); branches after 1-4 of its own controls: 35 attempted, 24 stored (all b = 1-2 and most b = 3; the b = 3-4 branches from prefixes 0-2, 7-8 exhausted the 25-correction budget or found no feasible plan, i.e. after three or four policy controls the state is beyond what the penalised 1.5 s planner recovers). The policy's first deviation from the spine control is 0.005 (median), inside the labeller's own resolution, so the divergence it suffers is the chaos acting on sub-resolution errors, which the branches then label at the states it actually reaches.
