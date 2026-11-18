# Findings: DAgger-style MPC labels at policy-visited states

Policy: `bootstrap_variant_0079_ms_suffix1to8_weighted_from74` checkpoint `online_4000` (agent.pt sha256 `bf81a1988a1f...`), deterministic inference (`mean.tanh()`), the same contract as the stage evaluators.
Demonstrations labelled: seed_194, seed_195, seed_196 (the three sources held out by stage 0079, on which its online_4000 policy scored 0/90 at every suffix length 1-8). Suffix lengths 1-6, branch points 0-3 policy controls, at most 20 MPC corrections per branch (archived recovery recipe: CEM 350/10/3, 6 knots, .5 s horizon, diagnostic PCA prior, .98 search bound fixed at the cold reset).

**Branches attempted: 48. Strict force-qualified successes: 28. Stored after bit-exact independent replay: 28.**

## Per demonstration and suffix

| demo | s | policy controls before failure (failure) | b=0 | b=1 | b=2 | b=3 |
|---|---|---|---|---|---|---|
| seed_194 | 1 | 3 (drop) | stored: 3 MPC | budget (20) exhausted | no feasible plan | n/a (policy episode already over) |
| seed_194 | 2 | 3 (drop) | stored: 4 MPC | no feasible plan | no feasible plan | n/a (policy episode already over) |
| seed_194 | 3 | 7 (drop) | stored: 4 MPC | stored: 10 MPC | budget (20) exhausted | boundary, not strict (None) |
| seed_194 | 4 | 1 (palm_support) | no feasible plan | n/a (policy episode already over) | n/a (policy episode already over) | n/a (policy episode already over) |
| seed_194 | 5 | 2 (palm_support) | stored: 2 MPC | boundary, not strict (None) | n/a (policy episode already over) | n/a (policy episode already over) |
| seed_194 | 6 | 3 (palm_support) | stored: 10 MPC | stored: 13 MPC | budget (20) exhausted | n/a (policy episode already over) |
| seed_195 | 1 | 2 (palm_support) | stored: 1 MPC | no feasible plan | n/a (policy episode already over) | n/a (policy episode already over) |
| seed_195 | 2 | 2 (drop) | stored: 2 MPC | no feasible plan | n/a (policy episode already over) | n/a (policy episode already over) |
| seed_195 | 3 | 3 (drop) | stored: 9 MPC | stored: 5 MPC | no feasible plan | n/a (policy episode already over) |
| seed_195 | 4 | 4 (palm_support) | stored: 3 MPC | no feasible plan | stored: 13 MPC | budget (20) exhausted |
| seed_195 | 5 | 1 (drop) | stored: 7 MPC | n/a (policy episode already over) | n/a (policy episode already over) | n/a (policy episode already over) |
| seed_195 | 6 | 2 (drop) | stored: 8 MPC | no feasible plan | n/a (policy episode already over) | n/a (policy episode already over) |
| seed_196 | 1 | 2 (drop) | stored: 1 MPC | no feasible plan | n/a (policy episode already over) | n/a (policy episode already over) |
| seed_196 | 2 | 4 (drop) | stored: 3 MPC | boundary, not strict (None) | stored: 2 MPC | no feasible plan |
| seed_196 | 3 | 2 (drop) | stored: 3 MPC | stored: 3 MPC | n/a (policy episode already over) | n/a (policy episode already over) |
| seed_196 | 4 | 2 (palm_support) | stored: 7 MPC | stored: 6 MPC | n/a (policy episode already over) | n/a (policy episode already over) |
| seed_196 | 5 | 8 (drop) | stored: 5 MPC | stored: 4 MPC | no feasible plan | stored: 14 MPC |
| seed_196 | 6 | 5 (drop) | stored: 5 MPC | stored: 5 MPC | stored: 9 MPC | boundary, not strict (None) |

## By branch point

| policy controls kept | attempted | no feasible plan | budget exhausted | boundary not strict | stored |
|---|---|---|---|---|---|
| 0 | 18 | 1 | 0 | 0 | 17 |
| 1 | 16 | 6 | 1 | 2 | 7 |
| 2 | 9 | 4 | 2 | 0 | 3 |
| 3 | 5 | 1 | 1 | 2 | 1 |

## Why most policy-visited states have no feasible MPC plan

Object state after each deterministic policy control from the handoff (`diag/policy_branch_states.json`; force-bearing fingers, contactless seconds, finite-cost candidates out of 350 in one planner batch):

- seed_195_s1: handoff: fingers 2, contactless 0.000 s; policy1: fingers 3, contactless 0.000 s, L2 to recorded control 2.80, max |a| 0.999, finite costs 0/350; policy2: fingers 2, contactless 0.000 s, episode ended: palm_support
- seed_196_s1: handoff: fingers 3, contactless 0.000 s; policy1: fingers 0, contactless 0.028 s, L2 to recorded control 4.08, max |a| 0.987, finite costs 0/350; policy2: fingers 0, contactless 0.128 s, episode ended: drop
- seed_194_s3: handoff: fingers 2, contactless 0.000 s; policy1: fingers 4, contactless 0.000 s, L2 to recorded control 3.77, max |a| 0.985, finite costs 16/350; policy2: fingers 3, contactless 0.000 s, L2 to recorded control 2.83, max |a| 0.998, finite costs 112/350; policy3: fingers 2, contactless 0.000 s, L2 to recorded control 2.56, max |a| 0.998, finite costs 103/350

The policy's controls at these never-seen states are far from the recorded ones (L2 2.5-4.1 in action space against a success basin of .02-.05) and near-saturated; after one such control the object has often lost finger contact or is heading for the palm, so every .5 s rollout of the planner ends in a rejected state. Labels therefore exist only at the states where the object is still held.

## Status

Every stored episode is a full cold-reset trajectory (demonstration prefix, 0-3 policy controls, MPC corrections) that replays bit-for-bit; it is labelled per control (`control_origin`). The policy rows are states to be labelled, never targets; the template uses only the MPC segment after the last policy control as demonstration rows. Nothing here is primitive-eligible, and no deployment claim follows from this stage; the labels are inputs to `bootstrap_variant_0080_ms_dagger_from79`.
