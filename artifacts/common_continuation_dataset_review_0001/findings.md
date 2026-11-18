# Common-continuation dataset review

The proposed **12 supervised pairs** remove the demonstrated continuation-label conflict exactly. Nominal rows 1–9 come from the successful common continuation; the added three pairs are only actual MPC corrections at actor-state rows 7–9. The actor prefix and original teacher continuation rows 7–22 are excluded. This is a supervised union of two branches, not a single replay episode.

| Corresponding nominal / actor-correction row | Observation L2 | Observation max | Action max difference | Pre-tanh target max difference |
| --- | ---: | ---: | ---: | ---: |
| 7 | 0.0130841462 | 0.00667476654 | 0 | 0 |
| 8 | 0.284974819 | 0.233534932 | 0 | 0 |
| 9 | 0.525219042 | 0.323663451 | 0 | 0 |

Previously, original teacher row 7 and actor correction row 7 differed by 0.765485778 in normalized action (L2 1.77603247) and 2.07965891 in pre-tanh target (L2 5.14975031), despite the same observation L2 0.0130841462. The new labels remove that demand for a steep local change.

**An inherited startup issue remains.** The nearest pair in both datasets is nominal rows 1 and 2: observation L2 0.00541486706, maximum difference 0.00308915949; action L2 1.77027179, maximum 0.881450314; pre-tanh target L2 3.27310856. The maximum pairwise action/observation ratio (326.928024) and latent/observation ratio (604.46702) are therefore unchanged by this dataset revision.

The first command holds the initial executed target, and the first two observations have identical previous-action inputs. Their differences reflect tiny settling motion: maximum joint displacement 0.000308931 rad, principal angle change 0.001305502 rad, and finite-difference joint velocity maximum 0.003089159 rad/s. Thus a memoryless actor must still distinguish initial hold from subsequent motion using very small physical differences. This is not an exact duplicate-observation contradiction: every observation is distinct. It is a remaining sensitivity concern, not proof that fitting or physical execution must fail.

Raw observation L2 mixes units and is reported only as a reproducible coordinate distance, not a physical metric. Semantic group distances are in report.json. The nominal/actor differences after identical correction commands grow mainly in finite-difference joint velocities; identical actions do not establish state convergence.

Existing strict audits independently show both branches reaching a supported full turn after 9 controls, without palm/table support or drop: nominal force-qualified rotation 6.649846605 rad; actor-recovery 6.685409149 rad. Their saved 2 s fixed-target holds retain 7.573568368 and 7.579596343 rad, respectively. This review verified hashes, matched environment contracts, action identity, transition chains, and terminal cutoffs; it did not rerun physics. Both branch outcomes support this common continuation, but do not demonstrate a learned actor reproduces it or is robust nearby.

All 12 action labels remain unchanged under the actual float32 ±0.98 BC clamp. Corresponding targets were computed with float32 Torch atanh on CUDA; distances use float64 differences. No model fitting, planning, physical execution, checkpoint changes, or shared-source edits occurred.
