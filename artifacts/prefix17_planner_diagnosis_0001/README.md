# Hold-candidate diagnosis at probe0002 prefix 17

The first 17 recorded controls replayed under the strict no-palm rotation environment with qpos/qvel matching the original recording bit-for-bit. The resulting state had retained angle **5.97913920 rad**, two contacting fingers, tilt **0.08101695 rad**, and no palm support anywhere in the prefix.

Every score below uses the same 0.5 s horizon, retained-progress cost, tilt weight 100, and exact native transition backend with one worker.

| Candidate | Cost | Feasible? | Evidence |
| --- | ---: | --- | --- |
| Current executed target held constant | **21.08852056** | Yes | All 250 steps completed; future retained progress +0.62185842 rad; palm force 0 |
| Next five historically executed controls | ∞ | No | Palm support at absolute simulation time 1.912 s; normal force 9.2121 N |
| Historical next command held constant | ∞ | No | Contact loss at absolute simulation time 1.932 s |
| Best candidate from one fresh CEM plan | **21.80366337** | Yes | 350 candidates × 3 iterations, seed 42 |

The historical open-loop CEM forecast and its optimizer state were not saved. The historically executed controls are a separate recorded continuation; they are not presented as that missing forecast. The fresh plan resets its nominal trajectory around the current executed target, rather than reconstructing the unavailable optimizer warmstart.

**The fresh stochastic search missed a known feasible candidate with lower cost.** Its best result cost 0.71514282 more than holding, a 3.39% increase. Only 53 of its 1,050 sampled trajectories were feasible, none beat the hold cost, and none represented the exact hold. Even the nearest sampled trajectory differed from hold by at least 0.172913 rad in one joint target. The decoded unperturbed initial nominal was the hold trajectory to floating-point precision, but `PrimitiveCEM.plan` only scored noise-perturbed samples.

Executing the selected first command changed a target by as much as 0.275147 rad and reduced retained angle to **5.96146639 rad**, a change of **−0.01767281 rad**. It caused neither a drop nor palm support during that control interval. The independently recorded and audited hold branch advances to **6.23174516 rad** after one hold command and reaches **6.71509217 rad** after two. That branch's later hold instability remains a separate limitation; this comparison only establishes the 0.5 s feasibility and cost ranking at prefix 17.

This supports an explicit diagnostic option that scores the unperturbed nominal and current-target hold candidates alongside stochastic candidates. Both should use the same dynamics, constraints, and cost. In a reset state they may coincide; after warmstarting they can differ. Candidate accounting and the departure from the original sampling budget should remain explicit in diagnostic provenance. This experiment does not establish robustness across seeds or supply a successful simulation SAC policy or a production primitive.

Exactly one CEM plan was run. Shared numerical sources were unchanged before and after the experiment. Full results, input/source hashes, native build provenance, and per-iteration rejection counts are in [report.json](report.json); sampled sequences/costs and the prefix integration state are in [sequences.npz](sequences.npz). The artifact-local [driver](diagnose.py) refuses repeat execution once its report exists.
