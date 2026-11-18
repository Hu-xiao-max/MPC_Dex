# Actor restricted to on-policy rows: no prefix-6 success, prefix-7 retention still lost, larger actor excursion

**Excluding demonstration rows from the SAC actor and temperature losses did not stop the drift diagnosed in SAC0004.** No frozen prefix-6 success at any checkpoint (0/24 trials); prefix-7 retention fell from 6/6 at baseline to 0/6 at online 500 and never recovered (0/18 across the three online checkpoints); zero of 130 completed online training episodes succeeded (SAC0004, same seeds: 2 of 94; both are changing-policy stochastic episodes on one seed set, so the difference is not attributable to the flag). The actor's deterministic distance to the MPC handoff label and its latent std were larger than SAC0004's at every online checkpoint. The stage completed its full budget (500 mixed critic updates, 2,000 mixed SAC updates, exit 0, 552.2 s wall).

## Actual execution

- Inherited SAC0001 final checkpoint (SHA `e4a73d89…`); final counters BC 12,000 / pretraining 1,000 / online 4,000, optimizer steps critic 5,000 and actor/alpha 4,000, as prescribed (`checkpoints/online_2000/state.json`). Single seed set, matched to SAC0004.
- Warm-up: 256 frozen-actor prefix-6 episodes, 812 suffix rows, all drops after 2–5 policy controls, zero successes (informational). `warmup_calibration_*`, `heldout_calibration_*` and `warmup_record.json` are byte-identical to SAC0004's (after pretraining: .079/.076 and .033/.036 RMSE); the demonstration-row biases before and after pretraining are also identical. The flag acts only in `runner_update`; `entropy_masked_pretrain_step` performs critic step, Polyak update and counter only, so pretraining reproduces SAC0004 exactly.
- Demonstration buffer: the same 8 rows (5 MPC, 3 recorded), 77 demonstration + 179 online rows per batch in pretraining and online; during pretraining 46–67 demonstration rows per update bootstrapped entropy-free and the demonstration target mean rose .407 → 1.036. Every one of the 2,000 online updates used 179 actor/temperature rows and 77 demonstration rows in the critic step (`actor_rows_used` = 179, `batch_demonstration_rows` = 77 in all 2,000 `online_update` records); `actor_bc_loss` null throughout.
- Online: 2,000 interactions, 130 completed assisted episodes, 0 force-qualified successes, 85 palm-support failures, 37 drops, 8 horizon truncations (episode 21 and episodes 122, 123, 125–129; episode 124 dropped after 74 controls); episode 130 remains open at 91 controls (`unfinished_suffix/report.json`). Peak online progress 6.250 rad (episode 124, then dropped). Online replay holds 2,812 rows. All 2,322 physical prefix controls (1,536 warm-up, 780 online, 6 open episode) replayed bit-exactly and never entered replay; no handoff terminated before the policy.
- Learned alpha .000305860 → .000376772 (raw-equivalent .0754), strictly increasing over the 2,000 updates.

## Demonstration calibration

Critic bias on the MPC rows: −1.258/−1.346 → −.125/−.093 after pretraining → +.107/+.104 (500) → −.281/−.280 (1,000) → −.208/−.198 (2,000); recorded rows −.677/−.619 → −.061/+.014 → +.019/+.014 → −.389/−.394 → −.407/−.403. Deterministic actor-mean distance to the MPC handoff label .961 (baseline and after pretraining) → 2.930 → 2.628 → 1.398 (SAC0004: .961 → 2.006 → 1.212 → .327); to the recorded label .413 → 2.664 → 2.394 → .827 (SAC0004: .413 → 1.781 → .629 → 1.242). Latent std at the handoff .0674 → .7814 → .5786 → .3384 (SAC0004: .0674 → .2129 → .3486 → .1022). Both distances end further from the baseline than they started; versus SAC0004 the MPC distance and the std are larger at every online checkpoint, while the recorded-label distance is larger at 500 and 1,000 and smaller at 2,000.

## Frozen evaluations

81 frozen trials: at each of four checkpoints one deterministic and five stochastic trials for prefixes 6, 7 and 8 plus one cold deterministic trial; five cold stochastic trials at online 2,000. Evaluations restored training RNG and left learner tensors and counters unchanged; all 72 prefix trials replayed bit-exactly.

| Checkpoint | Prefix-6 det. | Prefix-6 stochastic 17501–17505 | Prefix-7 det. / stochastic 17521–17525 | Prefix-8 det. / stochastic 17531–17535 | Cold det. | Cold stochastic 17511–17515 |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | Drop, 3 | 0/5, all drops at 3 | Pass, 2 controls / 5/5 | Pass, 1 control / 5/5 | Drop, 1 | not scheduled |
| Online 500 | Palm, 6 | 0/5, all palm (2–8) | Palm, 4 / 0/5 all palm (3–8) | Pass, 1 / 4/5 (17535 palm at 3) | Drop, 5 | not scheduled |
| Online 1,000 | Palm, 4 | 0/5 (4 palm at 3–5, 1 drop at 20) | Palm, 15 / 0/5 (4 palm at 4–7, 1 drop at 38) | Pass, 1 / 5/5 | Held 100, no turn, 1.33 rad final / 1.66 peak | not scheduled |
| Online 2,000 | Held 94, no turn, 4.87 rad final / 5.51 peak | 0/5, all held 94 (final 4.57–5.08, peak 5.29–5.36) | Held 93, 5.71 / 5.92 / 0/5 (3 held at 93 at 5.68–5.77; 2 drops at 92 and 87) | Pass, 1 / 5/5 | Drop, 75 (peak 2.77) | 0/5, all drops (6–26 controls) |

Prefix 6: 0/24. Prefix 7: 6/6 at baseline, 0/18 online. Prefix 8: 23/24 (deterministic passed at every checkpoint; one stochastic palm failure at online 500); SAC0004 evaluated only prefixes 6 and 7. Cold: 0/9. Eleven evaluation trials ended by truncation, all failures with unknown tails; in every held prefix-6 trial the final angle is below the 5.258 rad force-qualified handoff angle, so the object slipped back while held with 3–4 loaded fingers and zero palm force.

## Interpretation and scope

The hypothesis was that SAC actor updates on demonstration states drive the drift. With those updates removed the outcome is unchanged in kind: retention of the two-control prefix-7 completion is lost by online 500 in both stages, before any prefix-6 success exists to reinforce; the actor mean at the handoff moves away from both demonstration labels; std inflates to .78 before partially contracting. Actor gradients evaluated at demonstration states are therefore not necessary for the drift; it arises from the on-policy rows (prefix-6 failures, later long holding episodes) and the critic, which still trains on demonstration rows. Whether those gradients also contributed in SAC0004 is not tested here. The critic remains calibrated on demonstration rows (|bias| ≤ .41), which is again not sufficient for the actor to reproduce them.

Single seed; a seed replicate would be needed to separate the differences from SAC0004 (0 vs 2 training successes, larger actor excursion) from seed variance. All warm-up, online and prefix evaluations are assisted (prefix-replayed) and demonstration-mixed, so none is primitive evidence. Cold-reset results decide deployment eligibility: all nine cold trials failed, so no checkpoint from this stage is deployable or a valid initialization for a later stage. SAC0001 remains the last checkpoint with verified assisted competence.

Inputs and protocol: `protocol.json`, `execution_manifest.json`, `input_bindings.json`; results: `summary.json`, `warmup_record.json`, `metrics.jsonl`, `episodes/`, `evaluations/`, `checkpoints/`, `demonstration_buffer.json`, `demonstration_calibration_*.json`, `warmup_calibration_*`, `heldout_calibration_*`, `unfinished_suffix/`; final hashes: `manifest.json`.


_Audit note: drafted by an auditing agent from the artifact files and independently refuted-checked; 7 claim(s) corrected by the verifier before writing._
