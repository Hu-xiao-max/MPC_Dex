# Independent review of the clock recovery and 86-pair constraints

The 86-row dataset is authentic and internally consistent. I found no fabricated observations, clock reset at the MPC handoff, conflicting identical inputs, or unsupported recovery success. This does **not** establish that a policy fitting these points will reproduce the successful trajectory. The completed 86-row BC checkpoints and its one head-refined policy all failed their strict physical audits.

This review ran no physics, optimization, or additional head solve. `review.py` independently verifies hashes, traces every label to its measured source, recomputes force-qualified progress from saved 2 ms samples, evaluates frozen networks, and checks checkpoint differences. Numerical results and input hashes are in `checks.json` and `inputs_manifest.json`; frozen inference arrays are also saved. All five source artifact manifests passed. The review leaves the source artifacts unchanged.

## Data and clock integrity

- Rows 1–73 are bit-identical to `bc_common_noise_clock_0001/supervised_pairs.npz`. Rows 74–81 preserve the actual clock actor's controls 2–9 and their visited observations. Rows 82–86 preserve the five actual MPC corrections, controls 10–14. They are supervised pairs from multiple recorded trajectories, not one invented 86-transition episode.
- All 86 raw 60D observation/action pairs match their referenced source transitions. Every appended float32 clock agrees with its actual measured source physics boundary. The new recovery's clocks use simulator time relative to the original cold reset; they continue through the 0.9 s MPC handoff. Its next observations, raw observations, commands, and termination flags form one actual 14-control episode.
- The new reset observation is exactly the original reset input, while its learned action differs by up to `2.086162567e-7`. Excluding that additional reset label avoids an exact-input conflict. No exact duplicates remain among either the 60D or 61D inputs. The closest new pair is old row 2/new row 74: observation L2 `3.755163342e-6`, same clock, action maximum difference `6.556510925e-7`. Close inputs are not themselves contradictory labels.
- The local BC clipping epsilon `.01` preserves all labels. Main epsilon `.02` would change **112 action components** across the 86 rows. The five new MPC corrections survive `.02`; the inherited data and learned prefix require the explicitly recorded diagnostic interface. Maximum recorded action magnitude is `.9800992608`.

## What the recovery demonstrates

The actual nine-command actor prefix matches every available source transition field and all 450 source physics samples. The original actor archive lacks full integration snapshots; this review makes no claim of comparing an unavailable snapshot. The new branch snapshot matches the recorded ninth post-control qpos, qvel, ctrl, integration state, and subsequent observation. The recovery source preserves the live environment at the switch and asserts that each planning call leaves it unchanged. Reference projection affects only a search candidate: three projected batches, maximum `7.357040048e-5` radians. All five selected prediction diagnostics identify the corrected float32 action execution mapping; the archived evaluator applies that mapping before batch simulation.

The independently saved replay matches every new transition field and all 700 physics samples. Recomputing qualification from measured twist, loaded fingers, tilt, and sticky palm status gives final force-qualified progress **6.498011614 rad**, peak **6.980686441 rad**, three final loaded fingers, and final tilt `.123046955` rad. Palm, table, and other support forces are zero. Four prefix samples—`.212`, `.864`, `.866`, `.868` s—lack the required two loaded fingers; positive progress at those samples is correctly excluded. Thus neither every prefix instant nor raw angle alone is the success criterion. All five MPC control intervals are fully force-qualified in this trace. No post-success continuation was tested.

The eight added prefix labels are bit-exact outputs of the old polished clock actor at their saved inputs. They constrain the actual route into the recovery, but do not constitute eight new expert corrections. At the five actual MPC states, that old actor's action differs from the demonstrated correction by maximum components `.2793`, `1.3266`, `1.9255`, `1.3283`, and `1.2021`. These are substantial new feedback constraints. Their nearest old training observations have mixed-unit L2 distances `2.74–13.51`; this is descriptive coverage evidence, not a physical distance or a robustness margin.

## Numerical constraints and actual outcomes

The previous 73-row head solve had condition number `3.13e6`; its float32 singleton training error was only `4.47e-7`, yet the actual policy dropped after 19 controls without a turn. Its saved trace already shows sensitivity before large command differences: at control 3 the command differs from the nominal reference by at most `2.98e-7`, while the post-control qpos/qvel differences reach `.00170` and `.06924`. These comparisons include accumulated prior differences; they do not isolate a cause or prove the head solve alone caused the divergence.

Adding the 13 authentic rows to that fixed old trunk yields a full-row-rank float64 86×257 design with condition `9.26e7`. The **actually retrained 86-row trunk** has condition `8.35e7`. It is also full row rank under float64 tolerance, with no exact feature collisions. Many singular directions fall below the conventional float32 matrix-rank tolerance (64 of 86 remain under that tolerance for the new trunk). That sensitivity warning does not mean the float64 solve failed or that the labels are mathematically inconsistent.

I independently verified that the new head candidate changes exactly `actor.mean_head.weight` and `actor.mean_head.bias` from the final 86-row BC checkpoint. All other checkpoint tensors, optimizer states, and counters are bit-identical: BC 12,000, critic pretraining 0, online updates 0. Its saved correction is applied by the recorded float32 addition, with ideal float64 residual maximum `4.33e-15`; correction norm is `7.0718`, and head weight norm rises `3.0901 → 7.7055`. Recomputed singleton features, targets, and predictions match the saved solve arrays exactly. Large condition and head norm do not establish a useful off-data feedback direction.

The new candidate's deterministic inference on all 86 saved observations, using CUDA float32, one intra-op/interop thread, and both TF32 flags disabled:

| Input shape | Maximum error against actual labels | Maximum difference from rank-1 inference |
| --- | ---: | ---: |
| `(61,)`, one row at a time | `4.768371582e-7` | 0 |
| `(1,61)`, one row at a time | `4.768371582e-7` | 0, bit-identical |
| `(86,61)`, full batch | `5.662441254e-7` | `5.364418030e-7` across 593 components |

This supports deterministic rank-1/batch-one parity only for these observations and runtime. It does not establish stochastic sampling parity, other batch sizes, devices, precision settings, or future visited states. Same weights alone are insufficient runtime evidence for the batched bootstrap runner.

The 3k/6k/12k 86-pair BC audits all failed through palm support. Its final maximum training error was `.0199595`; it was not an exact interpolant. The one refined candidate did interpolate closely, but its actual strict audit still dropped after 19 controls: peak force-qualified angle **5.644944168 rad**, final **−.738627121 rad**, no palm support and no full turn. I recomputed all 19 commands from the frozen candidate and all force-qualified angles from the saved physics; both agree. This review did not physically rerun that audit.

## Narrow next-step cautions

The current data provide genuine visited-state correction coverage and contain no demonstrated exact state/action contradiction. Their observation interface still omits instantaneous joint velocities, object angular velocity, contact forces, solver integration state, retained-angle history, and sticky failure timers; the velocity slots are control-interval finite differences. A clock disambiguates elapsed schedule position but does not restore those hidden quantities. No exact harmful alias is demonstrated by these 86 rows, so changing the observation interface based only on this concern would be premature.

The measured limitation is that small training error does not preserve the physical route into the corrected states. Keep the actual 2 ms qualifications, action dtype/mapping, inference shape/runtime, and visited-state comparison as separate validation gates. Evaluate an imported actor using the intended runner's exact inference and sampling path before attributing a changed outcome to learning. Do not infer continuous stability, stochastic robustness, SAC success, or main-policy compatibility from a conditioned head solve, an MPC-assisted successful episode, or this 61D local clock diagnostic. No additional training or physical branch is recommended as an automatic extension of this review.
