# Prefix-6 SAC with mixed off-policy demonstrations; prepared, not executed

The trained SAC0001 checkpoint failed every one of 262 prefix-6 attempts, so the success-gated warm-up of [SAC0002](../bootstrap_suffix_sac_0002/findings.md) could not learn. The [handoff MPC recovery](../mpc_prefix6_handoff_recovery_0001/findings.md) then completed a verified turn from the identical handoff in five corrections. This stage learns from those rows. It is a reviewed protocol change: the learning gate is the presence of verified off-policy positives, not warm-up policy successes.

Inherit the entire SAC0001 final checkpoint (BC 12,000 / critic pretraining 500 / online 2,000, SHA `e4a73d89…`), unchanged through SAC0002: networks, targets, optimizer moments and step counts, learned std, learned alpha (0.00030585992499254644) and the cumulative actor learning-rate ramp. Fresh physical reset, fresh online replay and new RNG (seed 17391). At every reset the six recorded controls are physically replayed and verified bit-for-bit; the 100-control horizon includes the prefix.

**Demonstration buffer.** A separate immutable buffer holds exactly 8 rows in two terminal-success episodes that start from the identical handoff observation: the 5 executed MPC corrections (recovery controls 7–11) and the 3 recorded source controls 7–9. Successor actions inside each episode are the actual next executed controls, so unchanged SoftSARSA applies; no successor is fabricated and no prefix row enters any buffer. Both sources are hash-bound and re-verified against the recovery trajectory and the source prefix at runtime.

**Mixing.** Every update batch uses the main-method fixed split through the existing `sample_mixed`: 77 demonstration rows and 179 online rows per 256. With only 8 demonstration rows each is drawn about 9.6 times per batch; this repetition is disclosed rather than hidden by a smaller fraction. Sampled batch rewards alone divide by 200; raw rewards stay in both buffers.

The single prescribed path is:

1. Collect exactly 256 complete frozen-actor prefix-6 episodes with new seed 17391 into the online buffer. Warm-up successes are recorded but not required.
2. Run exactly 500 SoftSARSA critic updates on mixed batches with the actor and temperature unchanged, then exactly 2,000 UTD-1 SAC interactions/updates on mixed batches. Final counters: BC 12,000 / pretraining 1,000 / online 4,000; optimizer steps critic 5,000, actor/alpha 4,000.
3. Evaluate at baseline and at online 500/1,000/2,000: one deterministic prefix-6 trial, five stochastic prefix-6 trials (17401–17405), one deterministic cold trial; five stochastic cold trials (17411–17415) at the final checkpoint only; prefix-7 retention (deterministic plus 17421–17425) at baseline and final. Evaluations restore training RNG and never enter replay.
4. Demonstration calibration at baseline, after pretraining and at each online checkpoint: critic values on the 8 rows against entropy-free discounted returns, deterministic actor-mean distance to both control-7 labels, and actor log density of the demonstration actions.

Full learner/optimizer/replay/RNG/environment state is saved at every checkpoint; the demonstration buffer is saved once. All assisted and demonstration-mixed data remain ineligible for primitive export; a later qualified 60D policy still requires an independent frozen cold-policy collection.

```bash
.venv/bin/python artifacts/bootstrap_suffix_sac_0003/prepare.py
.venv/bin/python -m pytest -q artifacts/bootstrap_suffix_sac_0003/test_protocol.py
# Only on separate explicit execution authorization:
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_suffix_sac_0003/run.py --execute > artifacts/bootstrap_suffix_sac_0003/run.log 2>&1
```
