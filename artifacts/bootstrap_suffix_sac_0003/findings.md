# Mixed-demonstration prefix-6 SAC: no frozen success, prefix-7 competence lost

**The stage completed its full budget (500 mixed SoftSARSA updates, 2,000 mixed SAC updates, exit 0, 187.1 s loop) and produced no frozen-policy success at any checkpoint.** Only two of 228 completed online training episodes succeeded, both within the first ten episodes. Worse, the final checkpoint fails all six prefix-7 retention trials that the inherited checkpoint passed 6/6 at baseline. The verified cause of the value distortion is the unchanged SoftSARSA entropy term applied to off-policy demonstration actions far outside the actor's distribution. This is a negative result with a diagnosed mechanism, not a successful policy.

## Actual execution

- Inherited SAC0001 final checkpoint (SHA `e4a73d89…`), counters BC 12,000 / pretraining 500 / online 2,000; final counters 12,000 / 1,000 / 4,000, optimizer steps critic 5,000 and actor/alpha 4,000, as prescribed.
- Warm-up: 256 frozen-actor prefix-6 episodes, 814 suffix rows, all drops, zero successes (informational; the learning gate was the verified 8-row demonstration buffer).
- Demonstration buffer: exactly 8 rows in two terminal episodes from the identical handoff observation, 5 executed MPC corrections and recorded controls 7–9, verified against the recovery trajectory and source prefix at runtime; every update batch drew 77 demonstration and 179 online rows.
- Online: 2,000 policy interactions; 228 completed assisted episodes, 2 force-qualified successes (episodes 8 and 9, five suffix controls each, at cumulative updates 2,032 and 2,037), 48 drops, 177 palm-support failures, one horizon truncation. Successes per 100 completed episodes: 2, 0, 0 (final bin 28). The last episode remains open; final online replay holds 2,814 rows. All 2,910 physical prefix controls replayed bit-exactly and never entered replay.
- Learned alpha rose from .000305860 to .000391152 (raw-equivalent .0782).

## Frozen evaluations

| Checkpoint | Prefix-6 deterministic | Prefix-6 stochastic 17401–17405 | Cold deterministic | Cold stochastic 17411–17415 | Prefix-7 retention det. / stochastic |
| --- | --- | --- | --- | --- | --- |
| Baseline (inherited) | Drop, 3 controls | 0/5, all drops | Drop, control 1 | not scheduled | Pass / 5/5 |
| Online 500 | Palm, 3 | 0/5, all palm | Drop, 9 | not scheduled | not scheduled |
| Online 1,000 | Palm, 4 | 0/5 (3 drops, 2 palm) | Drop, 3 | not scheduled | not scheduled |
| Online 2,000 | Palm, 8 | 0/5, all palm | Palm, 6 | 0/5 (3 palm, 2 drops) | Fail / 0/5, all palm |

At online 500 every prefix-6 trial peaks exactly at the handoff angle 5.258154 rad: the policy makes no rotational progress and braces on the palm. Final prefix-6 trials survive 5–17 controls but end in palm support at 4.77–5.30 rad. Final prefix-7 retention trials, which previously completed the turn in two controls, now hold for 5–13 controls and fail by palm support at 4.86–5.16 rad. This is a loss of the previously learned assisted competence, not merely a failure to acquire prefix-6 competence.

## Diagnosed value distortion

The unchanged SoftSARSA target is r + γ(min Q′(s′,a′) − α log π(a′|s′)) with α = .0001 and the recorded next action a′. For on-policy rows log π is of order −10² at most. For the MPC demonstration actions the frozen actor's log densities are −1,018.6, −155,330.8, −20,034.5, −83,825.2 and −28,523.0 (latent std ≈ .067, so the actions lie tens of standard deviations from the mean), giving entropy bonuses of +15.53, +2.00, +8.38 and +2.85 normalized reward units per bootstrap step (raw-equivalent up to 3,107). The analytic SoftSARSA targets for the five MPC rows are 29.30, 14.06, 12.22, 3.90 and 1.05 against entropy-free discounted returns of 1.08–1.13, and the pretrained critic reproduces them: 29.40, 14.30, 12.33, 3.87 and 1.09 (critic 0). The recorded continuation rows, whose actions lie closer to the actor (log densities −16 to −1,073), are fitted near their true returns (1.04–1.16). Warm-up and held-out on-policy calibration nonetheless improved (RMSE .311/.332 → .150/.140 and .313/.320 → .172/.148), so the distortion is confined to the off-policy demonstration states.

Consequently the critic assigned values of about 29 to the MPC first correction at the handoff state, roughly 27 above any realizable soft return. The actor moved toward that region and its latent std expanded to .744 by online 500 (deterministic-mean L2 distance to the MPC label 0.96 → 3.11, to the recorded label 0.41 → 3.37), then partially contracted (std .117; distances 1.28 and 0.61 at 2,000) while the demonstration-state biases fell to −.14/−.18 (MPC) and +.82/+.84 (recorded). The phase of large std coincides with the shift from drops to palm-support failures.

## Interpretation and scope

Mixing sparse off-policy demonstrations into the existing SoftSARSA pretraining is unsound whenever the demonstration actions are improbable under the current actor: the entropy correction, valid for on-policy successors, becomes a large positive fictitious reward. bootstrap_0005 seeded nine MPC rows without a visible instance of this because its actor was BC-initialized on the same trajectory. Any further use of these labels needs a target that does not depend on the actor's density of demonstration actions (for example an entropy-free or clipped-density successor term for demonstration rows, or a separate supervised objective), and must be stated in a new protocol. The 30/70 repetition of 8 rows and the changed reset curriculum are additional confounds; this run does not isolate them.

The prefix-6 warm-up, checkpoints, evaluations, calibrations and per-episode physical evidence are complete and independently replayable. No trajectory qualifies for a primitive. The SAC0001 checkpoint remains the last one with verified assisted competence at prefix 7; the SAC0003 checkpoints should not be used as an initialization without addressing the distortion above.

Inputs and protocol: `protocol.json`, `execution_manifest.json`, `input_bindings.json`; results: `summary.json`, `warmup_record.json`, `metrics.jsonl`, `episodes/`, `evaluations/`, `checkpoints/`, `demonstration_buffer.json`, `demonstrations_replay.npz`, `demonstration_calibration_*.json`, `warmup_calibration_*`, `heldout_calibration_*`, `unfinished_suffix/`; final hashes: `manifest.json`.
