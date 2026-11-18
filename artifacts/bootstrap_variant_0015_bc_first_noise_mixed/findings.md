# First-row BC plus sigma .025 handoff noise with mixed prefixes: prefix-6 deterministic passes only at online 250, 750 and 1,250, stochastic at most 2/5 (11/30 large-sample), prefix-7 retention 0/6 at 750 and 2/6 at 1,000, back to 6/6 by 1,750, perturbed handoffs never above 3/5, cold fails

**Combining the two wave-1 winners reproduced neither.** The frozen policy completed the turn from the prefix-6 handoff deterministically at three of eight online checkpoints (250, 750, 1,250; 3 controls each) and in at most 2 of 5 stochastic trials (1,250); 22 of 167 completed prefix-6 training episodes succeeded (0004: 109 of 432). Prefix-7 retention fell to 0/6 at online 750 (deterministic palm-support failure, stochastic 0/5) and was 2/6 at 1,000 (deterministic pass, stochastic 1/5), 4/6 at 1,250, 3/6 at 1,500, then 6/6 at 1,750 and 2,000; prefix 8 passed deterministically at every point. Perturbed-handoff trials stayed at 0–2/5 for prefix 7 (never above its 2/5 baseline) and 1–3/5 for prefix 8 (3/5 at 750, 1,000, 1,250 and 1,750 against a 2/5 baseline, a one-trial difference with no large-sample counterpart). All cold trials failed. Full budget completed (500 mixed critic updates, 2,000 mixed SAC updates, exit 0, 582.0 s wall).

## Actual execution

- Inherited SAC0001 final checkpoint (`initial.pt` SHA `e4a73d89…`); final counters BC 12,000 / pretraining 1,000 / online 4,000, optimizer steps critic 5,000 and actor/alpha 4,000, as prescribed; auxiliary generator states (prefix, demonstration, handoff-noise) saved in the final `state.json`. Differs from SAC0004 in `prefix_set {6,7,8}`, no MPC rows, `actor_bc_lambda 10` on the first row, `handoff_noise_sigma .025` on prefixes 7 and 8, checkpoints every 250; the seeds shared with SAC0004 (train 17491, demonstration 17492, reset 42, evaluation 42 / 175xx) match it, and the prefix (17493) and handoff-noise (17494) generators are new.
- Warm-up: 256 frozen-actor episodes (105 prefix 6, 71 prefix 7+noise, 80 prefix 8+noise), 734 suffix rows, 58 informational successes (0 / 14 / 44); every prefix-6 episode dropped after 2–5 controls.
- Demonstration buffer: 3 recorded rows (source controls 7–9); 77 demonstration and 179 online rows per batch in all 2,500 updates; BC on 1 row per update. After pretraining, demonstration critic bias −.008/+.080 (RMSE .087/.095), held-out prefix-6 RMSE .057/.050.
- Online: 2,000 interactions, 465 completed assisted episodes (167 / 145 / 153), 126 successes (22 / 24 / 80), 274 drops, 65 palm-support failures (episodes 36, 386, 423 also drop-flagged), no truncation, no handoff terminated before the policy. Prefix-6 successes per 500-update window 5/56, 3/43, 5/33, 9/35. Interaction 2,000 fell inside episode 465 (1 control, kept in `unfinished_suffix/`), so the final checkpoint holds an open episode; replay 2,734 rows. All 5,014 physical prefix controls passed the bit-exact assertions.
- Actor BC loss .0480 → 5.2e−4. Deterministic distance to recorded control 7 at the handoff .413 (baseline, after pretraining) → .022 (250) → .082 (500) → .016 (750) → .021 (1,000) → .027 (1,250) → .050 (1,500) → .085 (1,750) → .054 (2,000); latent std .067 → .076 → .083 → .099 → .120 → .142 → .145 → .155 → .140. Recorded-row critic bias swung between +.20/+.07 (250) and −.28/−.14 (500), ending +.06/+.08. Alpha .000305893 → .000378784 (raw-equivalent .0758).

## Frozen evaluations

| Checkpoint (online counter) | Prefix-6 det / stoch 17501–17505 | Prefix-7 det / stoch 17521–17525 / noisy 17541–17545 | Prefix-8 det / stoch 17531–17535 / noisy 17551–17555 | Cold det / stoch 17511–17515 |
| --- | --- | --- | --- | --- |
| pretraining_baseline (2000) | Drop after 3 (peak 5.34) / 0/5 (all drop at 3, peaks 5.32–5.35) | Pass 2 controls (6.66) / 5/5 (2 controls, 6.56–6.92) / 2/5 (17542, 17545 pass in 3) | Pass 1 (6.84) / 5/5 (1 control, 6.69–6.87) / 2/5 (17552, 17555 pass in 1) | Drop at 1 / not scheduled |
| online_250 (2250) | **Pass 3 controls (6.57)** / 1/5 (17503 pass in 4, 6.60; 17501 palm 4; others drop 4–5, peaks 5.47–6.28) | Pass 2 (6.65) / 2/5 (17521, 17524) / 1/5 (17545) | Pass 1 (6.90) / 2/5 (17531, 17532) / 1/5 (17555) | Drop at 3 (peak .86) / — |
| online_500 (2500) | Drop after 5 (peak 5.27) / 0/5 (drops at 4–5, peaks 5.27–5.79) | Pass 2 (6.84) / 5/5 / 1/5 (17542) | Pass 1 (6.53) / 4/5 (17535 drop 2) / 2/5 (17552, 17555) | Drop at 2 (.81) / — |
| online_750 (2750) | **Pass 3 controls (6.56)** / 0/5 (drops at 3–6, peaks 5.36–5.77) | Palm after 4 (peak 5.46) / 0/5 (17521 palm 5, others drop 4–15) / 0/5 | Pass 1 (6.55) / 5/5 / 3/5 (17552, 17554, 17555; 17553 environment success at 6.256 rad fails the strict criterion) | Drop at 2 (.55) / — |
| online_1000 (3000) | Drop after 11 (peak 5.42) / 1/5 (17501 pass in 5, 6.30; 17504 palm 6) | Pass 3 (6.57) / 1/5 (17521) / 1/5 (17542, 6 controls) | Pass 1 (6.48) / 3/5 (17533, 17535 drop 2) / 3/5 (17552, 17554 in 5, 17555) | Drop at 2 (.18) / — |
| online_1250 (3250) | **Pass 3 controls (6.43)** / 2/5 (17501 pass in 5, 6.77; 17504 pass in 3, 6.34) | Pass 2 (6.45) / 3/5 (17521, 17522, 17524) / 1/5 (17545) | Pass 1 (6.53) / 4/5 (17533 drop 2) / 3/5 (17552, 17553 in 3, 17555) | Drop at 1 (.22) / — |
| online_1500 (3500) | Drop after 6 (peak 5.69) / 0/5 (drops at 5–9, peaks 5.28–5.58) | Pass 2 (6.63) / 2/5 (17523, 17524 in 4) / 2/5 (17542 in 6, 17545 in 8) | Pass 1 (6.53) / 4/5 (17533 drop 3) / 2/5 (17552, 17555) | Drop at 1 (.21) / — |
| online_1750 (3750) | Drop after 4 (peak 5.35) / 0/5 (all drop at 4, peaks 5.28–6.49) | Pass 2 (6.46) / 5/5 / 1/5 (17542) | Pass 1 (6.51) / 4/5 (17533 drop 2) / 3/5 (17552, 17553 in 5, 17555) | Drop at 1 (.19) / — |
| online_2000 (4000) | Palm after 4 (peak 5.63, 19.2 N) / 1/5 (17504 pass in 3, 6.70; 17501, 17505 palm 4; 17502, 17503 drop 6–7, peaks 6.11–6.22) | Pass 2 (6.45) / 5/5 (17525 in 3) / 1/5 (17542) | Pass 1 (6.59) / 4/5 (17533 drop 2) / 2/5 (17552, 17555; 17553 palm 4) | Drop at 2 (.29) / 0/5 (all drop at 2, peaks .31–.35) |

Totals: 266 trials (29 per point × 8 + 34 at online_2000); 0 truncated, 0 non-bit-exact, 0 handoff terminated before the policy; `training_rng_restored` and `learner_tensors_and_counters_unchanged` true at every point; every per-trial `report.json` identical to its `summary.json` entry.

## Large-sample frozen evaluation (30 stochastic trials per prefix, fresh seeds)

Source: `frozen_checkpoint_evaluation_0001/results/bootstrap_variant_0015_bc_first_noise_mixed/<checkpoint>.json` (seeds 23000 + 100·prefix + i, one deterministic trial seed 42; prefixes 5, 6, 7, 8 and cold = prefix 0; no perturbed-handoff trials). Present for all nine checkpoints; `agent_sha256` matches `checkpoints/<checkpoint>/agent.pt`, the deterministic trials reproduce the in-stage results (online_2000 prefix 6: palm support after 4 controls, final force-qualified angle 4.906653716436524), 0 truncated.

| Checkpoint (online counter) | p5 det / k/30 | p6 det / k/30 (Wilson 95 %) | p7 det / k/30 | p8 det / k/30 | Cold det / k/30 |
| --- | --- | --- | --- | --- | --- |
| pretrained (2000) | F / 0 | F / 0 [.00, .11] | T / 30 | T / 30 | F / 0 |
| online_250 (2250) | F / 0 | T / 5 [.07, .34] | T / 13 | T / 15 | F / 0 |
| online_500 (2500) | F / 0 | F / 2 [.02, .21] | T / 28 | T / 30 | F / 0 |
| online_750 (2750) | F / 1 | T / 6 [.10, .37] | F / 15 | T / 28 | F / 0 |
| online_1000 (3000) | F / 1 | F / 8 [.14, .44] | T / 11 | T / 25 | F / 0 |
| online_1250 (3250) | F / 1 | T / 11 [.22, .54] | T / 18 | T / 27 | F / 0 |
| online_1500 (3500) | F / 1 | F / 2 [.02, .21] | T / 19 | T / 26 | F / 0 |
| online_1750 (3750) | F / 1 | F / 3 [.03, .26] | T / 25 | T / 26 | F / 0 |
| online_2000 (4000) | F / 3 | F / 4 [.05, .30] | T / 28 | T / 28 | F / 0 |

Prefix 6 never exceeded 11/30 (1,250); 4/30 at 2,000. Prefix 7 dipped to 11–15/30 at 750–1,000 and recovered to 28/30 at 2,000; prefix 8 stayed 25–30/30 after the 15/30 dip at 250. Prefix 5 (never trained) 0–1/30 through 1,750, 3/30 at 2,000. Cold 0/30 everywhere. Comparison rows read from the same directory (not audited beyond these files): 0004_recorded_bc10_first p6 12 / 16D / 20D of 30 at online 500 / 1,000 / 2,000 (p7 F/21 at 500); 0019_bc_first_mixed (same design, no noise) p6 7D, 14D, 9, 6, 10, 16D, 19D, 0 of 30 at 250…2,000; 0016_bc_first_noise_mixed_r2 (seed replicate) p6 0, 11D, 6, 2, 5, 10D, 5, 2 of 30 at 250…2,000.

## Interpretation and scope

The hypothesis (first-row BC plus noise keeps 0004's retention and 0007's 5/5 peak) is not supported at this seed. The handoff mean reached the label by 250 but wandered (L2 up to .085); the three deterministic passes all fell at checkpoints with L2 ≤ .027, but online 1,000 (L2 .021) dropped after 11 controls, so a small L2 was not sufficient. The stochastic ceiling was 2/5 against 0004's 3/5, and large-sample prefix 6 never exceeded 11/30 (0004: 20/30 at 2,000; 0019, the same design without noise, from its large-sample files only: 19/30 at 1,750, 0/30 at 2,000). Noise is not shown to widen the basin: perturbed prefix-7 training success was 24/145, prefix-7 noisy frozen trials never exceeded their 2/5 baseline, and prefix-8 noisy trials reached 3/5 at four checkpoints against a 2/5 baseline, a one-trial difference on five trials with no large-sample perturbed-handoff evaluation to resolve it; the wave-1 synthesis reached the same conclusion for 0007. The prefix-7 deterministic loss at 750 coincided with the 0/5 noisy trials and recovered with no setting change (0004 lost its prefix-7 deterministic trial once as well, at online 500, and had no 750 checkpoint).

Single seed, matched to SAC0004; the replicate is 0016. All 126 training successes are assisted, perturbed or demonstration-mixed, collected under a changing policy; none is primitive evidence and no trajectory was exported. Cold trials failed at every point (0/14 in-stage, 0/270 large-sample), so no checkpoint is deployment-eligible.


_Audit note: drafted by an auditing agent from the artifact files and independently refuted-checked; 4 claim(s) corrected by the verifier before writing._
