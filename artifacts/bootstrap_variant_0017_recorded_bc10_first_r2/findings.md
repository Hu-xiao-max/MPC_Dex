# Seed replicate of first-row BC: prefix-6 deterministic passes at online 500, 750, 1,500, 1,750 (fails at 250, 1,000, 1,250, 2,000), stochastic peak 4/5 at 1,750 and 3/5 at 2,000, large-sample prefix-6 13/30 at 1,500 and 5/30 at 2,000, cold 0/14

**The fresh-seed replicate reproduces the outcome of `recorded_bc10_first` in kind but not per checkpoint:** the frozen policy completes the turn from the prefix-6 handoff deterministically at online 500, 750, 1,500 and 1,750 (3 controls, 6.40–6.77 rad) and drops or palm-fails at 250, 1,000, 1,250 and 2,000 (0004: passes at 1,000 and 2,000); stochastic prefix-6 success reaches 4/5 at 1,750 and 3/5 at 1,500 and 2,000 (0004: 3/5 at 1,000 and 2,000); prefix-7 and prefix-8 retention are 6/6 each at 2,000 after a transient loss of prefix 7 at 250 (0/6). 90 of 464 completed online training episodes succeeded (0004: 109/432). All cold trials fail. Full budget completed (500 mixed critic updates, 2,000 mixed SAC updates, exit 0, 472.5 s wall).

## Actual execution

- Inherited SAC0001 final checkpoint (SHA `e4a73d89…`); final counters BC 12,000 / pretraining 1,000 / online 4,000, optimizer steps critic 5,000 and actor/alpha 4,000, as prescribed; auxiliary RNG states (prefix, demonstration, handoff) saved in every checkpoint. Differs from SAC0004 in `demonstration_include_mpc false`, `actor_bc_lambda 10`, `actor_bc_rows first`, fresh training seeds (train 19170, demonstration 19171, prefix 19172, handoff-noise 19173 unused; SAC0004: 17491 / 17492, variant 0004 prefix 17493), fresh evaluation seeds (21761–21795; 0004 used 17501–17535), checkpoints and evaluation points every 250 updates (SAC0004: 500, 1,000, 2,000). The physical reset seed 42 and the deterministic evaluation seed 42 are the same as in SAC0004 and 0004.
- Warm-up: 256 frozen-actor prefix-6 episodes, 821 suffix rows, all drops after 2–7 controls (10 at 2, 190 at 3, 52 at 4, 2 at 5, 1 at 6, 1 at 7), zero successes (informational).
- Demonstration buffer: 3 recorded rows (source controls 7–9, one terminal episode, raw return 226.97); 77 demonstration and 179 online rows per batch in all 500 pretraining and 2,000 online updates; the BC term covered exactly 1 row per update. After pretraining, demonstration critic RMSE .019/.035 (from .878/.933; bias +.013/+.027), warm-up on-policy RMSE .079/.078 (from .320/.340), held-out .072/.080 (from .340/.352).
- Online: 2,000 interactions, 464 completed assisted episodes, 90 force-qualified successes (79 in 3 controls, 2 in 4, 9 in 5; final 6.34–6.91 rad, zero palm force, 48 with four force-bearing fingers, 37 with three, 5 with two), 285 drops, 89 palm-support failures (no episode carries both flags), no truncation. First success at episode 21 (5 controls, 6.64 rad, cumulative update 2,117). Successes per 500-update window by the update at which each episode closed: 13/100, 24/117, 31/126, 22/121. The 2,000th interaction fell inside episode 464 (5 controls, 5.18 rad, open, preserved in `unfinished_suffix/` and `checkpoints/online_2000/current_episode/`); replay holds 2,821 rows. All 4,326 physical prefix controls (721 × 6) passed the bit-exact replay assertions and never entered replay; no handoff terminated before the policy.
- Actor BC loss .0480 → 8.1e−5. Deterministic distance to recorded control 7 at the handoff .413 (baseline and after pretraining) → .044 (250) → .022 (500) → .024 (750) → .026 (1,000) → .047 (1,250) → .013 (1,500) → .037 (1,750) → .026 (2,000); latent std there .067 → .073 → .095 → .116 → .138 → .153 → .134 → .138 → .163. Demonstration critic bias re-drifted to −.15/−.20 at 500, −.18/−.12 at 1,000, −.13/−.17 at 2,000. Learned alpha .000305860 → .000377327 (raw-equivalent .0755); actor LR ramp .000120 → .000240.

## Frozen evaluations

| Checkpoint (online_updates counter) | Prefix-6 deterministic (seed 42) | Prefix-6 stochastic 21761–21765 | Prefix-7 deterministic / stochastic 21771–21775 | Prefix-8 deterministic / stochastic 21781–21785 | Cold deterministic | Cold stochastic 21791–21795 |
| --- | --- | --- | --- | --- | --- | --- |
| pretraining_baseline (2000) | Drop after 3 controls (peak 5.34 rad) | 0/5, all drops after 3–4 controls (peaks 5.29–5.37) | Pass (2 controls, 6.66) / 5/5 (2 controls, 6.52–6.68) | Pass (1 control, 6.84) / 5/5 (6.55–6.75) | Drop at control 1 | not scheduled |
| online_250 (2250) | Drop after 7 (peak 5.57) | 0/5 (21761/21765 drop at 8, 21763 drop at 7, 21762 drop at 7 peak 6.26, 21764 palm at 6) | Drop after 7 (peak 5.68) / 0/5 (21771 drop at 9, 21773 drop at 5, 21772/21774/21775 palm at 3–5) | Pass (1, 6.52) / 5/5 | Drop at control 2 | not scheduled |
| online_500 (2500) | Pass: 3 controls, 6.40 rad, 3 fingers, 0 palm | 2/5 (21763 6.36, 21765 6.51, both 3 controls; 21761 drop at 6, 21762 palm at 4, 21764 drop at 4) | Pass (2, 6.37) / 4/5 (21772 drop at 4) | Pass (1, 6.55) / 5/5 | Palm support at control 2 | not scheduled |
| online_750 (2750) | Pass: 3 controls, 6.77 rad, 3 fingers | 1/5 (21763 6.81; 21761 palm at 5, 21762 drop at 7, 21764 drop at 4, 21765 drop at 5) | Pass (2, 6.59) / 5/5 | Pass (1, 6.46) / 5/5 | Palm support at control 2 | not scheduled |
| online_1000 (3000) | Drop after 4 (peak 5.51) | 2/5 (21762 6.56, 21765 6.66; 21761/21763 palm at 4, 21764 drop at 6) | Pass (2, 6.59) / 5/5 | Pass (1, 6.59) / 5/5 | Palm support at control 2 | not scheduled |
| online_1250 (3250) | Drop after 4 (peak 5.33) | 1/5 (21761 6.64, 2 fingers; 21762/21765 palm at 4, 21763/21764 drop at 4) | Pass (2, 6.73) / 3/5 (21771 palm at 3, 21775 drop at 5) | Pass (1, 6.63) / 5/5 | Palm support at control 2 | not scheduled |
| online_1500 (3500) | Pass: 3 controls, 6.64 rad, 3 fingers | 3/5 (21762 6.66, 21763 6.68, 21765 6.53; 21761 palm at 4, 21764 drop at 4) | Pass (2, 6.64) / 5/5 | Pass (1, 6.54) / 5/5 | Drop at control 3 (peak 1.06) | not scheduled |
| online_1750 (3750) | Pass: 3 controls, 6.62 rad, 4 fingers | 4/5 (21761 6.43, 21762 6.48, 21763 6.58, 21765 6.61; 21764 drop at 4 peak 5.87) | Drop after 3 (peak 6.68) / 5/5 | Pass (1, 6.45) / 5/5 | Drop at control 2 | not scheduled |
| online_2000 (4000) | Palm support after 4 (peak 5.43, palm 11.4 N, 2 fingers) | 3/5 (21761 6.62, 21762 6.57, 21765 6.44; 21763 palm at 4, 21764 drop at 4) | Pass (2, 6.58) / 5/5 | Pass (1, 6.58) / 5/5 | Drop at control 3 (peak .95) | 0/5, drops after 2–3 controls (peaks .59–1.43) |

No noisy-handoff trials were scheduled (handoff_noise_sigma 0). 176 trials (8 × 19 + 24), all terminated, none truncated, prefix physics bit-exact, no handoff terminated before the policy, training RNG restored and learner tensors and counters unchanged at every point, per-trial report.json files match summary.json. Every prefix-6 success uses exactly 3 policy controls.

## Large-sample frozen evaluation

`frozen_checkpoint_evaluation_0001/results/bootstrap_variant_0017_recorded_bc10_first_r2/` (30 stochastic trials per prefix, seeds cold 23001–23030, p5 23501–23530, p6 23601–23630, p7 23701–23730, p8 23801–23830, disjoint from the stage's stochastic seeds; deterministic seed 42 agrees with the stage at all 36 cells the stage evaluated (prefixes 6, 7, 8 and cold × 9 checkpoints; prefix 5 has no stage counterpart); agent SHAs match the checkpoints; Wilson 95 % in brackets):

| Checkpoint | Cold det / stoch | Prefix-5 det / stoch | Prefix-6 det / stoch | Prefix-7 det / stoch | Prefix-8 det / stoch |
| --- | --- | --- | --- | --- | --- |
| pretrained (= pretraining_baseline, SHA c34584a7…) | fail / 0/30 | fail / 0/30 | fail / 0/30 [0, 11 %] | pass / 30/30 | pass / 30/30 |
| online_250 | fail / 0/30 | fail / 0/30 | fail / 5/30 (17 % [7, 34]; successes in 5–7 controls) | fail / 1/30 (3 % [1, 17]) | pass / 30/30 |
| online_500 | fail / 0/30 | fail / 0/30 | pass / 11/30 (37 % [22, 54]) | pass / 28/30 (93 %) | pass / 30/30 |
| online_750 | fail / 0/30 | fail / 0/30 | pass / 8/30 (27 % [14, 44]) | pass / 24/30 (80 %) | pass / 30/30 |
| online_1000 | fail / 0/30 | fail / 0/30 | fail / 6/30 (20 % [10, 37]) | pass / 27/30 (90 %) | pass / 30/30 |
| online_1250 | fail / 0/30 | fail / 0/30 | fail / 2/30 (7 % [2, 21]) | pass / 14/30 (47 % [30, 64]) | pass / 30/30 |
| online_1500 | fail / 0/30 | fail / 0/30 | pass / 13/30 (43 % [27, 61]) | pass / 22/30 (73 %) | pass / 30/30 |
| online_1750 | fail / 0/30 | fail / 0/30 | pass / 8/30 (27 % [14, 44]) | fail / 28/30 (93 %) | pass / 30/30 |
| online_2000 | fail / 0/30 | fail / 0/30 | fail / 5/30 (17 % [7, 34]) | pass / 21/30 (70 % [52, 83]) | pass / 30/30 |

All large-sample prefix-6 successes from online_500 on use 3 controls; failures are drops and palm-support violations (e.g. 15 drop / 10 palm at 2,000). No trial truncated. Cold and prefix 5: 0/270 each over the nine checkpoints. Every checkpoint of this variant has a large-sample result.

## Interpretation and scope

The design-level result of 0004 replicates (handoff mean within .01–.05 of recorded control 7 from 250 on, prefix-6 deterministic passes at half the checkpoints, prefix-7/8 6/6 at the end, no hold-to-horizon trials), consistent with the wave-1 attribution of first-row BC. The checkpoint-level claims do not: neither 0004's 1,000 nor 2,000 pass recurs here, and the 30-trial rates show the prefix-6 basin oscillating between 7 % and 43 % across 250-update intervals while the handoff latent std rises .067 → .163. The 4/5 at 1,750 corresponds to 8/30 in the large sample, so the five-trial peaks overstate the rate. The transient prefix-7 loss was already visible in 0004 at online 500 (deterministic drop, 4/5 stochastic, 21/30 large-sample) with recovery by 1,000; the dense schedule here adds the 250-update point, where it is deeper (deterministic drop, 0/5 and 1/30) and already recovered by 500 (28/30).

Single seed, not matched to 0004. All 90 training successes are assisted (six replayed controls) and demonstration-mixed, collected under a changing policy; none is primitive evidence and no trajectory was exported. Cold trials 0/14 in the stage and 0/270 in the large sample, so no checkpoint is deployment-eligible; the online_2000 checkpoint carries an open episode. Frozen success claims apply only to the checkpoints and seeds above.

Inputs and protocol: `protocol.json`, `execution_manifest.json`, `input_bindings.json`; results: `summary.json`, `warmup_record.json`, `metrics.jsonl`, `episodes/`, `evaluations/`, `checkpoints/`, `demonstration_buffer.json`, `demonstrations_replay.npz`, `demonstration_calibration_*.json`, `warmup_calibration_*`, `heldout_calibration_*`, `unfinished_suffix/`; final hashes: `manifest.json`; large sample: `../frozen_checkpoint_evaluation_0001/results/bootstrap_variant_0017_recorded_bc10_first_r2/`.

_Audit note: drafted by an auditing agent from the artifact files and independently refuted-checked; 4 claim(s) corrected by the verifier before writing (physical reset seed 42 is shared with SAC0004, not fresh; deterministic agreement holds at 36 cells, not 27; warm-up drop bins completed; prefix-7 transient qualified against 0004's online-500 dip)._


_Audit note: drafted by an auditing agent from the artifact files and independently refuted-checked; 4 claim(s) corrected by the verifier before writing._
