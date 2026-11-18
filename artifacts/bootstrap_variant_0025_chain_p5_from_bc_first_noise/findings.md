# Chained prefix-5 handoff with sigma .025 noise: prefix-5 deterministic pass only at online 1,000 (30-trial stochastic best 8/30 at 500), noisy prefix-6/7 trials 7/45 and 3/45, prefix-8 9/9 and 270/270, cold 0/14 and 0/270

**The handoff noise did not widen a basin; the stage reproduced the flicker of the inherited competence.** Prefix-5 deterministic passes only at online 1,000 (4 controls, 6.70 rad); prefix-5 stochastic 5/45 over the stage in the 5-trial evaluations (3/5 at 500, 1/5 at 750, 0/25 from 1,000) and 28/270 in the 30-trial evaluation (best 8/30 at 500, 5/30 at 1,000). Noisy-handoff trials (5 per cell, no large-sample counterpart): prefix 6 7/45 (4/5 at 2,000 only), prefix 7 3/45, prefix 8 21/45. Clean prefix-6 deterministic passes only at baseline and 1,000; prefix-7 deterministic fails at 250 (palm), 500 (drop), 750 (palm) and 1,250 (drop); prefix 8 9/9 deterministic, 45/45 and 270/270 stochastic. Cold 0/14 in the stage evaluations and 0/270 stochastic in the large-sample evaluation. Full budget, exit 0, 570.3 s.

## Actual execution

- Inherited `0004_recorded_bc10_first/checkpoints/online_2000` (SHA `7d539f09…`, counters 12,000/1,000/4,000, no open episode, 2,812 lineage rows, 0 imported). Final counters 12,000/1,500/6,000; optimizer steps critic 7,500, actor/alpha 6,000; `auxiliary_rng_states` (prefix, demonstration, handoff) present in the inherited state. Differs from 0024 by `handoff_noise_sigma .025` on prefixes 6-8 (handoff seed 17494, also present but unused in 0024) and the resulting noisy evaluation trials (`eval_noisy_prefixes`, seeds 17861-17865 / 17541-17545 / 17551-17555); train 17491, demonstration 17492, physical reset 42 and evaluation seeds match SAC0004; prefix seed 17493 is the variant family's, absent from SAC0004.
- Warm-up: 256 frozen-actor episodes (prefix 5: 82, 6+noise: 64, 7+noise: 58, 8+noise: 52), 1,139 rows, 38 informational successes (0/6/5/27). Perturbation L2 .049-.157 over 495 perturbed completed training episodes (174 warm-up, 321 online; the open episode 449 is also perturbed, L2 .086); no handoff terminated before the policy.
- Demonstration buffer: 4 recorded rows (source controls 6-9, raw return 259.35); 77 + 179 rows in all 2,500 updates; BC on 1 row (obs5 -> recorded control 6). After pretraining, demonstration critic bias -.596/-.588 -> +.049/+.167; held-out prefix-6 RMSE .169/.153 -> .095/.092.
- Online: 2,000 interactions, 449 completed assisted episodes (128/112/97/112), 114 successes (prefix 5: 19, 6+noise: 13, 7+noise: 18, 8+noise: 64; 6.29-7.33 rad, zero palm force), 301 drop-labelled failures, 33 palm-support failures (episode 317 carries both flags), 1 environment success excluded (episode 5, 6.234 rad), no truncation. First success episode 14 (8+noise, update 4,073). Prefix-5 successes per 500-update window 3/36, 7/36, 3/32, 6/24. Episode 449 open at interaction 2,000; replay 3,139 rows. All 4,510 prefix controls bit-exact.
- BC loss .0569 -> .000592; actor LR .000240 -> .0003 at cumulative 5,000. Deterministic L2 at obs5 to recorded control 6: .624 (baseline, after pretraining) -> .017 (250) -> .013-.047 (500-1,750) -> .072 (2,000); latent std .049 -> .083 -> .101 -> .126 -> .159 -> .223 -> .290 -> .280 -> .275. Demonstration critic bias at online checkpoints -.081 to +.228. Alpha .000377 -> .000456 (raw-equivalent .0912).

## Frozen evaluations (stage protocol: 1 deterministic and 5 stochastic trials per cell; noisy = perturbed final prefix control, L2 .069-.126)

| Point (online counter) | Prefix 5 det / stoch 17611-17615 | Prefix 6 det / stoch 17501-17505 / noisy 17861-17865 | Prefix 7 det / stoch 17521-17525 / noisy 17541-17545 | Prefix 8 det / stoch 17531-17535 / noisy 17551-17555 | Cold det / stoch 17511-17515 |
| --- | --- | --- | --- | --- | --- |
| pretraining_baseline (4,000) | drop after 6 (max 5.03) / 0/5 | pass 3 controls (6.70, 4 fingers) / 3/5 / 0/5 | pass 2 (6.48) / 5/5 / 0/5 | pass 1 (6.62) / 5/5 / 2/5 | drop after 3 (max 1.08) / not scheduled |
| online_250 (4,250) | palm after 6 (max 5.39) / 1/5 | drop after 5 (max 5.81) / 1/5 / 0/5 | palm after 5 (max 5.46) / 0/5 / 0/5 | pass 1 (6.52) / 5/5 / 1/5 | drop after 2 / not scheduled |
| online_500 (4,500) | drop after 5 (max 5.27) / 3/5 (17611, 17612, 17615) | palm after 4 (max 5.48) / 0/5 / 1/5 | drop after 8 (max 6.55) / 2/5 / 1/5 | pass 1 (6.67) / 5/5 / 2/5 | drop after 2 / not scheduled |
| online_750 (4,750) | drop after 9 (max 5.36) / 1/5 | drop after 6 (max 5.46) / 0/5 / 0/5 | palm after 3 (max 5.49) / 1/5 / 0/5 | pass 1 (6.59, 3 fingers) / 5/5 / 3/5 | drop after 2 / not scheduled |
| online_1000 (5,000) | pass 4 controls (6.70, 3 fingers) / 0/5 | pass 3 controls (6.43, 4 fingers) / 3/5 / 0/5 | pass 2 (6.47) / 5/5 / 1/5 | pass 1 (6.61) / 5/5 / 3/5 | drop after 2 / not scheduled |
| online_1250 (5,250) | drop after 5 (force-qualified angle reached 7.36 then dropped, 0 fingers) / 0/5 | drop after 6 (max 5.77) / 3/5 / 0/5 | drop after 4 (max 6.75) / 2/5 / 0/5 | pass 1 (6.60) / 5/5 / 4/5 | drop after 2 / not scheduled |
| online_1500 (5,500) | drop after 5 (max 5.31) / 0/5 | drop after 7 (max 5.42) / 1/5 / 1/5 | pass 2 (6.60) / 4/5 / 0/5 | pass 1 (6.60) / 5/5 / 2/5 | drop after 2 / not scheduled |
| online_1750 (5,750) | drop after 6 (max 6.36) / 0/5 | drop after 7 (max 6.05) / 2/5 / 1/5 | pass 2 (6.48) / 4/5 / 0/5 | pass 1 (6.80) / 5/5 / 2/5 | drop after 2 / not scheduled |
| online_2000 (6,000) | drop after 5 (max 5.17) / 0/5 | drop after 5 (max 5.83) / 1/5 / 4/5 | pass 2 (6.62) / 5/5 / 1/5 | pass 1 (6.80, 3 fingers) / 5/5 / 2/5 | drop after 2 (max 0.50) / 0/5 (drops after 2-3, peaks 0.87-1.02) |

Totals over the stage: 365 trials (8 x 40 + 45); prefix-5 stochastic 5/45; prefix-6 stochastic 14/45, noisy 7/45; prefix-7 stochastic 28/45, noisy 3/45; prefix-8 stochastic 45/45, noisy 21/45; cold 0/14. Deterministic passes: prefix 5 at 1,000 only; prefix 6 at baseline and 1,000; prefix 7 at baseline, 1,000, 1,500, 1,750, 2,000; prefix 8 at all 9 points. All trials terminated (0 truncated), prefix physics bit-exact in all 365, no handoff terminated before the policy, training RNG restored and learner tensors and counters unchanged at every point, per-trial report.json identical to the summary.json entries (0 mismatches).

## Large-sample evaluation (frozen_checkpoint_evaluation_0001, 30 stochastic trials per prefix per checkpoint, seeds 23000 + 100 x prefix + i, plus the seed-42 deterministic trial)

`results/bootstrap_variant_0025_chain_p5_from_bc_first_noise/` holds all nine checkpoints; every `agent_sha256` equals the artifact's `manifest.json` hash for that checkpoint, and the deterministic trial reproduces the stage's own deterministic trial in all 45 cells (same seed, same outcome, same angle and control count), so deterministic passes below are the same events as in the table above while the stochastic samples are independent of the 17xxx seeds. Clean prefixes only; noisy-handoff cells have no large-sample counterpart. Stochastic successes / 30 (D = deterministic pass; Wilson 95 % in brackets where cited):

| Checkpoint | Prefix 5 | Prefix 6 | Prefix 7 | Prefix 8 | Cold |
| --- | --- | --- | --- | --- | --- |
| pretrained | 0 | 20 D [.49, .81] | 26 D [.70, .95] | 30 D | 0 |
| online_250 | 1 | 2 | 9 | 30 D | 0 |
| online_500 | 8 [.14, .44] | 6 | 4 [.05, .30] | 30 D | 0 |
| online_750 | 3 | 5 | 4 | 30 D | 0 |
| online_1000 | 5 D [.07, .34] | 13 D [.27, .61] | 27 D [.74, .97] | 30 D | 0 |
| online_1250 | 1 | 7 | 16 | 30 D | 0 |
| online_1500 | 1 | 8 | 18 D | 30 D | 0 |
| online_1750 | 5 | 7 | 19 D | 30 D | 0 |
| online_2000 | 4 [.05, .30] | 14 [.30, .64] | 24 D [.63, .90] | 30 D | 0 |

Aggregates over the nine checkpoints: prefix 5 28/270, prefix 6 82/270, prefix 7 147/270, prefix 8 270/270, cold 0/270 (cold deterministic 0/9, identical to the stage trials). No trial truncated; per-trial physics and transitions are not stored. `summary.md` in that directory was regenerated before these files were written and lists no 0025 rows; the result JSONs and `run.log` do.

## Interpretation and scope

The hypothesis is not supported at this seed. The baseline noisy trials (prefix 6 0/5, prefix 7 0/5, against clean 3/5 and 5/5; clean 30-trial 20/30 and 26/30) are consistent with inherited prefix-6/7 basins narrower than sigma .025, on 5 noisy trials per cell; after 2,000 updates perturbed prefix 7 is still 1/5 while the perturbed prefix-6 4/5 at 2,000 is a single-checkpoint peak whose clean deterministic trial drops (clean stochastic 1/5, 14/30 there). Clean prefix-6 competence inherited at 20/30 fell to 2/30 by online 250 and never exceeded 14/30 afterwards; prefix 7 fell from 26/30 to 4/30 at 500-750 and recovered to 24/30 with a deterministic pass at 2,000; prefix 8 held 30/30 throughout. The prefix-5 handoff behaves like the wave-1 prefix-6 handoff: the mean sits within L2 .01-.07 of the label from 250 on, yet the deterministic pass is a single checkpoint (1,000, where stochastic is 5/30) and the stochastic best is 8/30 at 500 (5-trial 3/5), while the latent std grows .049 -> .28, the unresolved excursion the synthesis lists. Sibling 0024 (recounted from its evaluation and large-sample reports, not audited here) also passes prefix 5 deterministically only at 1,000 (5-trial best 1/5; 30-trial 21/270 over the stage vs 28/270 here) and shares the prefix-6 collapse (70/270 vs 82/270), but the prefix-7 trajectories differ beyond the synthesis's one-pass/two-trial seed spread at single checkpoints in both directions: 0024 loses prefix 7 at 2,000 (deterministic drop, 1/5, 13/30) where this run keeps it (pass, 5/5, 24/30), while 0024 holds 13/30 and 22/30 at 500 and 750 where this run has 4/30 and 4/30 (stage totals 186/270 vs 147/270). The two stages differ by a design factor, not by seed, and neither direction is supported by more than one seed. All 114 training successes are assisted, demonstration-mixed, changing-policy; none is primitive evidence. Cold 0/14 in the stage (drops after 2-3 controls, peak <= 1.08 rad) and 0/270 stochastic in the large-sample evaluation: no checkpoint is deployment-eligible. Single seed.

Inputs: `protocol.json`, `execution_manifest.json`, `inherited_state.json`; results: `summary.json`, `metrics.jsonl`, `episodes/`, `evaluations/`, `checkpoints/`, `unfinished_suffix/`, `demonstration_calibration_*.json`, `heldout_calibration_*`; hashes: `manifest.json`; large-sample: `frozen_checkpoint_evaluation_0001/results/bootstrap_variant_0025_chain_p5_from_bc_first_noise/*.json`.


_Audit note: drafted by an auditing agent from the artifact files and independently refuted-checked; 7 claim(s) corrected by the verifier before writing._
