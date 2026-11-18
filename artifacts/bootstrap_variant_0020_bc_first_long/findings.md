# First-row BC with 4,000 online updates: deterministic prefix-6 pass survives at 3,000–4,000, stochastic prefix-6 rate decays after 2,000, cold still fails

**Doubling the budget did not stabilize the stochastic result.** The first 2,000 online updates reproduce `bootstrap_variant_0004_recorded_bc10_first` bit-for-bit (all 256 warm-up and 432 online episode reports identical, all 76 shared evaluation trials identical, `pretrained` / `online_500` / `online_1000` / `online_2000` `agent.pt` SHAs equal, online_2000 `7d539f09…`), so the second 2,000 updates continue the wave-1 best checkpoint. The frozen deterministic prefix-6 trial passes at online 3,000, 3,500 and 4,000 (drop at 2,500), but the five-seed stochastic count falls from 3/5 (1,000, 2,000) to 0/5, 2/5, 1/5, 2/5, and the 30-trial rate from 20/30 at 2,000 to 6/30 at 4,000. Prefix-8 retention is 6/6 at every point; prefix-7 stochastic dips to 1/5 at 1,500 and 2/5 at 3,500. Every cold trial fails. Exit 0, 716.0 s wall.

## Actual execution

- Inherited SAC0001 (`e4a73d89…`); differs from SAC0004 in `demonstration_include_mpc false`, `actor_bc_lambda 10`, `actor_bc_rows first`, `online_policy_interactions 4000`, checkpoints every 500; seeds match SAC0004 (train 17491, demonstration 17492, physical reset 42, evaluation seeds identical). Final counters BC 12,000 / pretraining 1,000 / online 6,000; optimizer steps critic 7,000, actor/alpha 6,000; `auxiliary_rng_states` present in every checkpoint state.json.
- Warm-up: 256 prefix-6 episodes, 812 rows, all drops (13 after 2 controls, 188 after 3, 53 after 4, 2 after 5), 0 successes (informational).
- metrics.jsonl: 500 `critic_pretrain_update`, 4,000 `online_update`, 1,109 `episode_complete`; every batch 77 demonstration + 179 online rows, BC term on 1 row. Actor BC loss .0480 → 4.0e−5; actor LR .00012 → .0003; learned alpha .000306 → .000459 (raw-equivalent .0612 → .0918).
- Online: 853 completed assisted prefix-6 episodes, 191 force-qualified successes (171 in 3 controls, 14 in 4, 4 in 5, 1 in 6, 1 in 8; final 6.32–7.21 rad; 112/66/13 with four/three/two force-bearing fingers), 533 drop-labelled failures, 128 palm-support failures (episodes 19 and 61 carry both flags), 1 environment success below the strict criterion (episode 416, 6.270 rad), none truncated. Successes per 500-update window: 8/95, 47/124, 34/116, 20/97, 32/122, 10/99, 19/98, 21/102 (first half 109/432, second 82/421). Episode 853 open at the budget (3 controls); replay 4,812 rows = 812 + 3,997 + 3. All 6,660 prefix controls (1,110 × 6) bit-exact, never in replay; `handoff_terminated_before_policy` 0.
- Handoff calibration (mean at obs6 vs recorded control 7): L2 .413 → .023 (500), .019, .039 (1,500), .021, .011, .008 (3,000), .018, .014 (4,000); latent std .067 → .097, .127, .145, .133 (2,000), .141, .173 (3,000), .153, .140 (4,000). Recorded-row critic bias −.0003/+.008 after pretraining, then −.30/−.32 (500), −.14/−.09 (1,000), −.31/−.21 (1,500), −.20/−.26 (2,000), −.03/−.13 (2,500), −.24/−.21 (3,000), −.18/−.24 (3,500), −.19/−.28 (4,000).

## Frozen evaluations

| Point (online counter) | P6 det (seed 42) | P6 stoch 17501–17505 | P7 det / stoch 17521–17525 | P8 det / stoch 17531–17535 | Cold det | Cold stoch 17511–17515 |
| --- | --- | --- | --- | --- | --- | --- |
| pretraining_baseline (2000) | Drop after 3 (peak 5.34) | 0/5 (all drops after 3) | Pass (2 controls, 6.66) / 5/5 | Pass (1 control, 6.84) / 5/5 | Drop at 1 | not scheduled |
| online_500 (2500) | Drop after 5 (peak 6.27) | 1/5 (17505) | Drop after 3 (peak 6.10) / 4/5 (17522 drop) | Pass / 5/5 | Drop at 2 | not scheduled |
| online_1000 (3000) | Pass: 3 controls, 6.62 rad | 3/5 (17502, 17504, 17505) | Pass / 5/5 | Pass / 5/5 | Drop at 2 | not scheduled |
| online_1500 (3500) | Palm support after 4 (peak 5.40, 2.6 N) | 1/5 (17501; 4 drops) | Pass (2, 6.54) / 1/5 (17524 in 4; 17521 drop at 7; 17522, 17523, 17525 palm) | Pass / 5/5 | Drop at 3 | not scheduled |
| online_2000 (4000) | Pass: 3 controls, 6.70 rad | 3/5 (17501, 17504, 17505) | Pass / 5/5 | Pass / 5/5 | Drop at 3 (peak 1.08) | not scheduled |
| online_2500 (4500) | Drop after 5 (peak 5.77) | 0/5 (4 drops; 17505 palm after 4, peak 6.22) | Pass / 4/5 (17524 drop at 4, peak 6.59) | Pass / 5/5 | Drop at 2 | not scheduled |
| online_3000 (5000) | Pass: 3 controls, 6.68 rad | 2/5 (17504 in 3; 17505 in 7) | Pass / 3/5 (17523 palm, 17525 drop) | Pass / 5/5 | Drop at 1 | not scheduled |
| online_3500 (5500) | Pass: 3 controls, 6.70 rad | 1/5 (17501; 17504 palm after 6; 17505 drop after 8) | Pass / 2/5 (17521, 17523 drop; 17525 palm) | Pass / 5/5 | Drop at 1 | not scheduled |
| online_4000 (6000) | Pass: 3 controls, 6.57 rad, 3 fingers | 2/5 (17501, 17504) | Pass (3, 6.34) / 4/5 (17521 drop at 6) | Pass / 5/5 | Drop at 2 | 0/5 (drops after 1–2, peaks .21–.39) |

176 trials (19 per point, 24 at online_4000), none truncated, prefix physics bit-exact, `handoff_terminated_before_policy` 0, training RNG restored and learner tensors and counters unchanged after every point, per-trial report.json identical to summary.json reports. No noisy-handoff trials scheduled (`handoff_noise_sigma` 0). Prefix-6 stochastic over the eight online points: 13/40 (first four 8/20, last four 5/20); prefix-7 stochastic 28/40; prefix-8 6/6 at all nine points; cold 0/14.

## Large-sample evaluation

`frozen_checkpoint_evaluation_0001/results/bootstrap_variant_0020_bc_first_long/` covers all nine checkpoints (30 stochastic trials per prefix, seeds 23601–23630 / 23701–23730 / cold 23001–23030; deterministic seed 42 reproduces the stage's deterministic outcome, control count and final angle at all nine points). Prefix-6: 0/30 pretrained (0–11 %), 12/30 (40 %, 25–58 %), 16/30 (53 %, 36–70 %), 4/30 (13 %, 5–30 %), 20/30 at 2,000 (67 %, Wilson 49–81 %), 14/30 (47 %, 30–64 %), 7/30 (23 %, 12–41 %), 3/30 (10 %, 3–26 %), 6/30 at 4,000 (20 %, 10–37 %). Prefix-6 stochastic failures are mostly drops; palm-support failures 0, 2, 3, 9, 0, 3, 3, 6, 2 per checkpoint. Prefix-7: 30, 21, 26, 13, 26, 22, 22, 17, 21. Prefix-8 30/30 and prefix-5 0/30 everywhere; cold 0/30 at every checkpoint (0/270 stochastic, 0/9 deterministic). The pretrained/500/1,000/2,000 files equal 0004's (same agent SHAs, identical prefix records; 0004 has no online_1500 file).

## Interpretation and scope

The hypothesis asked whether the prefix-6 result stabilizes or drifts: it drifts. The deterministic trial passes at three of the four late points, but the 2,000 and 3,500–4,000 stochastic rates (20/30 vs 3/30, 6/30) have non-overlapping 30-trial intervals, and training successes fell to 10/99 in updates 2,501–3,000. The handoff mean stayed within .008–.023 of recorded control 7 at every online checkpoint except 1,500 (.039), so the decay is not first-control mean error. The handoff latent std rose from .133 (2,000) to .173 (3,000) but was back to .140 at 4,000, so it does not track the late decay either; alpha rose monotonically (up 50 % over the stage) and palm-support failures recur at every late point, consistent with the wave-1 rising-temperature and narrow-basin open questions but not resolved here. Prefix-7 also eroded (26/30 → 17–22/30) while prefix-8 held. The best checkpoint of this run remains online_2000, i.e. 0004's.

Single seed matched to 0004. All 191 training successes are assisted, demonstration-mixed, changing-policy episodes; none is primitive evidence; no trajectory exported. Cold 0/14 in the stage and 0/270 in the large sample: no checkpoint is deployment-eligible.

Inputs and results: `protocol.json`, `summary.json`, `warmup_record.json`, `metrics.jsonl`, `episodes/`, `evaluations/`, `checkpoints/`, `unfinished_suffix/`, `demonstration_calibration_*.json`, `manifest.json`.


_Audit note: drafted by an auditing agent from the artifact files and independently refuted-checked; 6 claim(s) corrected by the verifier before writing._
