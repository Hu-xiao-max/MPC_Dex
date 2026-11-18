# dart_recipe replicate: sporadic prefix-6 stochastic successes (peak 3/5 at online 1,000), deterministic prefix-6 never passes in this seed, cold 0/6

**The second seed of the dart recipe (BC lambda 10 to 3 recorded rows, sigma .05 handoff noise on prefixes 7/8, mixed prefixes 6/7/8, 5% online demonstration share) again yields only sporadic frozen prefix-6 successes, not a solved handoff:** frozen prefix-6 stochastic trials passed 1/5, 3/5 and 1/5 at online 500/1,000/2,000, the prefix-6 deterministic trial dropped at every point, and all six final cold trials dropped at control 2. Prefix-7/8 deterministic retention held throughout; their stochastic retention fell to 1/5 and 3/5 at online 500 and ended 3/5 and 5/5. The stage completed its full budget (exit 0, 595.1 s).

## Actual execution

- Inherited SAC0001 final checkpoint (SHA `e4a73d89…`), new seeds 19090–19093 (train / demonstration / prefix / handoff noise); final counters BC 12,000 / pretraining 1,000 / online 4,000, optimizer steps critic 5,000, actor/alpha 4,000; 500 `critic_pretrain_update` and 2,000 `online_update` events, counters monotone (501–1,000 and 2,001–4,000).
- Warm-up: 256 frozen-actor episodes, 799 suffix rows; prefix 6 0/94, 7+noise 10/80, 8+noise 24/82 (34 informational successes; 193 drops, 29 palm-support failures).
- Demonstrations: 3 recorded rows (controls 7–9), no MPC; 77 per pretraining batch, 13 per online batch. Pretraining moved demonstration-row critic bias from −.677/−.619 to −.032/−.033 and held-out prefix-6 RMSE from .296/.324 to .042/.047; online, the demonstration bias drifted back to −.620/−.620, −.746/−.715 and −.294/−.301 at 500/1,000/2,000.
- Online: 2,000 interactions, 548 completed changing-policy assisted episodes, 107 successes (prefix 6 32/182, 7+noise 17/169, 8+noise 58/197); by failure reason 380 drops and 61 palm-support failures (384 episodes carry the drop flag, four of them classified palm-support), 0 truncations, 0 handoffs terminated before the policy. All 5,631 physical prefix controls (1,780 warm-up, 3,851 online) replayed bit-exactly and never entered replay; replay holds 2,799 rows (799 + 2,000); no unfinished suffix.
- Actor BC loss .110 → .00062; deterministic L2 distance to recorded control 7 at the prefix-6 handoff .413 (baseline and after pretraining) → .025 → .016 → .032 at online 500/1,000/2,000; latent std there .067 → .097 → .161 → .163. Alpha .000305893 → .000366337 (raw-equivalent .0733).

## Frozen evaluations (finger-supported successes)

| Point | P6 det. / stoch. | P7 det. / stoch. | P7 noisy | P8 det. / stoch. | P8 noisy | Cold det. / stoch. |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | Drop, 3 / 0/5 | Pass / 5/5 | 0/5 | Pass / 5/5 | 1/5 | Drop, 1 / not scheduled |
| Online 500 | Drop, 6 / 1/5 | Pass / 1/5 | 0/5 | Pass / 3/5 | 0/5 | Drop, 2 / not scheduled |
| Online 1,000 | Drop, 6 / 3/5 | Pass / 1/5 | 1/5 | Pass / 3/5 | 0/5 | Drop, 2 / not scheduled |
| Online 2,000 | Drop, 5 / 1/5 | Pass / 3/5 | 0/5 | Pass / 5/5 | 1/5 | Drop, 2 / 0/5 |

Full cell contents (seeds: prefix 6 20961–20965, prefix 7 20971–20975, prefix 7 noisy 21971–21975, prefix 8 20981–20985, prefix 8 noisy 21981–21985, cold 20991–20995):

| Point (online_updates counter) | Prefix 6 det. / stoch. | Prefix 7 det. / stoch. | Prefix 7 noisy | Prefix 8 det. / stoch. | Prefix 8 noisy | Cold det. / cold stoch. |
| --- | --- | --- | --- | --- | --- | --- |
| pretraining_baseline (2000) | Drop after 3 controls, peak 5.338 rad / 0/5 (5 drops, 3 controls each) | Pass, 2 controls / 5/5 | 0/5 (5 drops, 3–7 controls) | Pass, 1 control / 5/5 | 1/5 (2 drops, 2 palm, 1 pass) | Drop, control 1 / not scheduled |
| online_500 (2500) | Drop after 6, peak 5.589 / 1/5 (4 drops, 1 pass in 3 controls) | Pass, 2 / 1/5 (2 palm, 2 drops, 1 pass) | 0/5 (5 drops) | Pass, 1 / 3/5 (3 passes, 1 drop, 1 environment success at 6.226 rad force-qualified counted as failure) | 0/5 (4 drops, 1 palm) | Drop, control 2 / not scheduled |
| online_1000 (3000) | Drop after 6, peak 5.601 / 3/5 (3 passes in 3 controls, 2 drops) | Pass, 2 / 1/5 (2 palm, 2 drops, 1 pass) | 1/5 (4 drops, 1 pass) | Pass, 1 / 3/5 (3 passes, 1 drop, 1 environment success with one force-bearing finger counted as failure) | 0/5 (3 drops, 2 palm) | Drop, control 2 / not scheduled |
| online_2000 (4000) | Drop after 5, peak 6.282 / 1/5 (3 drops, 1 palm, 1 pass in 3 controls) | Pass, 2 / 3/5 (3 passes, 2 drops) | 0/5 (5 drops) | Pass, 1 / 5/5 | 1/5 (3 drops, 1 palm, 1 pass) | Drop, control 2 / 0/5 (5 drops, all at control 2) |

Deterministic prefix-6 trials peaked at 5.34/5.59/5.60/6.28 rad before dropping. Two prefix-8 stochastic trials (online 500 seed 20985; online 1,000 seed 20981) were environment successes that fail the finger-supported criterion (force-qualified angle must reach 2π and the final state must have two force-bearing fingers: 6.226 rad < 2π; one force-bearing finger) and count as failures above.

All 121 evaluation trials: terminated=true, truncated=false, prefix_physics_bitexact=true, handoff_terminated_before_policy=false; each evaluation summary reports training_rng_restored true, learner_tensors_and_counters_unchanged true, eligible_for_primitive false. Noisy trials use the same per-seed perturbation at every point (handoff_noise_l2 identical across checkpoints, e.g. prefix-7 seed 21971 = 0.1718).

## Interpretation and scope

The replicate was meant to separate design effect from seed luck. Both seeds show the same kind of effect: unlike SAC0004 (prefix 6 0/6 at every point), the recipe yields prefix-6 successes in training (32/182 here, 9/190 in 0008) and sporadic frozen prefix-6 successes (0008: stochastic 0/5, 1/5, 0/5, 0/5 and a deterministic pass at online 1,000 only; 0009: stochastic 0/5, 1/5, 3/5, 1/5, deterministic never). In neither seed does the deterministic prefix-6 trial pass at 2,000 or the stochastic rate exceed 3/5, and neither seed ends with prefix-7 stochastic retention intact (0008: 5/5, 4/5, 3/5, 1/5, lost gradually; 0009: 5/5 to 1/5 already at online 500, recovering to 3/5). The BC term places the actor mean .016–.032 from recorded control 7 at the handoff, so the residual failures are not a first-control mean error at the handoff; the measured changes are the handoff latent std rising 2.4x (.067 → .163) and the critic's renewed negative bias on the demonstration rows (−.03 after pretraining, −.62 to −.75 online, −.30 at the end) under a 5% share. The fall from 3/5 to 1/5 between 1,000 and 2,000 rests on five trials and is not a trend.

One seed per configuration (two across 0008/0009), five stochastic trials per cell. Training successes are changing-policy assisted episodes and every evaluation trial replays recorded prefixes, so nothing here is primitive- or main-dataset-eligible. Cold reset is 0/1 at every point and 0/6 at 2,000, so no checkpoint is deployment-eligible. Variant 0008 has no findings file yet; its figures come from its `summary.json` and `evaluations/`.


_Audit note: drafted by an auditing agent from the artifact files and independently refuted-checked; 6 claim(s) corrected by the verifier before writing._
