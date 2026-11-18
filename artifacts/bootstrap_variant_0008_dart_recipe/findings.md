# Dart recipe: one deterministic prefix-6 pass at online 1,000, stochastic prefix 6 never above 1/5, prefix-7/8 stochastic retention degraded, cold fails

**The four-factor combination did not yield a robust prefix-6 continuation.** The frozen deterministic prefix-6 trial passed once (online 1,000, 3 controls, 6.448 rad) and dropped at 500 and 2,000; stochastic prefix 6 reached 1/5 only at online 500 and is 0/5 elsewhere. Deterministic prefix-7/8 retention held at every point, but stochastic retention fell from 5/5 and 5/5 at baseline to 1/5 and 2/5 at online 2,000. Noisy-handoff trials, the regime the sigma .05 noise was meant to master, never exceeded 1/5. Every cold trial failed within 1–3 controls. The stage completed its budget (500 mixed critic updates, 2,000 mixed SAC updates, exit 0, 596.1 s).

## Actual execution

- Inherited SAC0001 final checkpoint; final counters BC 12,000 / pretraining 1,000 / online 4,000; optimizer steps critic 5,000, actor/alpha 4,000; online replay 2,804 rows, last suffix finished (episode 573 ended at interaction 2,000).
- Warm-up: 256 frozen-actor episodes, 804 suffix rows, 1,767 prefix controls; keys 6 / 7+noise / 8+noise = 105 / 71 / 80 episodes with 0 / 5 / 31 force-qualified successes (36, informational); prefix 6 all drops (105/105).
- Demonstration buffer: 3 recorded rows (source controls 7–9); 77 per pretraining batch, 13 per online batch; BC term lambda 10 over all 3 rows. `actor_bc_loss` .1104 at update 2,001 → .00072 at 2,500 → .00038 at 4,000.
- Online: 2,000 interactions, 574 completed assisted episodes (190 / 184 / 200), 58 successes (9 / 9 / 40), 371 drops and 143 palm-support failures by primary failure reason (15 of the palm-support episodes also carry the drop flag), 2 environment successes below the strict criterion (episode 9: force-qualified 6.228 rad < 2π with 4 fingers; episode 478: 6.483 rad with one force-bearing finger). Prefix-6 successes by 500-update window: 0/58, 3/52, 3/47, 3/33. All 574 episodes terminated (no truncation); no handoff terminated before the policy.
- Prefix replay: 5,795 physical prefix controls (4,028 online). The 5,260 exact prefix controls (3,644 online) replayed bit-exactly; the 535 final prefix controls of perturbed episodes (151 warm-up + 384 online) were executed as recorded action + N(0, .05²) from the bit-exact recorded pre-state. Executed handoff-noise L2 .07–.34 pooled (warm-up .070–.337, online .083–.281). `prefix_physics_bitexact` is true on all 830 training reports; prefix 6 was never perturbed.
- Learned alpha .000305893 → .000364248 (raw-equivalent .0612 → .0728).

## Calibration

Demonstration-row critic bias −.677/−.619 before → +.049/+.019 after pretraining (RMSE .096/.097) → −.570/−.596 at 500 → −.775/−.694 at 1,000 → −.651/−.626 at 2,000. Warm-up rows RMSE .432/.410 → .206/.194; held-out prefix-6 rows .329/.339 → .056/.043. Deterministic distance to recorded control 7 at the handoff .413 (baseline, pretrained) → .040 → .023 → .062; latent std there .067 → .093 → .130 → .183.

## Frozen evaluations

| Checkpoint | Prefix-6 det. | Prefix-6 stoch. | Prefix-7 det. / stoch. | Prefix-7 noisy | Prefix-8 det. / stoch. | Prefix-8 noisy | Cold det. | Cold stoch. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | Drop, 3 | 0/5 | Pass / 5/5 | 0/5 | Pass / 5/5 | 1/5 | Drop, 1 | not scheduled |
| Online 500 | Drop, 5 | 1/5 (17504) | Pass / 4/5 | 0/5 | Pass / 2/5 | 0/5 | Drop, 2 | not scheduled |
| Online 1,000 | Pass, 3 | 0/5 | Pass / 3/5 | 1/5 | Pass / 2/5 | 0/5 | Drop, 2 | not scheduled |
| Online 2,000 | Drop, 4 | 0/5 | Pass / 1/5 | 1/5 | Pass / 2/5 | 1/5 | Drop, 1 | 0/5 |

Detail: baseline prefix-6 stochastic all drops at 3 controls; prefix-7 noisy 4 drops, 1 palm; prefix-8 noisy pass 17555, 2 palm, 2 drops. Online 500: 17504 passes in 3 controls, 2 palm, 2 drops; prefix-7 stochastic 17525 palm; prefix-8 stochastic 1 drop, 2 palm; prefix-7 noisy 4 drops, 1 palm; prefix-8 noisy 2 drops, 3 palm. Online 1,000: prefix-6 stochastic 4 drops, 1 palm; prefix-7 stochastic 1 drop, 1 palm; prefix-7 noisy pass 17542, 3 drops, 1 palm; prefix-8 stochastic 1 drop, 2 palm; prefix-8 noisy 2 drops, 3 palm. Online 2,000: prefix-6 stochastic all drops at 4 controls; prefix-7 stochastic pass 17522, 2 palm, 2 drops; prefix-7 noisy pass 17545, 4 drops (one after 11 controls); prefix-8 stochastic 3 drops; prefix-8 noisy pass 17555, 1 palm, 3 drops; cold stochastic 3 drops, 2 palm at 1–3 controls.

Every evaluation trial terminated (no truncation); all 121 trial reports have `prefix_physics_bitexact` true and `handoff_terminated_before_policy` false; each point records `training_rng_restored` and `learner_tensors_and_counters_unchanged` true; per-trial report.json files match the point summaries (0 mismatches). Noisy trials use the same perturbation per seed at every point (L2 .144–.250).

## Interpretation and scope

BC lambda 10 did fix the mean at obs6 (L2 .023–.062 versus .413 inherited), yet the deterministic trial from that state passed only at 1,000 and dropped at 500 and 2,000 with the mean within .07 of the label: the pass is a knife-edge, not a basin. Sigma .05 handoff noise (executed L2 .07–.34) strongly degrades the inherited continuation without defeating it: with the frozen inherited actor, perturbed warm-up episodes succeeded 5/71 (prefix 7) and 31/80 (prefix 8), and at baseline unperturbed prefix-7/8 frozen trials passed 10/10 against 1/10 perturbed. So 384 of the 574 online episodes were noisy-prefix data in which 335 failed and 49 succeeded, rather than the mastered suffix the mixed-prefix factor was meant to supply. Under the 5% online demonstration share the critic undervalued the demonstration rows throughout online training (bias −.57 to −.77, about −.65 at 2,000) while the latent std at the handoff rose to .18; this is consistent with, but not shown by this run to be caused by, the small share, and the widening std is consistent with stochastic draws leaving the narrow pass region. The joint-sufficiency hypothesis is not supported at this seed.

Scope: single seed (the replicate `bootstrap_variant_0009_dart_recipe_r2` is a separate stage). Training successes above come from a changing policy; only the frozen table is policy evidence. All data are assisted (prefix-replayed) or demonstration-mixed and never primitive-eligible. Cold results decide deployment eligibility and are negative at every point. No checkpoint here is a valid initialization without a separate decision.


_Audit note: drafted by an auditing agent from the artifact files and independently refuted-checked; 7 claim(s) corrected by the verifier before writing._
