# Chained prefix-5 handoff: obs5 mean moves to within .02–.07 of recorded control 6, frozen prefix-5 success peaks at 6/30, inherited prefix-6 rate falls from 20/30 to 7/30 within 250 updates and to 1/30 by 500, prefix 7 at 13/30 by online 2,000, cold still fails

**Moving the handoff one control earlier did not transfer the inherited competence.** From the prefix-5 handoff the frozen policy passes deterministically only at online 1,000 (4 controls, 6.62 rad) and never exceeds 1/5 in-stage or 6/30 large-sample stochastic trials; the inherited prefix-6 result (deterministic pass, 3/5, 20/30 at baseline) fell to a deterministic palm-support failure, 2/5 and 7/30 by online 250, to 0/5 and 1/30 by online 500, and ended at 0/5 and 7/30; prefix 7 ended at 1/5 and 13/30 with a deterministic drop; prefix 8 held 6/6 and 30/30 throughout. All 293 cold trials fail. Full budget completed (500 + 2,000 mixed updates, exit 0, 533.3 s).

## Actual execution

- Inherited `bootstrap_variant_0004_recorded_bc10_first/checkpoints/online_2000` (agent SHA `7d539f09…`, counters 12,000/1,000/4,000); final counters 12,000/1,500/6,000, optimizer steps critic 7,500 and actor/alpha 6,000, as prescribed; `auxiliary_rng_states` present in the final `state.json`. Settings vs SAC0004: `prefix_controls 5`, `prefix_set {5,6,7,8}`, no MPC rows, recorded rows 5–8, BC λ10 on the first row, no handoff noise; seeds match SAC0004 (train 17491, demonstration 17492, physical reset 42; prefix seed 17493 and handoff seed 17494 as in 0004).
- Warm-up (frozen inherited actor): 256 episodes, 918 suffix rows; prefix 5 0/82 (73 drops, 9 palm), prefix 6 40/64 (15 drops, 9 palm), prefix 7 47/58 (5 drops, 6 palm), prefix 8 52/52.
- Demonstration buffer: 4 recorded rows (source controls 6–9, raw return 259.35); 77 demonstration + 179 online rows in all 2,500 batches; BC on 1 row per update. Demonstration critic bias after pretraining −.06/−.08 (from −.60/−.59), held-out prefix-6 −.06/−.06 (from −.10/+.00); the actor is untouched by critic pretraining (obs5 distance and latent std identical before and after: .624, .049).
- Online: 2,000 interactions, 558 completed assisted episodes, 273 successes: prefix 5 9/143 (7 in 4 controls, one each in 6 and 8; first at episode 92, update 412), prefix 6 29/135, prefix 7 83/128, prefix 8 152/152; 265 drops, 20 palm failures. Episode 558 open at the budget (2 controls); replay 2,918 rows (918 warm-up + 2,000 online). All 5,258 replayed prefix controls bit-exact, none perturbed, none in replay; `handoff_terminated_before_policy` 0. `summary.json` matches these recounts exactly.
- BC loss .0569 → 3.0e−4. Obs5 mean distance to recorded control 6: .624 (baseline) → .071 (250) → .056 (500) → .028 (1,000) → .017 (1,250) → .039 (1,500) → .056 (2,000); latent std .049 → .079 → .107 → .187 → .199 → .264 → .275. Recorded-row critic bias re-drifted to −.30/−.33 at 1,500, −.26/−.12 at 2,000. Alpha .000377 → .000460 (raw-equivalent .0920), strictly monotone.

## Frozen evaluations (in-stage)

| Point (online counter) | Prefix 5 det / stoch 17611–17615 | Prefix 6 det / stoch 17501–17505 | Prefix 7 det / stoch 17521–17525 | Prefix 8 det / stoch 17531–17535 | Cold det (stoch 17511–17515 at final) |
| --- | --- | --- | --- | --- | --- |
| pretraining_baseline (4,000) | drop at 6 (peak 5.03) / 0/5 | pass 3c 6.70 / 3/5 (17501, 17504, 17505) | pass 2c 6.48 / 5/5 | pass 1c 6.62 / 5/5 | drop at 3 (peak 1.08) |
| online_250 (4,250) | drop at 5 / 0/5 | palm at 5 (peak 5.49) / 2/5 (17504 4c, 17505 6c) | palm at 5 / 3/5 (17522, 17524 palm) | pass / 5/5 | palm at 2 |
| online_500 (4,500) | drop at 6 / 0/5 | drop at 5 / 0/5 | pass 2c 6.62 / 0/5 | pass / 5/5 | palm at 2 |
| online_750 (4,750) | drop at 5 / 0/5 | drop at 5 / 0/5 | pass / 5/5 | pass / 5/5 | drop at 3 |
| online_1000 (5,000) | **pass 4c 6.62 rad, 4 fingers** / 1/5 (17615, 4c 6.68) | drop at 5 / 0/5 | pass / 1/5 (17521) | pass / 5/5 | drop at 2 |
| online_1250 (5,250) | drop at 5 / 0/5 | drop at 5 / 1/5 (17501) | pass / 5/5 | pass / 5/5 | drop at 2 |
| online_1500 (5,500) | drop at 5 / 0/5 | pass 3c 6.73 / 2/5 (17501, 17504) | pass / 5/5 | pass / 5/5 | drop at 2 |
| online_1750 (5,750) | drop at 4 / 0/5 | palm at 4 / 3/5 (17501, 17502, 17504) | pass / 4/5 (17522 drop at 8) | pass / 5/5 | drop at 2 |
| online_2000 (6,000) | drop at 6 / 0/5 | drop at 5 / 0/5 | drop at 3 / 1/5 (17524) | pass 1c 6.57 / 5/5 | drop at 2 (peak .83); 0/5, all drops at 2 |

No noisy trials were scheduled. All 230 trials (25 per point, 30 at online_2000) terminated (none truncated), prefix physics bit-exact, no handoff terminated before the policy, training RNG restored and learner unchanged at every point; per-trial `report.json` files match `summary.json`. The 19 trials the baseline shares with 0004's online_2000 evaluation (prefixes 6–8 and cold) are identical row-for-row (actions, physics, actor mean, latent std, log-probabilities).

## Large-sample frozen evaluation

`frozen_checkpoint_evaluation_0001` covers all nine checkpoints (agent hashes and learner counters match; deterministic trials agree with the table; 30 fresh-seed stochastic trials per prefix, seeds 23000 + 100·p + i; none truncated). Successes /30 for prefix 5 / 6 / 7 / 8 / cold: pretrained 0 / 20 / 26 / 30 / 0; 250: 0 / 7 / 17 / 30 / 0; 500: 0 / 1 / 13 / 30 / 0; 750: 6 / 7 / 22 / 30 / 0; 1,000: 2 / 8 / 22 / 30 / 0; 1,250: 3 / 5 / 28 / 30 / 0; 1,500: 3 / 9 / 22 / 30 / 0; 1,750: 4 / 6 / 23 / 30 / 0; 2,000: 3 / 7 / 13 / 30 / 0. Best prefix 5 is 6/30 = 20 % (Wilson 95 % 10–37 %) at 750, where the deterministic trial fails; of the 21 stochastic prefix-5 passes over the run, 20 use 4 controls and one 5. Prefix 6 is 7/30 (12–41 %) at 250 and 1/30 (1–17 %) at 500 against 20/30 (49–81 %) inherited, and peaks afterwards at 9/30 (17–48 %) at 1,500; prefix 7 at 2,000 is 13/30 (27–61 %) against 26/30 (70–95 %); prefix 8 is 30/30 with a deterministic pass everywhere; cold 0/279 (9 deterministic + 270 stochastic; drops after 2–5 controls or palm-support failures at control 2 or 4). The pretrained row equals 0004's online_2000 large-sample result (0/20/26/30/0).

## Interpretation and scope

The hypothesis (inherit 0004, supervise obs5 → recorded control 6, keep prefixes 6–8 in the stream against forgetting) is only partly supported: the BC term moved the obs5 mean from .62 to within .07 of the label by 250 updates and kept it within .02–.07 thereafter, yet this yielded at most 6/30 frozen prefix-5 successes, and 135 prefix-6 training episodes did not prevent the inherited prefix-6 rate falling from 20/30 to 7/30 within 250 updates and to 1/30 by 500 (recovering only to 5–9/30 afterwards), nor prefix 7 halving by 2,000. This matches the wave-1 findings that a mean near the label does not decide the outcome and that the handoff latent std grows under every prefix-6 stream (here .275, above the wave-1 range .13–.20; alpha .000460 above every wave-1 final value, .00036–.00038). The held-out prefix-6 critic bias after pretraining was −.06/−.06, so 0006's overvaluation (about +1.0) did not recur here. Variants 0028–0030 (both new handoffs supervised; 0034–0035 add an alpha freeze) are the designated follow-up.

Single seed, matched to SAC0004. All 273 training successes are assisted (five replayed controls) and demonstration-mixed under a changing policy; none is primitive evidence and no trajectory was exported. Cold trials fail at every checkpoint (in-stage 0/14, large-sample 0/279), so no checkpoint is deployment-eligible; frozen claims apply only to the checkpoints and seeds above.

Inputs and protocol: `protocol.json`, `execution_manifest.json`, `input_bindings.json`, `inherited_state.json`; results: `summary.json`, `warmup_record.json`, `metrics.jsonl`, `episodes/`, `unfinished_suffix/`, `evaluations/`, `checkpoints/`, `demonstration_calibration_*.json`, `heldout_calibration_*`; large sample: `../frozen_checkpoint_evaluation_0001/results/bootstrap_variant_0024_chain_p5_from_bc_first/`; final hashes: `manifest.json`.


_Audit note: drafted by an auditing agent from the artifact files and independently refuted-checked; 2 claim(s) corrected by the verifier before writing._
