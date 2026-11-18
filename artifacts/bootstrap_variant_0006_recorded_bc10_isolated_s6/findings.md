# BC-only handoff with isolated SAC: prefix-7/8 retention kept, prefix-6 deterministic pass at online 500 and 2,000 but not 1,000, stochastic 1/5 then 0/5, cold 0/9

**Removing prefix 6 from the on-policy stream and excluding demonstration rows from the SAC actor and temperature losses preserved the inherited suffix competence for the whole budget, and the supervised term alone brought the deterministic mean at obs6 to within .007–.010 (L2) of recorded control 7, but the frozen prefix-6 result is unstable:** the deterministic trial dropped at baseline (3 controls), passed at online 500 (3 controls), dropped at 1,000 (6 controls) and passed at 2,000 (3 controls); stochastic trials passed 1/5 at 500 and 0/5 at every other point. The stage completed its full budget (500 mixed critic updates, 2,000 mixed SAC updates, exit 0, 750.0 s wall).

## Actual execution

- Inherited SAC0001 final checkpoint (SHA `e4a73d89…`); final counters BC 12,000 / pretraining 1,000 / online 4,000, optimizer steps critic 5,000 and actor/alpha 4,000, as prescribed. Online replay 2,407 rows (407 warm-up + 2,000 online suffix rows), no open episode.
- Warm-up: 256 frozen-actor episodes at prefixes 7/8 (146/110), 407 suffix rows, 252 finger-supported successes (prefix 7: 142/146, prefix 8: 110/110); 3 drops (prefix 7) and one prefix-7 environment success (episode 000107) whose force-qualified angle 6.275 rad fell short of 2π. Successes informational; the gate was the 3 verified recorded demonstration rows (controls 7–9, one terminal episode).
- Pretraining: 77 demonstration + 179 online rows per batch, 40–64 demonstration rows bootstrapped entropy-free per update; demonstration target mean 1.074 → 1.135, critic loss .193 → .010.
- Online: 2,000 interactions, 1,213 completed assisted episodes (prefix 7: 598, prefix 8: 615), 1,133 successes (518 and 615), 54 drops and 26 palm-support failures, all at prefix 7; no truncation. These are changing-policy training episodes, not frozen evaluations. Every online update used 179 on-policy rows for the SAC actor/temperature losses and 3 rows for the BC term (λ 10), with 77 demonstration rows in the critic batch; BC loss .110 → .0009. All 11,008 physical prefix controls (1,902 warm-up + 9,106 online) replayed bit-exactly; none entered replay.
- Learned alpha .000305860 (inherited) → .000378973 (raw-equivalent .0758).

## Value and handoff calibration

Demonstration-row critic bias −.677/−.619 before pretraining, −.002/+.035 after, −.038/−.039 at online 2,000. The deterministic-mean L2 distance to recorded control 7 at obs6 was .413 at baseline and after pretraining (critic-only), then .0069 (online 500), .0103 (1,000), .0086 (2,000); latent std there .067 → .061 → .064 → .083. The held-out prefix-6 set (the 15 suffix rows of the five baseline prefix-6 stochastic drops 17501–17505, target −.43) is the exception: bias rose from +.16/+.05 to +.95/+1.00 after pretraining. No prefix-6 outcome of the actual policy enters any training batch, so nothing in the objective can correct this.

## Frozen evaluations

| Checkpoint (online counter) | Prefix-6 det. | Prefix-6 stochastic 17501–17505 | Prefix-7 det. / stoch. 17521–17525 | Prefix-8 det. / stoch. 17531–17535 | Cold det. | Cold stoch. 17511–17515 |
| --- | --- | --- | --- | --- | --- | --- |
| Pretraining baseline (2,000) | Drop, 3 controls | 0/5, all drops after 3 controls | Pass / 5/5 | Pass / 5/5 | Drop, 1 control | not scheduled |
| Online 500 (2,500) | Pass, 3 controls, 6.716 rad, 4 fingers | 1/5 (17503 pass, 3 controls, 6.352 rad); 4 drops after 4–8 controls | Pass / 5/5 | Pass / 5/5 | Drop, 2 controls | not scheduled |
| Online 1,000 (3,000) | Drop, 6 controls (final force-qualified 4.97 rad, max 5.46) | 0/5, all drops after 4–5 controls | Pass / 5/5 | Pass / 5/5 | Drop, 5 controls | not scheduled |
| Online 2,000 (4,000) | Pass, 3 controls, 6.480 rad, 4 fingers | 0/5, all drops after 3–4 controls (final force-qualified 2.99–5.05 rad) | Pass / 5/5 | Pass / 5/5 | Drop, 2 controls | 0/5, all drops after 2 controls |

All 81 evaluation trials (19+19+19+24) terminated (no truncation), prefix_physics_bitexact true, handoff_terminated_before_policy false, perturbed_last_prefix_control false; no noisy trials scheduled; learner counters in every trial equal the checkpoint's counters; training_rng_restored and learner_tensors_and_counters_unchanged true at every point. Prefix-7 trials complete in 2 suffix controls and prefix-8 in 1 at every point (12/12 prefix-7/8 trials per point, 48/48 across the run). Cold trials: 0/9 across the run (4 deterministic, 5 stochastic).

## Interpretation and scope

The hypothesis is confirmed in its two mechanistic parts: prefix-7/8 retention never degraded (12/12 at every point, unlike SAC0004's loss of prefix-7 retention by online 500), and the supervised term put the deterministic mean within .01 of the recorded control. The prefix-6 outcome is mixed rather than confirmed: the deterministic trial alternates drop/pass/drop/pass across the four checkpoints, and the stochastic trials drop after 3–8 suffix controls at every point except the single 17503 pass at online 500 (1/20 over the four points). A mean at the label with latent std .06–.08 is therefore not sufficient at this handoff, consistent with the narrow success basin around control 7 reported by the read-only probes cited in `protocol.json`; this artifact does not measure the basin itself. The critic overvalues the held-out prefix-6 drop states (bias about +1.0), which it never sees the policy fail from, and neither the on-policy actor loss (prefix-7/8 rows only) nor the 3-row BC term acts on prefix-6 policy outcomes. Single seed; the two deterministic passes are frozen results but come from different checkpoints of one run and are separated by a deterministic drop. All training episodes are assisted or demonstration-mixed and ineligible for primitives; cold reset fails 0/9 across the run and 0/6 at the final point, so no checkpoint is deployable.

Inputs and protocol: `protocol.json`, `execution_manifest.json`, `input_bindings.json`; results: `summary.json`, `warmup_record.json`, `metrics.jsonl`, `episodes/`, `evaluations/`, `checkpoints/`, `demonstration_buffer.json`, `demonstrations_replay.npz`, `demonstration_calibration_*.json`, `warmup_calibration_*`, `heldout_calibration_*`; final hashes: `manifest.json`.


_Audit note: drafted by an auditing agent from the artifact files and independently refuted-checked; 7 claim(s) corrected by the verifier before writing._
