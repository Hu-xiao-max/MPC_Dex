# Seed replicate of recorded_bc10_mixed: prefix-6 deterministic pass only at online 2,000, stochastic 1/5, retention kept, cold still fails

**The replicate reproduces the design's retention effect and its prefix-6 ceiling, not a robust handoff:** with fresh seeds (train 19100, prefix 19102, demonstration 19101), lambda-10 supervision toward the three recorded rows and prefixes {6,7,8} in the stream, prefix-7 and prefix-8 retention held at every checkpoint (prefix 7 deterministic 4/4, stochastic 19/20; prefix 8 deterministic 4/4, stochastic 20/20), the prefix-6 deterministic trial passed only at online 2,000 (3 controls, 6.646 rad, four loaded fingers), and prefix-6 stochastic never exceeded 1/5. Cold trials failed at control 1-2 at every point, including 0/5 stochastic at the end. The stage completed its budget (500 mixed critic updates, 2,000 mixed SAC updates, exit 0, 637.4 s).

## Actual execution

- Inherited SAC0001 final checkpoint (SHA `e4a73d89...`); final counters BC 12,000 / pretraining 1,000 / online 4,000; optimizer steps critic 5,000, actor/alpha 4,000, as prescribed (`checkpoints/online_2000/state.json`). Counters in `metrics.jsonl` are consecutive (500 `critic_pretrain_update` at 501-1,000, 2,000 `online_update` at 2,001-4,000).
- Warm-up: 256 episodes under the unchanged inherited learner (counters 12,000/500/2,000 in every warm-up report; 92/87/77 at prefixes 6/7/8), 548 suffix rows, 162 force-qualified successes (0/85/77 by prefix), 93 drops, 1 palm-support failure; informational only.
- Demonstration buffer: 3 recorded rows (source controls 7-9), no MPC rows; 77 demonstration + 179 online rows per batch in both phases. Pretraining critic loss .242 -> .009; demonstration-row bias -.677/-.619 -> -.025/+.031; held-out prefix-6 RMSE .299/.306 -> .185/.114 (14 terminal rows, never inserted in replay).
- Online: 2,000 interactions, 796 completed assisted episodes (239/260/297 by prefix): 525 force-qualified successes (44/186/295), 1 environment success that is not force-qualified (episode 428, prefix 6, one load-bearing finger at termination, 6.217 rad force-qualified), 218 drops, 52 palm-support failures (2 of which also register an independent drop), 0 truncations, 0 handoffs terminated before the policy; one further episode (796, prefix 6, one policy control) is open and its terminal was not fabricated. All 7,413 physical prefix controls (1,777 warm-up, 5,630 online, 6 in the open episode) replayed bit-exactly and never entered replay (2,548 replay rows = 548 + 2,000). Every training episode ran under a policy that changed between decisions.
- Prefix-6 force-qualified training success by 200-episode window (last window 196): 10/66, 15/62, 10/57, 9/54.
- Actor BC loss .110 -> .0008; deterministic distance to recorded control 7 at the prefix-6 handoff observation .413 -> .073 (500) -> .058 (1,000) -> .019 (2,000); latent std .067 -> .167. Learned alpha .000306 -> .000378 (raw-equivalent .0756). Demonstration-row critic bias re-drifted to -.221/+.022 (500), -.164/-.117 (1,000), -.234/-.102 (2,000).

## Frozen evaluations

Each point: one deterministic (seed 42) and five stochastic trials per prefix in {6,7,8}, cold deterministic at every point, five cold stochastic at the final point only; no noisy trials were scheduled. Training RNG restored and learner unchanged at every point. Every trial terminated by the environment; none was truncated.

| Checkpoint (learner online counter) | Prefix 6 det. / stoch. 21061-65 | Prefix 7 det. / stoch. 21071-75 | Prefix 8 det. / stoch. 21081-85 | Cold det. / stoch. 21091-95 |
| --- | --- | --- | --- | --- |
| pretraining_baseline (2,000) | Drop, 3 controls / 0/5 (all drops, 2-3 controls) | Pass / 5/5 | Pass / 5/5 | Drop, control 1 / not scheduled |
| online_500 (2,500) | Drop, 4 / 1/5 (21065 pass, 3 controls, 6.431 rad; 4 drops) | Pass / 5/5 | Pass / 5/5 | Palm support, control 2 / not scheduled |
| online_1000 (3,000) | Drop, 4 / 1/5 (21065 pass, 3 controls, 6.656 rad; 4 drops) | Pass / 4/5 (21072 palm support) | Pass / 5/5 | Palm support, control 2 / not scheduled |
| online_2000 (4,000) | Pass, 3 controls, 6.646 rad, four loaded fingers / 1/5 (21064 pass, 3 controls, 6.586 rad; 21061 and 21062 drops, 21062 reaching 6.200 rad before dropping; 21063 and 21065 palm support) | Pass / 5/5 | Pass / 5/5 | Palm support, control 2 / 0/5 (all palm support, control 2) |

## Interpretation and scope

The hypothesis under test was that variant 0005's outcome reflected its seeds (0005: train 17491, prefix 17493). Both seeds agree that the supervised pull moves the deterministic mean onto recorded control 7 (final L2 .019 here, .030 in 0005) without a retained stochastic gain at prefix 6 (here 1/5 at every online checkpoint; 0005 2/5 at online 500, then 0/5 at 1,000 and 2,000). Retention differed by seed: prefix-7/8 completion held fully here (19/20 and 20/20 stochastic), whereas 0005 lost a prefix-8 stochastic trial at online 500 and fell to 2/5 prefix-7 stochastic at online 2,000. The single prefix-6 deterministic pass at 2,000, absent at 500 and 1,000 and absent in 0005 at every point, is one frozen trial at a knife-edge handoff and is not evidence of a stable basin. The changing-policy training rate at prefix 6 (10/66, 15/62, 10/57, 9/54, stochastic sampling) is consistent with the frozen stochastic 1/5 and carries no information about deterministic stability. Cold behaviour remains immediate failure without rotation, as in SAC0001; the failure mode shifted from a drop at control 1 (baseline, as in SAC0001) to a palm-support violation at control 2 after online training.

Single seed per replicate; two seeds total for this design. All collected episodes are assisted or demonstration-mixed and none qualifies for a primitive; cold results (0/1 deterministic, 0/5 stochastic) decide deployment eligibility, so no checkpoint here is deployable or a validated initialization for a later stage.

Inputs and protocol: `protocol.json`, `execution_manifest.json`, `input_bindings.json`; results: `summary.json`, `warmup_record.json`, `metrics.jsonl`, `episodes/`, `evaluations/`, `checkpoints/`, `demonstration_buffer.json`, `demonstration_calibration_*.json`, `heldout_calibration_*`, `warmup_calibration_*`, `unfinished_suffix/`; final hashes: `manifest.json`.


_Audit note: drafted by an auditing agent from the artifact files and independently refuted-checked; 4 claim(s) corrected by the verifier before writing._
