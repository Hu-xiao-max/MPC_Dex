# Mixed prefixes {6,7,8}: prefix-8 retained, prefix-7 still lost by online 500, no prefix-6 frozen success

**Sampling the collection prefix uniformly from {6,7,8} did not prevent forgetting:** frozen prefix-7 trials fell from 6/6 at baseline to 0/6 by online 500 and never recovered, as in SAC0004 with a prefix-6-only stream (the counts match; the failure mode at online 500 differs, drops at 14–26 controls here versus a full-horizon hold there). Prefix-8 (one recorded control short of the turn) was retained at every checkpoint except a 2/5 stochastic dip at online 500. No frozen trial at prefix 6 or from a cold reset succeeded at any checkpoint. The stage completed its full budget (500 mixed critic updates, 2,000 mixed SAC updates, exit 0, 543.4 s wall).

## Actual execution

- Inherited SAC0001 final checkpoint (SHA `e4a73d89…`); final counters BC 12,000 / pretraining 1,000 / online 4,000, optimizer steps critic 5,000 and actor/alpha 4,000, as prescribed. Only `prefix_set` differs from SAC0004; seeds match.
- Warm-up: 256 frozen-actor episodes, prefix 6/7/8 drawn 105/71/80 times, 549 suffix rows (327/142/80 by prefix), 151 force-qualified successes (0/105 at prefix 6, all drops after 2–5 controls; 71/71 at prefix 7 in 2 controls; 80/80 at prefix 8 in 1 control). Informational only; the gate is the 8 verified demonstration rows.
- Pretraining: 77 demonstration + 179 online rows per batch; demonstration target mean .407 → 1.033; critic loss .186 → .0028.
- Online: 2,000 interactions, 91 completed assisted episodes (40/26/25 by prefix), 24 successes (0/4/20 by prefix), 32 drops (20/8/4), 24 palm-support failures (15/8/1), 11 horizon truncations (5 prefix-6 at 94 controls, 6 prefix-7 at 93) ending at cumulative updates 2,687–3,795 (episodes 62–83). The 4 prefix-7 successes are episodes 7, 12, 16, 37 (cumulative updates 2,024–2,343); the other 22 prefix-7 episodes failed. Prefix-8 failed only in episodes 34, 40, 41, 42, 46 (updates 2,324–2,393). The last suffix (episode 91, prefix 7, 4 controls) remains open; replay holds 2,549 rows (549 warm-up + 2,000 online, including the 4 open rows). All 2,396 physical prefix controls (1,767 warm-up, 622 online, 7 open) replayed bit-exactly, none perturbed, none terminated before handoff, none entered replay.
- Learned alpha .000305893 → .000374881 (raw-equivalent .0612 → .0750); no BC term (`actor_bc_loss` null throughout); no handoff noise; no noisy evaluation prefixes scheduled.

## Value calibration

| Rows | Critic bias / RMSE before | After pretraining | Online 500 | Online 1,000 | Online 2,000 |
| --- | --- | --- | --- | --- | --- |
| 5 MPC demonstration rows | −1.26/−1.35 ; 1.26/1.36 | −.13/−.13 ; .20/.21 | −.02/−.02 ; .05/.04 | −.00/−.01 ; .03/.07 | −.05/−.05 ; .06/.06 |
| 3 recorded demonstration rows | −.68/−.62 ; .88/.93 | −.06/−.07 ; .09/.11 | −.09/−.08 ; .11/.10 | −.01/−.04 ; .10/.14 | −.05/−.06 ; .17/.17 |
| 549 warm-up rows (pooled prefixes) | RMSE .248/.273 | .081/.072 | — | — | — |
| 15 held-out prefix-6 rows (baseline stochastic 17501–17505) | RMSE .329/.339 | .047/.054 | — | — | — |

Deterministic-mean L2 distance at the prefix-6 handoff: to the MPC label .96 → .96 → 2.07 → 1.38 → .55; to recorded control 7 .41 → .41 → 1.83 → 1.03 → 1.12; latent std .067 → .067 → .246 → .337 → .301.

## Frozen evaluations

| Checkpoint | Prefix-6 det. / stoch. 17501–17505 | Prefix-7 det. / stoch. 17521–17525 | Prefix-8 det. / stoch. 17531–17535 | Cold det. / cold stoch. 17511–17515 |
| --- | --- | --- | --- | --- |
| Pretraining baseline (online 2,000 inherited) | Drop at 3 / 0/5, all drops at 3 | Pass, 2 controls / 5/5 (2 controls, 6.56–6.92 rad) | Pass, 1 control / 5/5 (1 control, 6.69–6.87 rad) | Drop at control 1 / not scheduled |
| Online 500 | Truncated at 94 (final 4.50 rad, max 5.28, 3 fingers) / 0/5 (3 drops at 13–23, 2 palm at 11–12) | Drop at 16 (max 6.20 rad) / 0/5, all drops at 14–26 (max 5.48–6.02 rad) | Pass, 1 control / 2/5 (17531, 17532 pass; 17533–17535 drop at 3) | Drop at 3 / not scheduled |
| Online 1,000 | Truncated at 94 (final 5.43, max 5.60, 3 fingers) / 0/5, all truncated at 94 (final 5.11–5.53, max 5.66–5.80, 3 fingers) | Truncated at 93 (final 5.61, max 5.69, 3 fingers) / 0/5, all truncated at 93 (final 5.15–5.55, max 5.65–5.71, 3 fingers) | Pass, 1 control / 5/5 (1 control, 6.60–6.70 rad) | Truncated at 100 (held, final .82 rad, max 1.43, 3 fingers, no turn) / not scheduled |
| Online 2,000 | Palm support at 30 (5.61 rad) / 0/5 (palm at 7, 46, 31 for 17501, 17503, 17504; truncated at 94 for 17502 and 17505 at 5.70/5.61 rad, 3 and 4 fingers) | Truncated at 93 (final 5.75, 3 fingers) / 0/5, all truncated at 93 (final 5.50–5.80, max 5.67–5.89; 4 fingers in 17522 and 17524, 3 otherwise) | Pass, 1 control / 5/5 (1 control, 6.47–6.66 rad) | Drop at 3 / 0/5 (drops at 8, 3, 8, 2; palm at 13) |

\"Truncated\" trials end by the 100-control horizon; their return tail is unknown and they are failures. Every evaluation restored the training RNG, left learner tensors and counters unchanged, and inserted nothing into replay.

## Interpretation and scope

The hypothesis that forgetting is critic-led and cured by keeping mastered states in the stream is not supported for prefix 7: with 71 warm-up prefix-7 episodes (142 rows) in replay from the start, 13 online prefix-7 episodes completed before the online-500 evaluation (26 by the end of the stage), and demonstration-row critic bias within ±.13 after pretraining, the two-control prefix-7 completion was still lost by online 500. What survived is the one-control prefix-8 completion, whose training episodes kept succeeding (20/25) and so kept re-entering replay as positive rows; the prefix-7 training successes stopped after episode 37 (update 2,343). From online 1,000 on the policy holds the object to the horizon at prefixes 6 and 7 without completing the turn: final force-qualified angles 5.11–5.61 rad at online 1,000 and 5.50–5.80 rad at online 2,000 (peaks 5.60–5.89 rad), with three force-bearing fingers at the end in all twelve held online-1,000 trials and in five of the eight held online-2,000 trials (four in 17505, 17522, 17524). SAC0004 showed the same kind of stall near 5.6 rad. Mixed prefixes therefore preserved competence only where the policy's own successes continued. Whether an actor-side term (supervised pull at the handoff, handoff noise) changes this is not tested by this stage, which varied only `prefix_set`; those factors are separate variants in this family.

Single seed, matched to SAC0004; no replicate. All warm-up and online data are assisted (prefix-replayed) or demonstration-mixed and are never primitive evidence. Cold-reset trials fail at every point (one full-horizon hold at online 1,000 without a turn); no checkpoint here is deployable or a valid initialization without a separate decision, and SAC0001 remains the last checkpoint with verified prefix-7 competence.

Inputs and protocol: `protocol.json`, `execution_manifest.json`, `input_bindings.json`; results: `summary.json`, `warmup_record.json`, `metrics.jsonl`, `episodes/`, `evaluations/`, `checkpoints/`, `demonstration_buffer.json`, `demonstration_calibration_*.json`, `warmup_calibration_*`, `heldout_calibration_*`, `unfinished_suffix/`; final hashes: `manifest.json`.


_Audit note: drafted by an auditing agent from the artifact files and independently refuted-checked; 6 claim(s) corrected by the verifier before writing._
