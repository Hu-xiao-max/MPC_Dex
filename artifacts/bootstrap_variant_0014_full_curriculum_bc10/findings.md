# Full reverse curriculum with a 9-row BC pull: prefix-7/8 retention held at 2,000, prefix-6 pass transient, one prefix-5 stochastic pass, nothing cold

**The variant finished its 2,000 online updates with prefix-7 and prefix-8 retention intact at the final checkpoint (6/6 each at online 2,000; prefix 8 dipped to 4/6 at online 500 and prefix 7 to 4/6 at online 1,000), and produced this artifact's only frozen prefix-6 deterministic pass (online 500, with 3/5 stochastic), which was gone by online 1,000 (1/6, one stochastic pass) and 2,000 (0/6).** Neither result is a first within the wave-1 family: variants 0004, 0006 and 0010 also hold prefixes 7 and 8 at 6/6 at online 2,000 (0006 at every point), and variants 0003, 0004, 0006, 0007, 0008 and 0010 each have at least one frozen prefix-6 deterministic pass. Prefixes 1–4 never passed any of their 24 trials per checkpoint (0/96); prefix 5 passed one stochastic trial (17612, online 1,000) and nothing else. All 24 cold trials failed, dropping at control 1 at baseline and at control 3 by 2,000. The stage ran its full budget (500 mixed critic updates, 2,000 mixed SAC updates, exit 0, 536.0 s).

## Actual execution

- Inherited SAC0001 final checkpoint (SHA `e4a73d89…`, pinned runner `241f96de…`); final counters BC 12,000 / pretraining 1,000 / online 4,000, optimizer steps critic 5,000 and actor/alpha 4,000, as prescribed.
- Warm-up: 256 frozen-actor episodes sampled uniformly over prefixes 0–8 (40/33/32/31/17/23/32/19/29), 705 suffix rows, 930 prefix controls. 48 successes, all at prefixes 7 (19/19) and 8 (29/29); 161 drops, 47 palm-support failures; all 40 cold episodes dropped after 1–2 controls with zero force-qualified progress.
- Demonstration buffer: 9 recorded rows (source controls 1–9, one terminal episode, no MPC rows), 77 of 256 rows per batch; BC term lambda 10 on all 9 rows. Pretraining critic loss .136 → .024; demonstration target mean .151 → .575 against entropy-free targets 1.11–1.63.
- Online: 2,000 interactions, 443 completed episodes: 395 assisted with 89 force-qualified successes (prefix 5: 4/45, prefix 6: 5/44, prefix 7: 42/60, prefix 8: 38/39, prefixes 1–4: 0/207) and 48 cold with 0 successes (36 drops, 12 palm support, 2–12 controls). Every prefix-5/6 success used exactly 4/3 suffix controls, the recorded trajectory length; every prefix-7/8 success used 2/1. 227 drops, 127 palm-support failures, no truncation. Last episode (prefix 1) remains open; replay holds 2,705 rows. All 2,637 physical prefix controls replayed bit-exactly and never entered replay; handoff never terminated before the policy.
- BC loss .294 at the first online update → .0017 by update 2,250 → .00061 at 3,000 → .00063 at 4,000, never above .0017 after 2,250. Learned alpha .000305893 → .000378939 (raw-equivalent .0612 → .0758).

## Value calibration

| Rows | Before | After pretraining | Online 500 | Online 1,000 | Online 2,000 |
| --- | --- | --- | --- | --- | --- |
| 9 recorded demonstration rows, bias | −1.41 / −1.46 | −.83 / −.81 | −.40 / −.42 | −.46 / −.48 | −.71 / −.61 |
| same, RMSE | 1.55 / 1.67 | 1.04 / 1.02 | .49 / .51 | .59 / .65 | .84 / .71 |
| 15 held-out prefix-6 rows, RMSE | .329 / .339 | .095 / .085 | — | — | — |
| 705 pooled warm-up rows, RMSE | .220 / .300 | .166 / .159 | — | — | — |

Unlike SAC0004 (pooled bias −.10 / −.05 over its 8 rows after pretraining: −.125/−.093 on the 5 MPC rows, −.061/+.014 on the 3 recorded rows), the critic never reaches the recorded rows' values; the bias is smallest at online 500 and grows again afterwards. At the first row (cold reset → recorded control 1) the deterministic mean moved from L2 2.00 to .147 (500), .146 (1,000) and .112 (2,000) while latent std went .053 → .059 → .021 → .0066. The actor log density of the prefix-6 handoff action (row 6) was 60.2 at 500, 49.9 at 1,000 and 8.9 at 2,000.

## Frozen evaluations

| Checkpoint | Prefix 1 det / sto | Prefix 2 | Prefix 3 | Prefix 4 | Prefix 5 | Prefix 6 | Prefix 7 | Prefix 8 | Cold det | Cold sto 17511–17515 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline (counters 12,000/500/2,000) | Drop@1 / 0/5 | Drop@8 / 0/5 | Drop@8 / 0/5 | Palm@1 / 0/5 | Drop@5 / 0/5 | Drop@3 / 0/5 | Pass@2 / 5/5 | Pass@1 / 5/5 | Drop@1, peak 0 rad | 0/5 (all drop@1, peak 0 rad) |
| Online 500 | Drop@9 / 0/5 | Palm@3 / 0/5 | Drop@7 / 0/5 | Drop@7 / 0/5 | Drop@5 / 0/5 | Pass@3 / 3/5 (2 drops@4) | Pass@2 / 5/5 | Pass@1 / 3/5 (2 drops@2) | Drop@6, peak .93 rad | 0/5 (4 drops, 1 palm; 2–4 controls; peak .12–.80 rad) |
| Online 1,000 | Palm@3 / 0/5 | Drop@8 / 0/5 | Palm@2 / 0/5 | Drop@8 / 0/5 | Drop@5 / 1/5 (17612 pass@4) | Drop@5 / 1/5 (17502 pass@3) | Pass@2 / 3/5 (2 palm@3–4) | Pass@1 / 5/5 | Drop@4, peak .01 rad | 0/5 (all drops, 2–6 controls; peak .00–.04 rad) |
| Online 2,000 | Drop@7 / 0/5 | Palm@8 / 0/5 | Palm@2 / 0/5 | Drop@6 / 0/5 | Drop@4 / 0/5 | Drop@4 / 0/5 | Pass@2 / 5/5 | Pass@1 / 5/5 | Drop@3, peak .43 rad | 0/5 (all drop@3; peak .18–.60 rad) |

No noisy prefixes were scheduled (handoff noise off). Every point ran 54 trials (8 prefixes x 6 + cold deterministic + 5 cold stochastic); all terminated by drop, palm support or success, none by truncation. Every evaluation restored the training RNG and left learner tensors and counters unchanged.

## Interpretation and scope

The hypothesis was that sampling every handoff state at once, with the whole recorded trajectory as demonstrations and a supervised pull, would preserve the mastered suffix and could show cold progress. The first half holds at the final checkpoint: with prefix-7/8 episodes in the stream (99 of 443 online episodes, 80 successes) both prefixes are 6/6 at 2,000, where SAC0004 lost prefix 7 by 500; the same retention was also obtained by variants 0004, 0006 and 0010 with fewer demonstration rows, so this variant does not isolate the full curriculum as its cause. The prefix-6 pass at 500 shows the mean can be placed inside the narrow basin at obs6, but only 9 of 89 training episodes at prefixes 5–6 succeeded and the critic undervalues the demonstration rows by .40–.83 throughout. The pass was lost by 1,000 while the BC loss stayed at or below .0017, so the supervised term kept the actor mean on the 9 demonstration states; what was lost is the continuation from those states as the latent std collapsed and the critic bias grew again. This stage does not isolate which of those changes is responsible.

Cold behaviour changed but never succeeded: frozen cold drops moved from control 1 (baseline) to control 3 (2,000), with peak progress .93 rad (deterministic, 500) and .43 rad (deterministic, 2,000) but at most .04 rad in all six online-1,000 cold trials; during training one cold episode reached 3.49 rad and 87 stayed below 2.4 rad. The recorded control 1 is matched to L2 .11 at reset yet its continuation is not.

This is a single seed matched to SAC0004 (no replicate). Prefix-sampled and demonstration-mixed episodes are assisted data and are not primitive evidence; the cold results (0/24 frozen, 0/88 training) are the ones that decide deployment eligibility, and they are negative. The online-500 checkpoint holds the only prefix-6 deterministic pass and full prefix-7 retention, but its prefix-8 stochastic 3/5 and the later loss of prefix 6 (one stochastic pass at 1,000, none at 2,000) mean no checkpoint here is a validated initialization without a further decision. No trajectory qualifies for a primitive.

Inputs and protocol: `protocol.json`, `execution_manifest.json`, `input_bindings.json`; results: `summary.json`, `warmup_record.json`, `metrics.jsonl`, `episodes/`, `evaluations/`, `checkpoints/`, `demonstration_buffer.json`, `demonstrations_replay.npz`, `demonstration_calibration_*.json`, `warmup_calibration_*`, `heldout_calibration_*`, `unfinished_suffix/`; final hashes: `manifest.json`. Family comparisons come from the sibling variants' `evaluations/<point>/summary.json`.


_Audit note: drafted by an auditing agent from the artifact files and independently refuted-checked; 7 claim(s) corrected by the verifier before writing._
