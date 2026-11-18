# Control reseed: the SAC0004 failure and the online-500 retention loss reproduce under fresh seeds

**Changing only the seeds did not change the outcome in kind:** no deterministic frozen success at prefix 6 or 7 after the baseline, 1 of 85 completed online training episodes succeeded, and prefix-7 retention fell from 6/6 at baseline to 0/6 at online 500 as in SAC0003 and SAC0004. The stage completed its full budget (500 critic-pretraining updates, 2,000 online SAC updates, exit 0, 589.0 s wall). Two differences from SAC0004 appear, neither a frozen deterministic success: at online 1,000 one of five stochastic prefix-6 trials and one of five stochastic prefix-7 trials completed a turn, and at online 2,000 every prefix-6 and prefix-7 trial held the object to the horizon without turning, where SAC0004's stochastic trials at that point dropped or braced.

## Actual execution

- Inherited the SAC0001 final checkpoint (SHA `e4a73d89…`) through SAC0004's hash-bound inputs; final counters BC 12,000 / pretraining 1,000 / online 4,000, optimizer steps critic 5,000 and actor/alpha 4,000, as prescribed. Seeds: train 19130, demonstration 19131, prefix 19132; SAC0004 used 17491/17492.
- Warm-up: 256 frozen-actor prefix-6 episodes, 809 suffix rows, all drops after 2–5 controls (14/190/49/3 episodes at 2/3/4/5), zero successes (informational; gate passed on the 8 demonstration rows).
- Demonstration buffer: the same 8 rows (5 MPC, 3 recorded), 77 demonstration + 179 online rows per batch in both phases; 46–68 entropy-free bootstrap rows per pretraining update, demonstration target mean .537 → 1.040, pretraining critic loss .315 → .0023.
- Online: 2,000 interactions, 85 completed assisted episodes, 1 force-qualified success (episode 66, 79 suffix controls, cumulative update 2,953, 6.531 rad, 3 loaded fingers, no palm force), 63 independent drops (62 drop-labelled plus episode 29, a palm failure that also dropped), 18 palm-support failures, 4 horizon truncations (episodes 64, 67, 73, 75, each 94 suffix controls). Mean suffix length rose from 6.6 controls (episodes 0–27) through 11.4 (28–56) to 50.2 (57–84). The last episode (85) remains open at 75 controls; online replay holds 2,809 rows (809 warm-up + 2,000 online). handoff_terminated_before_policy 0.
- Physical prefix controls: 2,052 on the training side (1,536 warm-up + 510 completed online + 6 in the open episode) and a further 504 in the 72 prefix evaluation trials; all 2,556 replayed bit-exactly (prefix_physics_bitexact true in all 342 training reports including the open episode and all 81 evaluation reports) and none entered replay.
- Learned alpha .000305893 → .000369432 (raw-equivalent .0612 → .0739); actor_bc_loss null throughout (lambda 0).

## Value calibration

| Rows | Critic RMSE before | After pretraining | Online 500 | Online 1,000 | Online 2,000 |
| --- | --- | --- | --- | --- | --- |
| 5 MPC demonstration rows (bias) | 1.262 / 1.356 (−1.26 / −1.35) | .147 / .155 (−.091 / −.096) | .127 / .130 (−.087 / −.094) | .266 / .263 (−.219 / −.215) | .167 / .167 (−.142 / −.143) |
| 3 recorded demonstration rows (bias) | .878 / .933 (−.68 / −.62) | .041 / .038 (+.034 / +.028) | .107 / .132 (−.054 / −.068) | .266 / .264 (−.209 / −.203) | .394 / .402 (−.322 / −.328) |
| 809 warm-up on-policy rows | .319 / .335 | .083 / .076 | — | — | — |
| 15 held-out baseline rows | .268 / .289 | .082 / .086 | — | — | — |

The entropy-free target again removes the demonstration-row distortion (bias about −.1 after pretraining, versus +11.1 in SAC0003). At the prefix-6 handoff the actor's latent std jumped from .067 to .777 at online 500, then fell to .304 and .125; the deterministic distance to the MPC label went .961 → 2.402 → 1.318 → .707 and to the recorded label .413 → 2.069 → .967 → 1.296.

## Frozen evaluations

| Checkpoint | Prefix-6 det. | Prefix-6 stochastic 21361–21365 | Prefix-7 det. | Prefix-7 stochastic 21371–21375 | Prefix-8 det. | Prefix-8 stochastic 21381–21385 | Cold det. | Cold stochastic 21391–21395 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline (online 2,000 inherited) | Drop, 3 controls | 0/5, all drops (2–4 controls) | Pass, 2 controls | 5/5 | Pass, 1 control | 5/5 | Drop, control 1 | not scheduled |
| Online 500 | Palm, 7 | 0/5 (4 palm, 1 drop at 32) | Palm, 10 | 0/5 (2 palm, 3 drops at 18–23) | Pass, 1 | 3/5 (drops at 4 and 7) | Drop, 3 | not scheduled |
| Online 1,000 | Held 94, truncated, 5.76 rad, no turn | 1/5 (21362 success after 68 controls; 2 held to horizon; 1 palm at 6; 1 drop at 70 after 6.50 rad) | Held 93, truncated, 5.85 rad | 1/5 (21373 success after 66; 4 drops at 31–54) | Pass, 1 | 5/5 | Held 100, truncated, 2.73 rad, 3 fingers, no drop | not scheduled |
| Online 2,000 | Held 94, truncated, 5.67 rad | 0/5, all held to horizon (final 4.41–5.62 rad) | Held 93, truncated, 5.67 rad | 0/5, all held to horizon (4.89–5.76 rad) | Pass, 1 | 2/5 (1 palm at 3, 1 held to horizon, 1 drop at 2) | Palm, 4 | 0/5 (4 drops, 1 palm+drop; 3–6 controls) |

No noisy (perturbed-handoff) trials were scheduled (eval_noisy_prefixes empty). Every per-trial evaluation report has prefix_physics_bitexact true and handoff_terminated_before_policy false; every evaluation-point summary has training_rng_restored true and learner_tensors_and_counters_unchanged true. 18 evaluation trials ended by horizon truncation (5 at online 1,000, 13 at online 2,000); their return tails are unknown.

## Interpretation and scope

The variant's hypothesis was to measure seed variance of the SAC0004 failure and of the online-500 retention loss. On this single replicate both reproduce: prefix-7 goes 6/6 → 0/6 at online 500 with a large std excursion, prefix 6 never passes deterministically, and at online 1,000 the deterministic policy holds without turning at both prefixes, the same holding-not-turning behaviour SAC0004 showed at online 1,000. Where SAC0004's stochastic prefix-6/7 trials at online 2,000 dropped or braced (0/5 and 0/5, 13–80 controls), this run's twelve prefix-6/7 trials at online 2,000 all held to the horizon; that is holding retained longer, not a turn. The two stochastic successes at online 1,000 (seeds 21362 and 21373) are the first post-baseline frozen prefix-6 or prefix-7 successes under the SAC0004 settings (SAC0003 and SAC0004 evaluations hold none beyond their baseline prefix-7 retention); they are 1/5 each, absent from the deterministic trials, and the same seeds hold without turning at online 2,000, so they show the turn is reachable by sampling from the online-1,000 policy and nothing more. Sibling variants of this family that change other factors are assessed in their own findings. Prefix-8 retention was 6/6 at baseline, 4/6 at online 500, 6/6 at online 1,000 and 3/6 at online 2,000; SAC0004 scheduled no prefix-8 trials, so there is no counterpart. Cold trials all failed: the online-1,000 cold deterministic trial held for 100 controls at 2.73 rad, the others dropped or braced within 1–6 controls; no cold competence exists, so no checkpoint here is deployment-eligible.

Single seed per configuration; two seeds (SAC0004 and this run) show the outcome in kind recurred under a second seed and do not bound the variance. All training episodes are assisted (six replayed controls) and demonstration-mixed, so none are primitive-eligible; changing-policy training episodes are not evidence of a frozen policy. No checkpoint from this run improves on the assisted competence verified for SAC0001.

Inputs and protocol: `protocol.json`, `execution_manifest.json`, `input_bindings.json`; results: `summary.json`, `warmup_record.json`, `metrics.jsonl`, `episodes/`, `evaluations/`, `checkpoints/`, `demonstration_calibration_*.json`, `warmup_calibration_*`, `heldout_calibration_*`, `unfinished_suffix/`; final hashes: `manifest.json`. Note: `README.md` still carries the pre-execution header.


_Audit note: drafted by an auditing agent from the artifact files and independently refuted-checked; 6 claim(s) corrected by the verifier before writing._
