# Entropy-free demonstration targets: values fixed, no frozen success, retention still lost

**The single objective change removed the value distortion diagnosed in SAC0003, but the policy outcome is unchanged in kind:** no frozen-policy success at any checkpoint, two of 94 completed online training episodes succeeded, and prefix-7 retention fell from 6/6 at baseline to 0/6 by online 500 and stayed there. The stage completed its full budget (500 mixed critic updates, 2,000 mixed SAC updates, exit 0, 225.1 s loop). The remaining causes are therefore not the entropy term; they lie in the online actor updates on this mixed, failure-dominated prefix-6 data.

## Actual execution

- Inherited SAC0001 final checkpoint (SHA `e4a73d89…`); final counters BC 12,000 / pretraining 1,000 / online 4,000, optimizer steps critic 5,000 and actor/alpha 4,000, as prescribed.
- Warm-up: 256 frozen-actor prefix-6 episodes, 812 suffix rows, all drops, zero successes (informational).
- Demonstration buffer: the same 8 rows in two terminal episodes, verified at runtime; 77 demonstration and 179 online rows per batch. During pretraining 46–67 demonstration rows per update bootstrapped entropy-free; their mean target rose from .407 to 1.036 as the bootstrapped values converged toward the entropy-free returns (1.05–1.14).
- Online: 2,000 interactions, 94 completed assisted episodes (longer episodes than SAC0003's 228), 2 force-qualified successes (episode 34 after 9 suffix controls at cumulative update 2,306; episode 76 after 71 suffix controls at 3,438), 58 drops, 27 palm-support failures, 7 horizon truncations in which the object was held to the 100-control limit without completing the turn. The last episode remains open; online replay holds 2,812 rows. All 2,106 physical prefix controls replayed bit-exactly and never entered replay.
- Learned alpha .000305860 → .000372934 (raw-equivalent .0746).

## Value calibration (the intended effect)

| Rows | Critic RMSE before | After pretraining | Online 2,000 |
| --- | --- | --- | --- |
| 5 MPC demonstration rows (entropy-free targets) | 1.262 / 1.356 | .205 / .150 | .103 / .101 |
| 3 recorded demonstration rows | .878 / .933 | .109 / .020 | .287 / .268 |
| 812 warm-up on-policy rows (SoftSARSA targets) | .319 / .336 | .079 / .076 | — |
| 15 held-out baseline rows | .329 / .339 | .033 / .036 | — |

Demonstration-row biases after pretraining are −.125/−.093 (MPC) and −.061/+.014 (recorded), versus +11.1 (MPC) in SAC0003. On-policy calibration is the best recorded in this curriculum. The critic no longer assigns fictitious value to the demonstration actions; the unchanged online `update` never evaluates their density.

## Frozen evaluations

| Checkpoint | Prefix-6 deterministic | Prefix-6 stochastic 17501–17505 | Cold deterministic | Cold stochastic 17511–17515 | Prefix-7 retention det. / stochastic |
| --- | --- | --- | --- | --- | --- |
| Baseline | Drop, 3 | 0/5, all drops | Drop, control 1 | not scheduled | Pass / 5/5 |
| Online 500 | Palm, 11 | 0/5 (3 palm, 2 drops) | Drop, 3 | not scheduled | Fail (held 93, no turn) / 0/5 |
| Online 1,000 | Held 94 controls, no turn, 5.62 rad | 0/5 (4 held to horizon at 5.52–5.59 rad, 1 palm) | Drop, 3 | not scheduled | Fail (held) / 0/5, all held to horizon |
| Online 2,000 | Held 94, no turn, 5.56 rad | 0/5 (3 drops, 2 palm; 13–80 controls) | Held 100, 1.19 rad | 0/5, all drops | Fail (held) / 0/5 (4 drops, 1 palm; 37–74 controls) |

At online 1,000 the deterministic policy holds the object with four loaded fingers and zero palm force for the entire horizon while stalling at about 5.6 rad, roughly .35 rad beyond the handoff and .7 rad short of a turn; peak progress in these trials is 5.67–5.86 rad. By 2,000 the stochastic trials drop or brace again after 13–80 controls. The policy therefore acquired stable holding, not the turn, and lost the two-control prefix-7 completion it inherited. Cold behaviour remains a failure from the first controls.

## Interpretation and scope

Comparing SAC0003 and SAC0004, which differ only in the demonstration pretraining target, the entropy term explains the critic inflation and nothing more: retention is lost at online 500 in both, before any prefix-6 success exists to reinforce. The inherited prefix-7 competence is a narrow two-control behaviour that the online actor updates overwrite while 70% of each batch is prefix-6 failure data and 30% is 8 repeated demonstration rows; a critic that values the demonstration actions correctly (bias about −.1) is not sufficient for the actor to reproduce them from a mean that starts .96 (MPC) and .41 (recorded) away in action space with latent std .067. The deterministic distance to the MPC label at the handoff shrank to .33 by 2,000, yet the trial that starts from exactly that state stalls, so matching the first label is not sufficient either.

Untested and now the leading candidates: reducing the demonstration fraction or the actor step on demonstration states, adding prefix-7 rows to the on-policy stream (a two-phase curriculum), supervising the actor toward demonstration actions directly, and recovering from the trained policy's own successor states. The stage does not isolate these; each needs its own protocol. No policy here is a valid initialization for a later stage without that decision, and SAC0001 remains the last checkpoint with verified assisted competence. No trajectory qualifies for a primitive.

Inputs and protocol: `protocol.json`, `execution_manifest.json`, `input_bindings.json`; results: `summary.json`, `warmup_record.json`, `metrics.jsonl`, `episodes/`, `evaluations/`, `checkpoints/`, `demonstration_buffer.json`, `demonstrations_replay.npz`, `demonstration_calibration_*.json`, `warmup_calibration_*`, `heldout_calibration_*`, `unfinished_suffix/`; final hashes: `manifest.json`.
