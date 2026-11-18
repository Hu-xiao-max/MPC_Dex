# Fixed descending-prefix support screen

The fixed screen stopped at prefix 7: **2/5 stochastic assisted suffixes completed a strict, independently force-qualified turn**. Prefixes 6 through 1 were not executed. The deterministic suffix reproduced the original full nine-control successful trace bit-for-bit. No learning or MPC query occurred.

| Seed | Mode | Suffix controls | Outcome | Final force-qualified angle (rad) | Observed discounted soft return |
|---|---|---:|---|---:|---:|
| 42 | deterministic | 2 | strict success | 6.651565369 | 219.609334 |
| 17041 | stochastic | 7 | drop | 2.902410708 | -63.191136 |
| 17042 | stochastic | 2 | strict success | 6.667173026 | 220.825213 |
| 17043 | stochastic | 7 | palm_support | 2.546527061 | -90.602805 |
| 17044 | stochastic | 2 | strict success | 6.772404110 | 222.591273 |
| 17045 | stochastic | 3 | drop | 4.313477808 | -48.435331 |

Every episode began with a real cold reset and the same seven recorded controls. There was no simulator state injection, time reset, progress rebasing or command clipping at the handoff. The initial observation/qpos/qvel/targets/full integration state, all seven transition rows, and all 350 independently observed 2 ms physics rows match the source exactly. The handoff force-qualified angle is 5.455151260798702 rad, with three loaded fingers and tilt 0.10836057649836096 rad. The source prefix retains the original full task history. The total horizon is 100 controls including the prefix.

The same immutable bootstrap0005 initialization supplied the 60D absolute-target actor: BC count 12,000, critic pretraining 0, online SAC updates 0, latent std 0.05. Actual suffix inference used CUDA float32 batches of shape (1,60), one intra/inter-op thread, TF32 disabled. RNG states prove that no policy sample was drawn before handoff. A separate read-only reconstruction reproduced every raw mean, latent std, actual policy action, sampled log probability and final CPU/CUDA RNG state bit-for-bit. All actor, critic, target-critic parameters, alpha and learner counters remained unchanged.

All six episodes truly terminated, so the reported discounted soft returns have no unmeasured time-limit tail. They exclude the first suffix action entropy and include subsequent sampled-action entropy with frozen alpha and discount. They are measured suffix returns, not critic predictions or causal evidence that SAC will improve.

The experiment executed 65 actual controls: 42 recorded-prefix controls and 23 policy-controlled suffix controls, with 3,250 physical samples. The measured audit loop took 1.938 seconds, excluding process/CUDA initialization and archival verification. Full initial/handoff/terminal environment snapshots, transition and integration arrays, actual commands, 2 ms force audits, RNG states and frozen numerical sources are retained.

The preceding independently completed prefix-8 probe had 5/5 stochastic first-action completions with these same five seeds. This paired phase screen supports **prefix 7 as the next bounded SAC training candidate** because it supplies both positive and failed sampled outcomes, while prefix 8 can serve as an assisted retention check. The four-of-five threshold is an operational selection rule, not a reliability estimate. Five selected, reused seeds do not establish the general success probability; later training evaluations must use new held-out seeds.

Every completion in this screen is prefix-assisted and BC-origin, including the deterministic completion. None is an unassisted SAC-policy turn, a successful extraction trajectory, a main-stage demonstration, or an eligible primitive. No post-success hold or continued-policy behavior was tested.

Frozen protocol: `protocol.json`; launch input identities: `inputs_manifest.json`; complete results: `summary.json`; inference reconstruction: `sampling_verification.json`; per-episode proof: `prefix_07/`. The referenced prefix-8 protocol and summary are archived under `prefix8_reference/`.
