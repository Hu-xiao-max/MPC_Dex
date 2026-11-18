# Frozen SAC policy: earlier-prefix support screen

**The fixed screen stopped at prefix 6 with 0/5 stochastic completions and a deterministic failure.** All six episodes dropped after three policy suffix actions except seed 17143, which dropped after four. Prefixes 5–1 were not executed. No learning, MPC query, or additional branch was run.

| Sampling seed | Mode | Actual suffix controls | Final force-qualified angle (rad) | Peak force-qualified angle (rad) |
|---|---|---:|---:|---:|
| 42 | deterministic | 3 | 0.491118943 | 5.337720872 |
| 17141 | stochastic | 3 | -0.371340937 | 5.328028915 |
| 17142 | stochastic | 3 | 0.091622249 | 5.361113004 |
| 17143 | stochastic | 4 | 2.163106258 | 5.280865221 |
| 17144 | stochastic | 3 | 0.817588036 | 5.339470242 |
| 17145 | stochastic | 3 | 1.519582966 | 5.345016139 |

Each episode cold-reset with physical seed 42 and actually executed the same six original commands. Every initial observation/qpos/qvel/target/full integration field, every prefix transition, and all 300 prefix physics samples matched the original source bit-for-bit. No handoff state injection, progress/history rebase or command clipping occurred. Handoff time is 0.6 s, force-qualified angle 5.258153506 rad, with three loaded fingers and tilt 0.344508809 rad. The entire horizon remains 100 controls including the physical prefix.

The frozen source is the final prefix 7 SAC checkpoint, SHA e4a73d890963a70afc9337db7d29d345c38e192218a7b81dbd51cd6589232892, with true counts BC 12,000 / critic pretraining 500 / online SAC 2,000. Its learned state-dependent std and all parameters were retained. In the deterministic prefix 6 handoff state, std spans .05106–.09849; its mean action differs from the old recorded next action by max .31915, with latent KL(new||original)=156.18. This records a distribution shift at an untrained earlier state, not proof of its sole causal role.

CUDA float32 singleton (1,60), one intra/inter-op thread and TF32 off matched the source inference contract. Initial and handoff RNG states prove no policy sampling during the prefix. Separate read-only reconstruction reproduced every raw mean, std, actual policy action, log probability and final CPU/CUDA RNG state exactly. Actor/critics/targets/alpha/counters and all frozen inputs remained unchanged. All episodes truly terminated; no time-limit return tail was fabricated. Soft-return reports use actual normalized alpha .000305859925 with rewards/200, equivalently raw-reward entropy weight .061171985.

The screen executed 55 controls: 36 physical-prefix controls and 19 policy-controlled suffix controls, with 2,750 independent 2 ms physics samples. The measured audit loop took 1.612 seconds, excluding process initialization and inference reconstruction. Full initial/handoff/terminal environment snapshots, transitions/integration states, command traces, physics and RNG are retained.

**Automatic stage advance is not supported by this screen.** Prefix7 remains the previously demonstrated supported phase; zero successes in five does not establish zero probability of success at 6. A separately fixed larger support collection could distinguish very sparse positives from this small-screen result before committing to a new learning budget. The screen itself ended at its prescribed bound. It does not justify changing exploration, physics or the success gate.

All outcomes are prefix-assisted. The checkpoint has genuine SAC updates, but these trajectories cannot become cold-policy extraction data or main demonstrations. No post-success retention or continued-policy experiment was performed here. The earlier final-prefix 7 and prefix 8 successes remain valid; this result narrows their generalization to the earlier state.

Frozen inputs/protocol: `inputs_manifest.json`, `protocol.json`; complete results: `summary.json`; inference verification: `sampling_verification.json`; episode evidence: `prefix_06/`. The trained deterministic suffix was deliberately not asserted equal to the original nominal suffix. Source manifests and the predecessor audit harness are archived.

Preparation note: the first preflight lookup encountered the source top-level manifest’s intentional omission of nested files named `manifest.json`. Before any physics, preparation was corrected to verify every checkpoint-manifest entry against the root manifest; the actual checkpoint and all numerical inputs were unchanged.
