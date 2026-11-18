# Fixed-standard-deviation exploration samples

The original BC-plus-head-refinement checkpoint is unchanged. Its strict deterministic nine-control full turn and actual continued-policy drop at control 12 replay exactly under singleton CUDA inference, one Torch/inter-op thread, and TF32 disabled. This policy has 12,000 historical BC updates and zero critic/online SAC updates.

An in-memory copy sets the latent Gaussian standard deviation to .05, an explicit independent bootstrap choice. Only log_std_head.bias changes; mean, trunk, critics, optimizer state and counters are preserved. No modified checkpoint was saved. Actual float32 standard deviation is 0.05000000074505806. Each sample starts from the same verified cold grasp, with its prescribed seed controlling Torch sampling and environment reset. There is no extra action clipping or smoothing.

| Seed | Controls to failure | Failure | Peak force-qualified radians |
|---|---:|---|---:|
| 17021 | 6 | drop | 0.063695603 |
| 17022 | 5 | drop | 0.112586718 |
| 17023 | 3 | palm_support | 0.225148920 |
| 17024 | 3 | palm_support | 0.744157178 |
| 17025 | 7 | drop | 1.349482820 |

None of these five prescribed samples completes a full turn. This is evidence of poor initial exploration support at this one standard deviation in these samples; it is not a robustness rate estimate, a standard-deviation sweep, or a learned SAC result.

Each audit directory contains actual observations/actions/conditional means/standard deviations, rewards, terminal flags, qpos/qvel/targets, full integration states, initial state, every 2 ms force/rotation sample, command reports, and initial/final Torch RNG state. runtime.json binds the numerical sources, exact original checkpoint, model contract and device configuration. The source checkpoint and frozen code/assets are copied here; the parent 149-file manifest remains unchanged.
