# Fixed-seed exploration at the existing minimum standard deviation

All five prescribed samples failed before a full turn. Each incurred palm support; seed 17023 also dropped. This is a five-sample diagnostic, not a robustness estimate or SAC result. No learning or additional noise levels were run.

The unchanged BC-plus-refined-head mean was loaded from the finalized checkpoint with SHA-256 `bc00ed2358e2d50e33d9634da22e8bb350bda149b56644c4dfad268bfbddd77b`. Only the in-memory `log_std_head.bias` changed. The requested standard deviation is the archived `LearnerConfig.min_std`, 1e-5; effective float32 latent standard deviation is 1.0000003385357559e-5. Mean/trunk tensors, critics, target critics, alpha, and historical update counters remain unchanged. Newly performed BC, critic, and online updates are all zero.

Inference uses singleton float32 CUDA, one Torch thread and one interop thread, with TF32 disabled, exactly as the preceding .05 diagnostic. Each episode uses the prescribed Torch/CUDA/NumPy/Python seed and a cold reset. The control limit remains 100, with unchanged strict success/drop/palm termination, force, retention, and tilt rules. There is no action smoothing or extra clipping. Physics and force auditing retain every 2 ms transition, including the terminal control interval.

| Seed | Controls | First palm support (s) | Maximum force-qualified angle (rad) | Peak palm normal force (N) | Outcome |
|---|---:|---:|---:|---:|---|
| 17021 | 5 | 0.402 | 3.949042518 | 12.439398551 | Palm support |
| 17022 | 4 | 0.318 | 3.696822274 | 22.969675100 | Palm support |
| 17023 | 10 | 0.904 | 4.876613249 | 10.806114850 | Palm support; also drop |
| 17024 | 8 | 0.736 | 4.603562600 | 4.981438791 | Palm support |
| 17025 | 5 | 0.476 | 3.852320242 | 0.911739287 | Palm support |

The deterministic strict success at control 9 and actual continued-policy drop at control 12 were already independently reproduced bit-for-bit in `../bootstrap_exploration_0001`. Their unchanged archived results are copied here and explicitly identified in `runtime.json`; they were not rerun. Exactly five new physical episodes were performed for this follow-up.

The original 149-file policy archive and preceding 107-file exploration archive retain their original manifests. This artifact contains copies of the original checkpoint, numerical sources, assets, commands, transition/integration states, 2 ms physics, initial/final RNG states, runtime records, and execution script. No modified checkpoint was saved.
