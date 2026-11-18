One numerical mean-head correction substantially improves the final BC actor's recorded-state precision, but **does not produce a qualified full turn in closed loop**.

The frozen singleton-inference feature matrix, augmented with a bias column, has shape 22×257, full row rank 22, and condition number **22.2133**. Its largest/smallest singular values are 24.4126/1.09901. A single float64 minimum-norm least-squares solve fits the existing float32 pre-tanh target residual. The correction is cast to float32 and added once to the original mean-head weight and bias; maximum actual weight change is 1.61e-6. No optimizer, critic, SAC, or additional fitting update is performed.

| Exact Torch singleton action metric | Original 12k BC | Refined mean head |
|---|---:|---:|
| Teacher action MSE | 1.990e-12 | 1.848e-15 |
| Maximum teacher action error | 1.350e-5 | 2.086e-7 |
| Initial-state action error | 1.350e-5 | 1.490e-7 |

The one independent refined-policy rollout reaches **5.265992 force-qualified radians**, then drops at **1.5 seconds** without palm contact. Its final force-qualified angle is −.034557 rad. The unmodified baseline hit the palm at .4 seconds after peak progress 4.142790 rad. The correction therefore improves this trajectory's progress and duration, while the full-turn criterion remains unmet. No post-success continuation is applicable.

On actual visited states, maximum action error against the time-aligned teacher grows from 1.490e-7 on control 1 to 9.775e-6 on control 2, 1.049e-4 on control 5, .0009226 on control 6, and .0113508 on control 8. This occurs despite near-float32 fitting accuracy on all recorded states. Precision matters, but recorded-state interpolation alone is insufficient to establish stable learned feedback here.

`proof.json` compares every saved checkpoint payload field. **Only `actor/mean_head.weight` and `actor/mean_head.bias` change.** The trunk, std head, critics, target critics, alpha, optimizer states, configurations, and update counters remain bitwise unchanged. The original baseline file hash is unchanged, and an immutable baseline copy is included. `solve_inputs_and_result.npz` contains the exact features, targets, residual, singular values, correction, and before/after parameters. `actual_torch_teacher_predictions.npz` records the actual policy inference route; these Torch action metrics are authoritative at float32 precision, rather than NumPy tanh reconstruction metrics.

The entire experiment imports archived numerical sources and assets from the parent artifact. Concurrent live CLI changes are unrelated and unused. `polished.pt` is a diagnostic BC actor with one numerical refinement; it is **not paper BC, a SAC-trained policy, or a production primitive**. No second solve or optimizer sweep was run.
