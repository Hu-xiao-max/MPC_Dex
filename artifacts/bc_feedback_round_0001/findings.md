# One feedback-data imitation round: unsuccessful

This completed diagnostic used 25 actual supervised pairs: the original 22 bounded-teacher commands and only recovery controls 7–9, which were actually selected and executed by corrected MPC. None of the six preceding actor commands was used as an expert label. This is neither a replay dataset for main-stage training nor a successful SAC policy.

The initial actor is an immutable copy of the previously polished checkpoint, SHA-256 `a3a9473e6388a4a2a870240582d1ca048dbf4a961866d35f2d1c3aa0ecd13374`. Training restored its BC optimizer state and performed exactly 12,000 additional mean-only Adam updates at learning rate 0.003, batch size 256, epsilon 0.02, seed 17001, uniformly sampling the 25 pairs with replacement. The exact batch indices, configuration, frozen numerical sources, weights, and physical traces are saved. Critics, target critics, and temperature remained bitwise unchanged. There were no SAC updates.

| Additional BC updates | Original 22 action MSE / maximum error | Three corrections action MSE / maximum error | Strict physical result | Peak force-qualified angle |
|---:|---:|---:|---|---:|
| 3,000 | 0.009399 / 0.67823 | 0.009792 / 0.45843 | Drop at control 6, 0.6 s | 2.40357 rad |
| 6,000 | 0.004177 / 0.59114 | 0.013037 / 0.50775 | Drop at control 6, 0.6 s | 1.73359 rad |
| 12,000 | 0.004859 / 0.63808 | 0.008714 / 0.44916 | Drop at control 5, 0.5 s | 0.02663 rad |

All rollouts use the original strict 40 mm physics, palm gate, and independent per-physics-step force audit. No evaluation reached a qualified full turn; consequently there was no successful actor warranting the conditional two-second continued-policy audit. The original BC metrics use batched inference and NumPy tanh. The head-refinement report additionally measures the exact singleton Torch action route; its baseline action MSE differs only at float32 rounding scale.

## One permitted frozen-feature head correction

The final 25-by-257 augmented feature matrix had full row rank 25, checked before the sole float64 minimum-norm residual solve. Its condition number was 19,295.54, with singular values from 18.99827 to 0.00098459. The correction had Frobenius norm 3,698.20 and maximum coefficient 763.90. The mean-head weight norm increased from 5.1482 to 3,697.95. This was a large interpolation correction, unlike the preceding tiny precision refinement.

After casting the single correction to the original float32 head, actual Torch action fit improved to MSE 1.247e-9 / maximum error 2.600e-4 on the original 22 pairs and MSE 2.150e-9 / maximum error 2.358e-4 on the three corrections. The ideal float64 residual MSE was 2.362e-25; float32 execution of the large head did not preserve that precision. The actual final actor dropped at control 3, 0.3 s, with peak force-qualified angle 0.001016 rad and no palm contact. Its first command differed from the original teacher by 2.084e-4; on the next physical observation, the action error was already 1.79346. Exactly the mean-head weight and bias changed; every other checkpoint payload value remained bitwise identical to the saved final-BC baseline.

The first correction state and original teacher row 7 differ by only 0.006675 in maximum observation component, but their target actions differ by 0.765486 and target latent vectors by L2 distance 5.14975. Their final frozen features are only 0.002449 apart in L2 distance. Fitting this pair therefore requires linear-head gain at least 2,102.73 along that feature difference. This quantifies the steep local mapping created by retaining both different continuations. It does not establish that the physical system requires incompatible actions at those nearby states; the separately authorized common-continuation counterfactual addresses that question.

The float32 Torch action clamp changes **zero** teacher scalars, bitwise. Metrics named `pure_clipping_action_mse` or `pure_clipping_action_max_abs_error` include NumPy float32 atanh/tanh reconstruction; their roughly 6e-8 maximum value is not a change made by clipping.

This bounded round and its one head solve are complete. No further optimizer sweep, policy rollout, or online update was run.

Evidence: `bc_feedback/report.json`, `head_polish_0001/report.json`, `fit_conditioning.json`, exact trajectories under each audit directory, `inputs_manifest.json`, and final `manifest.json`.
