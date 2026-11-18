# Fixed-mean minimum-noise comparison

Five prescribed seeds (17021–17025) were executed once each, with the strict nine-control cap. Each command used the polished actor’s **raw latent mean at its saved actual deterministic trajectory observation**, then sampled noise at the archived minimum std. Live observations were recorded but never fed back into the mean. No training, replanning, extra hold, or continuation was performed.

All nine deterministic commands were reproduced bit-for-bit using singleton float32 CUDA inference, one Torch thread, and TF32 disabled. The requested std was 1e-5; the exact float32 exp(log(std)) used in both experiments was 1.0000003385357559e-5. Every archived feedback action, conditional mean/std and final CPU/all-CUDA RNG state was reconstructed exactly from its saved initial RNG state. Directly generated standardized Gaussian draws matched for every shared command (5, 4, 9, 8, and 5 respectively); no inverse tanh estimate was used.

| Seed | Fixed-mean final / peak retained rad | Fixed-mean result at control 9 | Existing feedback failure control | Max fixed-mean action perturbation | Max feedback action deviation* |
| --- | ---: | --- | ---: | ---: | ---: |
| 17021 | 3.683945 / 5.389528 | partial; no drop/support | 5 | 2.19e-05 | 0.197 |
| 17022 | 5.646855 / 5.646855 | partial; no drop/support | 4 | 2.67e-05 | 0.0159 |
| 17023 | 4.559690 / 5.275812 | partial; no drop/support | 10 | 1.91e-05 | 0.902 |
| 17024 | 5.413047 / 5.413047 | partial; no drop/support | 8 | 3.06e-05 | 1.67 |
| 17025 | 4.524553 / 4.867103 | partial; no drop/support | 5 | 2.1e-05 | 0.0689 |

*Actions are normalized absolute joint targets. Deviations are from the corresponding noiseless actual deterministic commands, over recorded shared steps. All existing feedback terminations were palm-support failures. Seed 17023 fails at control 10, outside this experiment’s nine-control cap; its matched nine-control prefix has no palm violation.

In every paired trial, the first sampled action and resulting observation/qpos/qvel were bit-identical. By command 2, feedback and fixed-mean actions differed by 0.000193–0.004213, although the largest fixed-mean action perturbation over all commands was only 0.00003059. Thus recomputing neural means on perturbed states materially amplifies the command changes in these recorded trajectories.

Direct physical sensitivity also remains: holding the means fixed did not reproduce the deterministic full turn in any prescribed trial. All five stopped as partial at the cap, with 3–4 loaded fingers and no palm/table/other support or drop. These are paired mechanism examples, not a learned-policy result or a robustness-rate estimate. The experiment does not establish that removing feedback yields a successful controller.

Frozen source/checkpoint/input hashes, exact raw means and noiseless commands are archived. Each seed directory contains standardized draws, sampled latents/actions, visited observations, qpos/qvel, complete physics, RNG states and force-audit results. comparison.json records detailed paired action differences. All original inputs remain unchanged.
