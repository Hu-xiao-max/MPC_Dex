# Independent suffix-support review

No material integrity issue found. All 104 source-manifest entries and all original input bindings verify. This review ran frozen inference and read saved physics only: **zero new physics steps and zero optimization updates**.

The checkpoint is exactly bootstrap0005 initialization, SHA256 `66681b0f9651b21edb39ef94cb9827e8dcb1ab08249ac7e82010f81c114cf937`: 60D absolute joint-target interface, historical BC12,000, critic pretraining0, online/SAC0, initialized latent standard deviation .05. Its actor/critic/temperature optimizer states are fresh. Source and local configs differ only in a byte-identical initial-grasp path.

For each of the six episodes, all eight recorded prefix transitions, full integration boundaries, initial state, and 400 physics samples match the source exactly. The deterministic nine-control episode matches all source transition fields and all450 physics samples. The policy-control mask is false for the prefix and true only for control9. No reconstructed or reset-injected handoff replaces the actual prefix execution in the archived driver.

For stochastic seeds17041–17045, I restored the saved initial CPU/all-CUDA RNG states, independently verified their declared seeds, and repeated the exact frozen actor calls on saved observations. Calls use float32 CUDA `(1,60)`, one intra/inter-op thread, highest matmul precision, both TF32 flags off. Every conditional mean, latent standard deviation, sampled action, and directly returned log probability is bit-identical. The full final RNG states also match. Prefix mean evaluations consume no policy RNG and their saved densities remain NaN; no fictitious prefix densities are assigned. The deterministic branch consumes no sampling draws.

The handoff has 5.208014490 force-qualified radians. All five sampled suffixes terminate on their first policy action, with final force-qualified angles 6.630435593–6.705910615 radians. I independently recomputed qualification from every2ms twist, loaded-finger count, tilt and sticky palm flags, and reconstructed the actual raw-angle rewards. Each episode has450 samples, matching9controls×50steps. All are true terminals, so the one-step suffix Q return is exactly its reward; it excludes current-action entropy and needs no invented tail/bootstrap.

| Seed | Final force-qualified radians | Loaded fingers | Actual suffix reward | Actual log probability |
| --- | ---: | ---: | ---: | ---: |
| 17041 | 6.683531315 | 4 | 229.210336503 | 41.661651611 |
| 17042 | 6.697374191 | 4 | 229.487194016 | 48.473072052 |
| 17043 | 6.705910615 | 4 | 229.657922503 | 44.123046875 |
| 17044 | 6.630435593 | 4 | 228.148422051 | 45.574882507 |
| 17045 | 6.683805931 | 4 | 229.215828815 | 46.192459106 |

These are **five prescribed prefix-assisted samples**, not a robustness-rate estimate, cold-reset stochastic-policy success, or SAC success. Eight controls are replayed BC-origin commands. No post-success continuation was tested. The unchanged policy/critic/target/temperature tensors and zero learning counters were verified; primitive/main-stage eligibility remains false.

Detailed assertions, reconstructed inference arrays, runtime and source bindings are saved beside this report.
