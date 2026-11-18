# Frozen bootstrap_0005 value calibration

The imported deterministic BC mean completed a strict full turn at initialization. After 1,000 and 2,000 environment interactions, the deterministic policies failed. All fifteen prescribed stochastic audit episodes failed. No SAC-success claim is supported by these audits or the trainer exports.

Each checkpoint was copied from an immutable, hash-verified committed generation. Audits used the unchanged frozen actor, both critics, learned alpha, and learned standard deviation: singleton float32 CUDA, one Torch thread, one interop thread, TF32 disabled. No parameter, optimizer, reward, action mapping, or physics setting was changed; no learning occurred in these audits.

| Environment steps | Online SAC updates | Deterministic strict outcome | Stochastic outcomes, five seeds | Maximum stochastic force-qualified angle |
|---:|---:|---|---|---:|
| 0 | 0 | BC-origin success at 9 controls; 6.651565 rad max | 3 drops; 2 palm violations; 0 full turns | 1.148294 rad |
| 1000 | 745 | drop at 2 controls; 1.175104 rad max | 5 drops; 0 palm violations; 0 full turns | 1.219663 rad |
| 2000 | 1745 | palm_support at 7 controls; 2.098150 rad max | 5 drops; 0 palm violations; 0 full turns | 2.874495 rad |

The five seeds are 17041–17045 at each freeze. These are fixed-seed diagnostics, not a robustness-rate estimate. Initial success has zero online updates and retains its BC origin. Trainer checkpoint state reports no successful trajectory exports at any audited freeze, so there are no exports to promote as learned SAC primitives.

Critic calibration uses **the actual first action of each episode**. For frozen alpha and discount gamma=.995, the sampled soft-Q return is `sum(gamma**t * reward[t]) - alpha * sum(t>=1, gamma**t * logpi[t])`. The initial action entropy term is excluded to match the implemented SAC Q backup. Full state-value return including `-alpha*logpi[0]` is saved separately. Log probabilities come directly from `actor.sample`; there is no inverse-tanh reconstruction of rounded actions.

| Environment steps | Mean initial Q1 | Mean initial Q2 | Mean discounted reward | Mean realized soft-Q return | Mean Q1 bias | Mean Q2 bias |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.385683 | 0.054043 | -69.286108 | -74.593765 | 74.979448 | 74.647807 |
| 1000 | 33.661839 | 34.746263 | -31.873681 | -34.093475 | 67.755314 | 68.839738 |
| 2000 | 14.422121 | 16.128823 | -42.381114 | -53.628073 | 68.050195 | 69.756896 |

Positive initial-state Q estimates overpredict these sampled soft returns. Bias does not grow monotonically: the five-sample means remain roughly +68 to +75. At 1,000 steps, initial Q estimates peak near +34 while sampled soft returns average −34.09; at 2,000, Q means are +14.42/+16.13 while sampled soft returns average −53.63. This differs from training-log averages over mixed replay states, which may be negative.

All 18 episodes end with genuine terminal flags; there is no missing time-limit continuation here. For any future nonterminal horizon cutoff, the omitted `gamma**T * V_soft(s_T)` tail must remain explicit rather than be treated as zero. Deterministic returns are behavior diagnostics and are not stochastic-policy Q samples. Ensemble min/mean/max values are always for the same executed action; the audit performs no maximization over actions.

Policy movement is evaluated on ten fixed observations: the cold reset plus the nine archived common-BC successful observations. Initialization exactly preserves the imported mean/trunk; only configured standard deviation changes from .3 to .05. Compared with the initial training checkpoint:

| Environment steps | Cold-reset latent std range | Fixed-observation latent std range | Max normalized mean-action change | Cold-reset max mean-action change |
|---:|---|---|---:|---:|
| 1000 | 0.114485–0.126411 | 0.061884–0.126711 | 1.469441 | 1.227381 |
| 2000 | 0.055690–0.361853 | 0.055690–0.595504 | 1.360196 | 1.072636 |

The configured 5,000-update actor warmup ramps the learning rate from a small positive value; it does not freeze the actor. These observations establish parameter/behavior drift, but do not isolate a single cause.

Independent reconstruction from saved initial RNG states reproduces every recorded action, actual log probability, both critic/target-critic outputs, and final CPU/CUDA RNG states bit-for-bit. Discounted-return calculations were separately checked. Every control-boundary qpos/qvel/integration state exactly matches its corresponding 2 ms physics record. The initial deterministic transition archive also matches the original successful BC audit bit-for-bit. All files are retained, including checkpoint/config/source copies, runtime, observations, actions, rewards, terminal flags, means/stds/logprobs/Q estimates, complete physics, and RNG snapshots.
