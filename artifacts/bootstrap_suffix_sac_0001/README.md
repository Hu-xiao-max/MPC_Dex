# Prepared prefix-7 suffix SAC diagnostic

This document is the **pre-launch protocol snapshot**: no execution had occurred when it was written. Later execution status belongs to `runtime.json`, `latest.json`, and `summary.json` (or `failure.json`); their absence means the runner has not started. Root selected the single prefix-7 stage from `bootstrap_suffix_rollback_0001`; its source actor had two successful stochastic suffixes out of the five fixed screen trials. Prefix 8 had five successful one-action suffixes. These are assisted conditional-support observations, not cold-start success rates.

The runner imports only copied source, initializes from the exact bootstrap0005 initial 60D actor/critic checkpoint (BC 12000, critic pretraining 0, SAC 0), and physically executes the seven immutable recorded commands at every cold reset. The entire prefix and suffix are observed at 2 ms. Only actual policy suffix transitions enter replay. The physical horizon remains 100 controls including the prefix; task progress, sticky failure state, time, action history and finite-difference observation context remain intact.

The fixed sequence is 256 complete frozen-actor stochastic suffix episodes with sampling seed 17081, then 500 actual SoftSARSA critic-only updates, then 2000 actual policy suffix interactions with one existing SAC update each. Replay uses uniform transition sampling; raw environment rewards remain in episode records and replay. Only each sampled update batch has rewards divided by 200. Fresh initial alpha is .05/200 and SoftSARSA alpha is .02/200; all other source hyperparameters remain unchanged, including std .05, two critics, actor/critic learning rates .0003, actor warm-up 5000 and batch size 256. The actor remains bit-exact through the warm-up and critic pretraining. Temperature is learned during online updates by the existing method.

Reward/temperature scaling preserves the intended relative reward and entropy weight; it is not bit-exact algorithm invariance. Curriculum and scaling change together, so this is not a pure causal test of the curriculum. Warm-up and learned online outcomes have separate counters. Environment rewards are never rewritten when the independent force audit fails. Claims of qualified success require both environment success and the independent force/no-support/no-drop criteria.

Each policy action records the BC/pretrain/online counters at inference, before its following update. Recorded-prefix commands carry `-1` counter sentinels and `policy_controlled=False`. Thus even the first online action can be recognized as preceding any SAC update. An online training episode can span changing policy versions and never establishes frozen-policy success by itself.

Evaluations use separate environments and preserve CPU/CUDA Torch, NumPy and Python training RNG. Baseline before critic pretraining and online checkpoints 500, 1000 and 2000 each receive one deterministic prefix-7 evaluation, five stochastic prefix-7 evaluations (17101–17105), and one deterministic cold evaluation. Baseline and final additionally receive one deterministic and five stochastic prefix-8 retention evaluations (17121–17125). Only the final checkpoint receives five stochastic cold evaluations (17111–17115). Evaluation transitions never enter replay. There are no post-success continuations or automatic stage changes.

Complete terminal warm-up trajectories provide empirical soft-Q targets at both alpha .02/200 (pretraining objective) and .05/200 (initial online objective), excluding current-action entropy and terminal bootstraps. Truncated returns have unknown tails and are excluded. The same held-out baseline prefix-7 stochastic trajectories are compared before/after critic pretraining without fitting them. Raw equivalent and normalized values are labeled separately.

The unchanged SoftSARSA implementation evaluates recorded next-action density with an inverse tanh and a 1e-6 action clamp. Empirical return calibration uses the direct density returned at sampling; this numerical distinction is preserved and documented rather than modifying the learner.

Checkpoints save learner/optimizer/counters, unmodified replay and its sampler RNG, global RNG, exact MuJoCo binary and integration states, Python episode fields/RNG, observations and observer state, current episode transitions and physics. They are published only after a complete physical transition and its update. An unfinished final suffix stays open with its genuine nonterminal flags and no invented next action. This runner preserves exact evidence but does not implement automatic resume. An unexpected failure leaves prior checkpoint generations intact and writes failure evidence; it does not publish partially updated learner state.

Every assisted episode is diagnostic and ineligible for primitive export. A later genuinely trained 60D checkpoint would require a separate frozen cold-policy collector before any primitive claim; its BC origin does not permanently prohibit that future collection. This artifact exports no primitive trajectories and does not satisfy the main 20-demonstration gate.

Preparation and execution are separate:

```bash
.venv/bin/python artifacts/bootstrap_suffix_sac_0001/prepare.py
# Only after code review and rollback binding:
.venv/bin/python artifacts/bootstrap_suffix_sac_0001/prepare.py --finalize-prefix 7
# A distinct explicit execution instruction is still required:
.venv/bin/python artifacts/bootstrap_suffix_sac_0001/run.py --execute
```

`test_protocol.py` uses synthetic schema/math fixtures only. It runs no policy, learner update or MuJoCo physics and provides no success evidence.
