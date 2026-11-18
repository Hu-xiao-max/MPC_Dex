# Frozen SAC trajectories for the rotation primitive

The main pipeline requires five successful, independently force-audited episodes from one frozen simulation SAC checkpoint on the matched 40 mm hexagonal object. Use the dedicated collector after training produces that checkpoint:

```bash
uv run --frozen python scripts/collect_policy_trajectories.py \
  --checkpoint /path/to/agent.pt \
  --config /path/to/matching_config.json \
  --output artifacts/frozen_collection_NEW \
  --device cuda:0 --seed 1000000
uv run --frozen python -m mpc_dex extract-primitive \
  --trajectory artifacts/frozen_collection_NEW/successful_joint_trajectories.npz \
  --output artifacts/rotation_primitive_NEW.npz
```

The output directory must be new. The configuration must match the checkpoint's complete environment contract, use 2 ms physics and terminate on the first successful full turn. Deterministic singleton inference is the default; `--stochastic` samples the frozen distribution. The command sets one Torch thread, highest float32 matmul precision and disabled TF32, and records the actual runtime. These are local collection choices; they do not establish the unpublished authors' bootstrap protocol.

Before execution the collector copies the checkpoint, metadata, source, configuration and assets. It verifies actual positive SAC update counts from the checkpoint payload; a BC-only checkpoint cannot enter this collection. Each of the five actual attempts saves observations, executed actions, rewards, terminal flags, full control-boundary states, initial and terminal measured joints, reset-relative timestamps, every 2 ms physics sample and an independent force audit. An interruption or failed attempt is retained. The eligible trajectory archive is written only when all five episodes satisfy the existing full-turn, retention, force-bearing-finger, no-palm, no-external-support and no-drop checks. Exit code 2 means the collection did not qualify; it does not erase the failed episodes.

Extraction verifies the copied checkpoint and audit file hashes, five-episode continuity, actual learner counters, measured frame/timestamp identity, physics coverage and force-supported rotation. Main MPC collection and online training recheck that evidence and the resulting 90-percent PCA arrays before planning. Generic `PrimitiveBasis` and `planner_for` remain available for clearly labeled numerical and diagnostic work.

Temporal alignment currently means preserving the original samples relative to each reset. There is no interpolation or duration padding. Individual durations, identical resets and identical trajectories are reported. Five deterministic repetitions from one cold reset are not a robustness estimate, and local durations are not represented as the companion project's ambiguous approximate minute. The authors' temporal alignment algorithm remains unavailable.

Ordinary `python -m mpc_dex evaluate` outputs and bootstrap training histories remain diagnostic. Their metadata reports actual BC, critic-pretraining and SAC update counts and marks them ineligible for primitive extraction. A successful BC origin trajectory or a set collected while the policy changed cannot replace the frozen SAC prerequisite. Existing historical artifacts are unchanged.

The collector's tests use explicit synthetic fixtures to exercise serialization and corruption rejection, plus short real failed physics episodes. Synthetic fixtures provide no task-success evidence. At the completion of bootstrap0005 there is still no qualified SAC collection, so the main 20-demonstration and learning stages remain pending.

Verification after integration: the full suite passes 247 tests. A real collection from bootstrap0005's final checkpoint completed all five attempts, each failing by palm support at control 7. Every saved control/state and all 350 physics samples per episode match the earlier independent audit bit-for-bit; no eligible archive was created. See [the integration findings](../artifacts/bootstrap_0005_frozen_collection_0001/findings.md). The stricter state and audit-identity checks subsequently passed the focused regression and full suite; no physical success is inferred from those checks.
