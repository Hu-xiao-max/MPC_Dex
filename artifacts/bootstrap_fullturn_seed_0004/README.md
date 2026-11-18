# Verified common-continuation demonstration

This is one physically replayed, independently force-audited 40 mm MPC diagnostic episode. It contains the original bounded teacher's first six commands followed by the three common recovery commands. It is not a SAC demonstration, production primitive, or main-method demonstration dataset. Preparation did not launch training.

The nine actual controls complete a turn at 0.9 simulated seconds: force-qualified angle 6.649846605355571 radians, four loaded fingers, tilt .06119702443975995 radians, and zero palm/table/other/applied support. All recorded observations, next observations, qpos and qvel replay exactly. No source controls are unused.

The source's separate two-second fixed hold remains grasped, ending at 7.5735683677319745 force-qualified radians with at least three loaded fingers throughout. The hold is not included in this terminal replay episode and is not continued learned-policy control. Its audit is in `../mpc_polished_actor_recovery_0001/nominal_prefix_corrections_0001/`.

Preparation command:

```sh
uv run --frozen python scripts/prepare_bootstrap_demonstrations.py \
  --source artifacts/mpc_polished_actor_recovery_0001/nominal_prefix_corrections_0001/strict_audit/transitions.npz \
  --config artifacts/bootstrap_fullturn_seed_0004/seed_config.json \
  --output artifacts/bootstrap_fullturn_seed_0004/replay.npz
```

The config is an unchanged copy of seed0002's absolute-target, 40 mm, strict environment configuration. The loader verifies the environment and interface contracts. Eight successor actions are actually recorded; the terminal next action is invalid. The float32 stored actions match physical execution and survive the learner's epsilon .02 clamp bit-for-bit. `verification.json` records these checks; `replay_audit/` contains source snapshots, all transitions, physics samples and independent force measurements.

Replay SHA256: `fdfc5087956d8ec534a323333281e2bd36e02fe9f8605b9289630ec3042eed41`.

Metadata SHA256: `8f535bc295a6ebce37568cee5aa4a5e8fd013320b126409d983b88b23e533de1`.

Demonstration successes remain separate from SAC successes, online interactions and exported policy trajectories.
