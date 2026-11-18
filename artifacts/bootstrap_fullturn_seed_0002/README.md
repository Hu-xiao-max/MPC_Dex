# Verified bootstrap demonstration with bounded commands

This is one physically executed and independently audited 40 mm MPC diagnostic demonstration. It is not a SAC policy demonstration, production primitive, or the main method's 20-episode dataset. No training was launched.

The source is `artifacts/mpc_matched40_bound098_0001/transitions.npz`. Preparation used its first **22 controls**, ending at the first genuine strict full-turn success after 2.2 simulated seconds. The remaining 38 of 60 controls were reported as unused and did not become extra episodes.

The final retained angle was `6.337595308501611` radians; the independently measured force-qualified angle was `6.287874648934319` radians. Three fingers bore force at the final state, with axis tilt `0.027819344860093624` radians. Peak palm, table, and other normal support forces were zero. All recorded observations, next observations, qpos, and qvel replayed with zero error.

The largest absolute normalized action is `0.9799998998641968`. The learner's default BC clipping epsilon of `0.02` leaves every action unchanged. This checks the clipping operation only; it does not establish that a trained policy will reproduce the trajectory.

`seed_config.json` is an exact copy of seed 0001's strict absolute-target 40 mm preparation config. Preparation command:

```sh
.venv/bin/python scripts/prepare_bootstrap_demonstrations.py \
  --source artifacts/mpc_matched40_bound098_0001/transitions.npz \
  --config artifacts/bootstrap_fullturn_seed_0002/seed_config.json \
  --output artifacts/bootstrap_fullturn_seed_0002/replay.npz
```

`replay.npz` contains one terminal episode with 22 transitions and 21 recorded successor actions. Its terminal next action is invalid, and no transition is truncated. The loader's environment, policy, shape, hash, episode-boundary, and audit-evidence checks passed. Stored replay actions exactly match both the physical execution and independent audit; the source was already float32, so preparation changed no action values.

`replay_audit/` preserves source snapshots, the input recording, actual executed transitions and initial state, every physics-step force measurement, and the audit commands/report. `verification.json` records the clipping, boundary, source-preservation, and contract checks.

Replay SHA-256: `69f74bed08e92fdbd1f41c4a174cd4c37ce5658ff5e6936e5555ddeea2945014`.

Metadata SHA-256: `fdebbf4b8c686b08ada50a009e1432f190a8498b11da2d9e351614567140e58a`.

Seed 0001 and its metadata were preserved. Demonstration successes remain separate from SAC successes, online steps, and SAC trajectory exports.
