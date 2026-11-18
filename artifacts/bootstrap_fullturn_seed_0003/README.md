# Strict first-success prefix from the slew diagnostic

This is one physically executed and independently force-audited MPC diagnostic demonstration from `artifacts/mpc_matched40_slew01_0001/transitions.npz`. It is not a SAC policy success, production primitive, or the main method's MPC dataset. No training was launched.

Preparation used the source's first **17 controls**, ending at its first strict full-turn success after 1.7 simulated seconds. The remaining 43 controls were unused. Final raw and retained rotation were `6.424931099585724` radians, and independent force-qualified rotation was `6.39237056055268` radians. Four fingers contacted the object, final tilt was `0.11945313802773075` radians, and there was no drop or palm support. Every supplied observation, next observation, qpos, and qvel replayed with zero error.

The replay has one terminal episode, 17 transitions, 16 known successor actions, and an invalid terminal next action. The loader's contracts, dimensions, source/hash, boundary, and audit-evidence checks passed. Stored actions exactly match the independent audit. Maximum absolute normalized action is `0.9799998998641968`; the default BC clipping epsilon of `0.02` changes no actions.

`seed_config.json` preserves seed 0001's strict absolute-target 40 mm configuration. Preparation ran after the parent recording passed its independent exact replay and final two-second hold audit:

```sh
.venv/bin/python scripts/prepare_bootstrap_demonstrations.py \
  --source artifacts/mpc_matched40_slew01_0001/transitions.npz \
  --config artifacts/bootstrap_fullturn_seed_0003/seed_config.json \
  --output artifacts/bootstrap_fullturn_seed_0003/replay.npz
```

`replay_audit/` preserves the input archive, actual transitions and initial state, every physics-step audit sample, commands, reports, and sources. `verification.json` records the loader, boundary, action and clipping checks. Earlier seeds were preserved.

Replay SHA-256: `ff3b721a97c14d1d5bb4d4984162dcc956b94371d4d2d0023f4425cd2a6ba372`.

Metadata SHA-256: `ed7fbab62d0996b730171d4b77cbfc964ceb18f5d8cfc7a2450b3ac273e887c2`.
