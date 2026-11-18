# Partial bootstrap seed under no-palm validation v3

This is an actual physics replay of `artifacts/rotation_probe/orbit_0.8_s8_p1_f0.npz` using the current 45 mm incremental bootstrap environment. It contains **110 transitions, 41.31350263319776 degrees retained rotation, and zero successful full turns**. It is only a partial bootstrap seed, not a full-turn demonstration or a primitive extraction input.

Reproduction command:

```sh
.venv/bin/python scripts/prepare_bootstrap_seed.py \
  --source artifacts/rotation_probe/orbit_0.8_s8_p1_f0.npz \
  --config configs/bootstrap_incremental_seeded.json \
  --output artifacts/bootstrap_seed_0003/replay.npz
```

The generator refuses an existing output. Every source command passed the incremental-action limits and actual-physics observation comparison. The maximum observation error was `5.960464477539063e-08`; no environment termination or truncation occurred. The saved replay marks only its last transition truncated because this partial recording ends there; its final next action is unknown.

`replay.npz.metadata.json` binds the replay to the current environment and incremental policy contracts. `verification.json` records the matching contracts, exact boundary checks, source hashes, and validation name `incremental-principal-twist-with-grasp-retention-no-palm-v3`.

Replay SHA-256: `d36c01d1c766ecbc41362340303ebe17901ed0d97a4d57fd34925b9af31cf796`.

Metadata SHA-256: `ff3ac9f3736057cbcae253466786aa644cbac6d0654dbab9412f51dbcaf4434a`.

The replay bytes equal seed 0002 because this partial trajectory never encounters the newly forbidden support state; the regenerated metadata records the current source contracts. Earlier seed files, checkpoints, and archived run 0004 sources were preserved. `configs/bootstrap_incremental_seeded.json` now selects this seed. No training was launched.
