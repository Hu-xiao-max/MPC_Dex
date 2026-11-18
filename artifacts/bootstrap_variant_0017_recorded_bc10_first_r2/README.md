# Variant 17 `recorded_bc10_first_r2`; prepared, not executed

Seed replicate of recorded_bc10_first with checkpoints every 250.

Settings that differ from the SAC0004 baseline: `{"demonstration_include_mpc": false, "actor_bc_lambda": 10.0, "actor_bc_rows": "first", "seed_matched": false, "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000]}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds are fresh. Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0017_recorded_bc10_first_r2/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0017_recorded_bc10_first_r2/run.py --execute > artifacts/bootstrap_variant_0017_recorded_bc10_first_r2/run.log 2>&1
```
