# Variant 10 `recorded_bc10_mixed_r2`; prepared, not executed

Seed replicate of recorded_bc10_mixed.

Settings that differ from the SAC0004 baseline: `{"prefix_set": [6, 7, 8], "demonstration_include_mpc": false, "actor_bc_lambda": 10.0, "seed_matched": false}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds are fresh. Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0010_recorded_bc10_mixed_r2/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0010_recorded_bc10_mixed_r2/run.py --execute > artifacts/bootstrap_variant_0010_recorded_bc10_mixed_r2/run.log 2>&1
```
