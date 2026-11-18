# Variant 13 `control_reseed`; prepared, not executed

SAC0004 settings with new seeds: measures seed variance of the SAC0004 failure and of retention loss at online 500.

Settings that differ from the SAC0004 baseline: `{"seed_matched": false}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds are fresh. Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0013_control_reseed/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0013_control_reseed/run.py --execute > artifacts/bootstrap_variant_0013_control_reseed/run.log 2>&1
```
