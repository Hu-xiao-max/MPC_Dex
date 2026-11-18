# Variant 03 `recorded_bc10`; prepared, not executed

Supervised pull (lambda 10, all three recorded rows) toward the unique recorded control 7, whose successor state the actor already solves 6/6; prefix-6 deterministic pass expected, stochastic limited by the narrow basin.

Settings that differ from the SAC0004 baseline: `{"demonstration_include_mpc": false, "actor_bc_lambda": 10.0}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds match SAC0004 (warm-up and on-policy pretraining draws reproduce it exactly when the collection settings coincide). Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0003_recorded_bc10/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0003_recorded_bc10/run.py --execute > artifacts/bootstrap_variant_0003_recorded_bc10/run.log 2>&1
```
