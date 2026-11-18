# Variant 04 `recorded_bc10_first`; prepared, not executed

As recorded_bc10 but the BC term covers only the handoff row (obs6 -> recorded control 7), leaving the learned prefix-7/8 completion untouched by supervision.

Settings that differ from the SAC0004 baseline: `{"demonstration_include_mpc": false, "actor_bc_lambda": 10.0, "actor_bc_rows": "first"}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds match SAC0004 (warm-up and on-policy pretraining draws reproduce it exactly when the collection settings coincide). Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0004_recorded_bc10_first/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0004_recorded_bc10_first/run.py --execute > artifacts/bootstrap_variant_0004_recorded_bc10_first/run.log 2>&1
```
