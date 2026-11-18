# Variant 22 `bc_first_lambda3`; prepared, not executed

Weaker first-row supervision (lambda 3) to test whether lambda 10 over-constrains the mean at the handoff.

Settings that differ from the SAC0004 baseline: `{"demonstration_include_mpc": false, "actor_bc_lambda": 3.0, "actor_bc_rows": "first", "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000]}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds match SAC0004 (warm-up and on-policy pretraining draws reproduce it exactly when the collection settings coincide). Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0022_bc_first_lambda3/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0022_bc_first_lambda3/run.py --execute > artifacts/bootstrap_variant_0022_bc_first_lambda3/run.log 2>&1
```
