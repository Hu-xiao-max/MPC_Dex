# Variant 40 `bc_first_l100_std01_alphafreeze`; prepared, not executed

Strong mean pin, tight spread anchor and frozen temperature.

Settings that differ from the SAC0004 baseline: `{"demonstration_include_mpc": false, "actor_bc_lambda": 100.0, "actor_bc_rows": "first", "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "eval_stochastic_trials": 10, "alpha_freeze": true, "actor_bc_std_target": 0.01}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds match SAC0004 (warm-up and on-policy pretraining draws reproduce it exactly when the collection settings coincide). Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0040_bc_first_l100_std01_alphafreeze/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0040_bc_first_l100_std01_alphafreeze/run.py --execute > artifacts/bootstrap_variant_0040_bc_first_l100_std01_alphafreeze/run.log 2>&1
```
