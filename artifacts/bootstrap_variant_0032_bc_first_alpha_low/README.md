# Variant 32 `bc_first_alpha_low`; prepared, not executed

Temperature frozen at a third of the inherited value (raw-equivalent .02), weakening the entropy bonus so SAC can shrink the std where failures occur.

Settings that differ from the SAC0004 baseline: `{"demonstration_include_mpc": false, "actor_bc_lambda": 10.0, "actor_bc_rows": "first", "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "eval_stochastic_trials": 10, "alpha_freeze": true, "alpha_override": 0.0001}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds match SAC0004 (warm-up and on-policy pretraining draws reproduce it exactly when the collection settings coincide). Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0032_bc_first_alpha_low/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0032_bc_first_alpha_low/run.py --execute > artifacts/bootstrap_variant_0032_bc_first_alpha_low/run.log 2>&1
```
