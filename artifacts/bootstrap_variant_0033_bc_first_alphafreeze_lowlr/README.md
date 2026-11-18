# Variant 33 `bc_first_alphafreeze_lowlr`; prepared, not executed

Both stabilizers: frozen temperature and the smaller actor step.

Settings that differ from the SAC0004 baseline: `{"demonstration_include_mpc": false, "actor_bc_lambda": 10.0, "actor_bc_rows": "first", "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "actor_lr_override": 0.0001, "eval_stochastic_trials": 10, "alpha_freeze": true}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds match SAC0004 (warm-up and on-policy pretraining draws reproduce it exactly when the collection settings coincide). Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0033_bc_first_alphafreeze_lowlr/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0033_bc_first_alphafreeze_lowlr/run.py --execute > artifacts/bootstrap_variant_0033_bc_first_alphafreeze_lowlr/run.log 2>&1
```
