# Variant 37 `bc_first_std02`; prepared, not executed

Spread anchor at the handoff row (latent std toward .02 per dimension, L2 about .08): the handoff std grew from .067 to .13-.17 in every BC run while the basin is about .025-.05 wide, capping stochastic success near two thirds.

Settings that differ from the SAC0004 baseline: `{"demonstration_include_mpc": false, "actor_bc_lambda": 10.0, "actor_bc_rows": "first", "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "eval_stochastic_trials": 10, "actor_bc_std_target": 0.02}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds match SAC0004 (warm-up and on-policy pretraining draws reproduce it exactly when the collection settings coincide). Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0037_bc_first_std02/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0037_bc_first_std02/run.py --execute > artifacts/bootstrap_variant_0037_bc_first_std02/run.log 2>&1
```
