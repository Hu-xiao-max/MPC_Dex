# Variant 27 `bc_first_lowlr_long`; prepared, not executed

bc_first_lowlr with twice the budget, since a smaller step may need more updates to reach the same handoff residual.

Settings that differ from the SAC0004 baseline: `{"demonstration_include_mpc": false, "actor_bc_lambda": 10.0, "actor_bc_rows": "first", "online_policy_interactions": 4000, "checkpoints": [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], "actor_lr_override": 0.0001, "eval_stochastic_trials": 10}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds match SAC0004 (warm-up and on-policy pretraining draws reproduce it exactly when the collection settings coincide). Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0027_bc_first_lowlr_long/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0027_bc_first_lowlr_long/run.py --execute > artifacts/bootstrap_variant_0027_bc_first_lowlr_long/run.log 2>&1
```
