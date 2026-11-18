# Variant 46 `bc_first_l100_std01_long`; prepared, not executed

bc_first_l100_std01 with twice the budget: does the pinned handoff stay stable beyond 2,000 updates (bc_first_long decayed from 20/30 to 6/30 by 4,000)?

Settings that differ from the SAC0004 baseline: `{"demonstration_include_mpc": false, "actor_bc_lambda": 100.0, "actor_bc_rows": "first", "online_policy_interactions": 4000, "checkpoints": [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], "eval_stochastic_trials": 10, "actor_bc_std_target": 0.01}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds match SAC0004 (warm-up and on-policy pretraining draws reproduce it exactly when the collection settings coincide). Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0046_bc_first_l100_std01_long/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0046_bc_first_l100_std01_long/run.py --execute > artifacts/bootstrap_variant_0046_bc_first_l100_std01_long/run.log 2>&1
```
