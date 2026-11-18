# Variant 36 `bc_first_l100`; prepared, not executed

Ten times stronger first-row supervision: 30-trial collapses coincide with the handoff mean drifting to L2 .09 from the label (0019 online_2000) in a basin about .025-.05 wide.

Settings that differ from the SAC0004 baseline: `{"demonstration_include_mpc": false, "actor_bc_lambda": 100.0, "actor_bc_rows": "first", "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "eval_stochastic_trials": 10}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds match SAC0004 (warm-up and on-policy pretraining draws reproduce it exactly when the collection settings coincide). Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0036_bc_first_l100/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0036_bc_first_l100/run.py --execute > artifacts/bootstrap_variant_0036_bc_first_l100/run.log 2>&1
```
