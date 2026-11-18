# Variant 31 `bc_first_alphafreeze`; prepared, not executed

recorded_bc10_first with the temperature frozen at its inherited value (.000306 normalized): alpha rose monotonically in every run because the policy entropy never approaches the -16 target, inflating the handoff std from .067 to .13-.17 while the mean stayed on the label; 30-trial rates drifted 20/30 -> 6/30 with more updates.

Settings that differ from the SAC0004 baseline: `{"demonstration_include_mpc": false, "actor_bc_lambda": 10.0, "actor_bc_rows": "first", "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "eval_stochastic_trials": 10, "alpha_freeze": true}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds match SAC0004 (warm-up and on-policy pretraining draws reproduce it exactly when the collection settings coincide). Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0031_bc_first_alphafreeze/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0031_bc_first_alphafreeze/run.py --execute > artifacts/bootstrap_variant_0031_bc_first_alphafreeze/run.log 2>&1
```
