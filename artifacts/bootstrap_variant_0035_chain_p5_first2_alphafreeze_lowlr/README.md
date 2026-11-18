# Variant 35 `chain_p5_first2_alphafreeze_lowlr`; prepared, not executed

Chained prefix-5 stage with both stabilizers.

Settings that differ from the SAC0004 baseline: `{"prefix_controls": 5, "prefix_set": [5, 6, 7, 8], "demonstration_include_mpc": false, "demonstration_recorded_rows": [5, 9], "actor_bc_lambda": 10.0, "actor_bc_rows": "first_k", "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "initial_from": ["bootstrap_variant_0004_recorded_bc10_first", "online_2000"], "actor_bc_first_rows": 2, "actor_lr_override": 0.0001, "eval_stochastic_trials": 10, "alpha_freeze": true}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds match SAC0004 (warm-up and on-policy pretraining draws reproduce it exactly when the collection settings coincide). Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0035_chain_p5_first2_alphafreeze_lowlr/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0035_chain_p5_first2_alphafreeze_lowlr/run.py --execute > artifacts/bootstrap_variant_0035_chain_p5_first2_alphafreeze_lowlr/run.log 2>&1
```
