# Variant 64 `full_pinned_lowlr_from57`; prepared, not executed

As full_pinned_lowlr_noise_from57 without handoff noise.

Settings that differ from the SAC0004 baseline: `{"prefix_controls": 0, "prefix_set": [0, 1, 2, 3, 4, 5, 6, 7, 8], "demonstration_include_mpc": false, "demonstration_recorded_rows": [0, 9], "actor_bc_lambda": 100.0, "actor_bc_rows": "first_k", "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "initial_from": ["bootstrap_variant_0057_chain_p4_pinned_lowlr_from56", "online_2000"], "actor_bc_first_rows": 7, "actor_lr_override": 0.0001, "eval_stochastic_trials": 10, "actor_bc_std_target": 0.01}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds match SAC0004 (warm-up and on-policy pretraining draws reproduce it exactly when the collection settings coincide). Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0064_full_pinned_lowlr_from57/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0064_full_pinned_lowlr_from57/run.py --execute > artifacts/bootstrap_variant_0064_full_pinned_lowlr_from57/run.log 2>&1
```
