# Variant 78 `ms_suffix1to7_from74`; prepared, not executed

Smaller curriculum step: suffix lengths 1-7 from 0074 online_4000 (one added control instead of two).

Settings that differ from the SAC0004 baseline: `{"prefix_set": [8], "demonstration_include_mpc": false, "actor_bc_lambda": 100.0, "online_policy_interactions": 4000, "checkpoints": [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], "initial_from": ["bootstrap_variant_0074_ms_suffix1to6_cont_from70", "online_4000"], "actor_lr_override": 0.0001, "eval_stochastic_trials": 3, "actor_bc_std_target": 0.01, "prefix_sources": ["mpc_cold_demonstrations_0001", 80], "suffix_set": [1, 2, 3, 4, 5, 6, 7], "demonstration_suffix_rows": 7, "eval_source_count": 2, "holdout_source_count": 2}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds match SAC0004 (warm-up and on-policy pretraining draws reproduce it exactly when the collection settings coincide). Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0078_ms_suffix1to7_from74/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0078_ms_suffix1to7_from74/run.py --execute > artifacts/bootstrap_variant_0078_ms_suffix1to7_from74/run.log 2>&1
```
