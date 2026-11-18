# Variant 74 `ms_suffix1to6_cont_from70`; prepared, not executed

Continue the suffix-1-6 stage for 4,000 more updates over every available demonstration: its rates were still rising at online 1,500 (20/25, 16/25, 10/25, 6/25, 6/25, 4/25).

Settings that differ from the SAC0004 baseline: `{"prefix_set": [8], "demonstration_include_mpc": false, "actor_bc_lambda": 100.0, "online_policy_interactions": 4000, "checkpoints": [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], "initial_from": ["bootstrap_variant_0070_ms_suffix1to6_from68", "online_2000"], "actor_lr_override": 0.0001, "eval_stochastic_trials": 3, "actor_bc_std_target": 0.01, "prefix_sources": ["mpc_cold_demonstrations_0001", 60], "suffix_set": [1, 2, 3, 4, 5, 6], "demonstration_suffix_rows": 6, "eval_source_count": 2, "holdout_source_count": 2}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds match SAC0004 (warm-up and on-policy pretraining draws reproduce it exactly when the collection settings coincide). Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0074_ms_suffix1to6_cont_from70/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0074_ms_suffix1to6_cont_from70/run.py --execute > artifacts/bootstrap_variant_0074_ms_suffix1to6_cont_from70/run.log 2>&1
```
