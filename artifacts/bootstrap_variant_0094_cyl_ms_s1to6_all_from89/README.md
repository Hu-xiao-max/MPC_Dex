# Variant 94 `cyl_ms_s1to6_all_from89`; prepared, not executed

Same as index 93 (lengths 1-6 weighted from 0089 online_2500, 4,000 updates) over every verified cylinder demonstration (54 at stamp time; 51 training) instead of the 24-source snapshot, with the same three held-out sources, to measure what source count contributes on the cylinder, as 0074 did on the hexagon.

Settings that differ from the SAC0004 baseline: `{"prefix_set": [8], "demonstration_include_mpc": false, "actor_bc_lambda": 100.0, "online_policy_interactions": 4000, "checkpoints": [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], "initial_from": ["bootstrap_variant_0089_cyl_ms_from82", "online_2500"], "actor_lr_override": 0.0001, "eval_stochastic_trials": 3, "actor_bc_std_target": 0.01, "prefix_sources": ["cylinder_demonstrations_0001", 120], "suffix_set": [1, 2, 3, 4, 5, 6], "demonstration_suffix_rows": 6, "suffix_weights": [2, 2, 2, 1, 1, 1], "holdout_source_names": ["seed_153", "seed_154", "seed_155"], "eval_source_names": ["seed_101", "seed_102"], "env_overrides": {"object_shape": "circular"}, "eval_recorded_prefixes": false, "eval_cold_every_point": true}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds match SAC0004 (warm-up and on-policy pretraining draws reproduce it exactly when the collection settings coincide). Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0094_cyl_ms_s1to6_all_from89/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0094_cyl_ms_s1to6_all_from89/run.py --execute > artifacts/bootstrap_variant_0094_cyl_ms_s1to6_all_from89/run.log 2>&1
```
