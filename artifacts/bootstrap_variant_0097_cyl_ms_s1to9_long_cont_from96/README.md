# Variant 97 `cyl_ms_s1to9_long_cont_from96`; prepared, not executed

Continuation of 0096 (lengths 1-9 weighted toward long suffixes over all 54 sources) for 4,000 more updates from its online_4000, whose in-run cells reached deterministic passes at lengths 5-6 on the 8-control source seed_172 (its cold reset is length 8) and 1-3 on seed_116; same evaluated and held-out sources.

Settings that differ from the SAC0004 baseline: `{"prefix_set": [8], "demonstration_include_mpc": false, "actor_bc_lambda": 100.0, "online_policy_interactions": 4000, "checkpoints": [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], "initial_from": ["bootstrap_variant_0096_cyl_ms_s1to9_long_from94", "online_4000"], "actor_lr_override": 0.0001, "eval_stochastic_trials": 3, "actor_bc_std_target": 0.01, "prefix_sources": ["cylinder_demonstrations_0001", 120], "suffix_set": [1, 2, 3, 4, 5, 6, 7, 8, 9], "demonstration_suffix_rows": 9, "suffix_weights": [1, 1, 1, 2, 2, 2, 2, 2, 2], "holdout_source_names": ["seed_153", "seed_154", "seed_155"], "eval_source_names": ["seed_172", "seed_116", "seed_174"], "env_overrides": {"object_shape": "circular"}, "eval_recorded_prefixes": false, "eval_cold_every_point": true}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds match SAC0004 (warm-up and on-policy pretraining draws reproduce it exactly when the collection settings coincide). Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0097_cyl_ms_s1to9_long_cont_from96/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0097_cyl_ms_s1to9_long_cont_from96/run.py --execute > artifacts/bootstrap_variant_0097_cyl_ms_s1to9_long_cont_from96/run.log 2>&1
```
