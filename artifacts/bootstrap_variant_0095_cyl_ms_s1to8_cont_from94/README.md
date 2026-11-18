# Variant 95 `cyl_ms_s1to8_cont_from94`; prepared, not executed

Continuation of the all-source cylinder stage (0094 online_4000: 120/150 at the last control, held-out 60/90; 31, 27, 5 of 150 at lengths 2-4) with lengths 1-8 weighted 2:2:2:2:1:1:1:1 for 4,000 more updates, as the hexagon continuation 0074 gained a length; evaluated on a 10-control source (seed_101) and the shortest 8-control source (seed_172) plus the held-out trio.

Settings that differ from the SAC0004 baseline: `{"prefix_set": [8], "demonstration_include_mpc": false, "actor_bc_lambda": 100.0, "online_policy_interactions": 4000, "checkpoints": [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], "initial_from": ["bootstrap_variant_0094_cyl_ms_s1to6_all_from89", "online_4000"], "actor_lr_override": 0.0001, "eval_stochastic_trials": 3, "actor_bc_std_target": 0.01, "prefix_sources": ["cylinder_demonstrations_0001", 120], "suffix_set": [1, 2, 3, 4, 5, 6, 7, 8], "demonstration_suffix_rows": 8, "suffix_weights": [2, 2, 2, 2, 1, 1, 1, 1], "holdout_source_names": ["seed_153", "seed_154", "seed_155"], "eval_source_names": ["seed_101", "seed_172"], "env_overrides": {"object_shape": "circular"}, "eval_recorded_prefixes": false, "eval_cold_every_point": true}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds match SAC0004 (warm-up and on-policy pretraining draws reproduce it exactly when the collection settings coincide). Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0095_cyl_ms_s1to8_cont_from94/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0095_cyl_ms_s1to8_cont_from94/run.py --execute > artifacts/bootstrap_variant_0095_cyl_ms_s1to8_cont_from94/run.log 2>&1
```
