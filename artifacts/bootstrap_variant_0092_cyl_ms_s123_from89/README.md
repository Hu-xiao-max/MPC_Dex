# Variant 92 `cyl_ms_s123_from89`; prepared, not executed

Consolidation candidate: the best cylinder checkpoint so far (0089 online_2500: 90/150 at the last control over five evaluated sources, 60/90 on the three held-out ones, 48/150 at the last two) continued with the narrow suffix set 1-3 on the same 24-source snapshot, against 0091 which starts the same staged curriculum from the hexagon checkpoint. The better of the two at 30 trials seeds the extension to lengths 1-6.

Settings that differ from the SAC0004 baseline: `{"prefix_set": [8], "demonstration_include_mpc": false, "actor_bc_lambda": 100.0, "checkpoints": [500, 1000, 1500, 2000], "initial_from": ["bootstrap_variant_0089_cyl_ms_from82", "online_2500"], "actor_lr_override": 0.0001, "eval_stochastic_trials": 3, "actor_bc_std_target": 0.01, "prefix_sources": ["cylinder_demonstrations_0001", 24], "suffix_set": [1, 2, 3], "demonstration_suffix_rows": 3, "holdout_source_names": ["seed_153", "seed_154", "seed_155"], "eval_source_names": ["seed_101", "seed_102"], "env_overrides": {"object_shape": "circular"}, "eval_recorded_prefixes": false, "eval_cold_every_point": true}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds match SAC0004 (warm-up and on-policy pretraining draws reproduce it exactly when the collection settings coincide). Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0092_cyl_ms_s123_from89/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0092_cyl_ms_s123_from89/run.py --execute > artifacts/bootstrap_variant_0092_cyl_ms_s123_from89/run.log 2>&1
```
