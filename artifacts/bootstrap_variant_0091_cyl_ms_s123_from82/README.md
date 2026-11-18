# Variant 91 `cyl_ms_s123_from82`; prepared, not executed

Staged start of the cylinder curriculum: on the hexagon the multi-source route only progressed when the suffix set grew gradually (0068: lengths 1-3 over 15 sources reached 10/15, 10/15, 9/15 by 1,000 updates; the direct 1-8 stages regressed), whereas 0089 jumped to lengths 1-8 on the cylinder from the transferred checkpoint. Same 24-source snapshot and held-out sources as 0089/0090 (sorted names, first 24), lengths 1-3 uniform, three demonstration rows per source, 2,000 updates, inheriting the hexagon checkpoint 0082 online_4000; the follow-up extends to 1-6 and 1-8 from its best checkpoint as 0070/0074 did.

Settings that differ from the SAC0004 baseline: `{"prefix_set": [8], "demonstration_include_mpc": false, "actor_bc_lambda": 100.0, "checkpoints": [500, 1000, 1500, 2000], "initial_from": ["bootstrap_variant_0082_ms_shared_prefix_from81", "online_4000"], "actor_lr_override": 0.0001, "eval_stochastic_trials": 3, "actor_bc_std_target": 0.01, "prefix_sources": ["cylinder_demonstrations_0001", 24], "suffix_set": [1, 2, 3], "demonstration_suffix_rows": 3, "holdout_source_names": ["seed_153", "seed_154", "seed_155"], "eval_source_names": ["seed_101", "seed_102"], "env_overrides": {"object_shape": "circular"}, "eval_recorded_prefixes": false, "eval_cold_every_point": true}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds match SAC0004 (warm-up and on-policy pretraining draws reproduce it exactly when the collection settings coincide). Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0091_cyl_ms_s123_from82/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0091_cyl_ms_s123_from82/run.py --execute > artifacts/bootstrap_variant_0091_cyl_ms_s123_from82/run.log 2>&1
```
