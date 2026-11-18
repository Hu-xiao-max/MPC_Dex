# Variant 90 `cyl_ms_from_sac0001`; prepared, not executed

Control for index 89: the same cylinder curriculum from the archived SAC0001 prefix-7 checkpoint (the default initial checkpoint of this family) instead of the hexagon multi-source checkpoint, to separate what the inherited hexagon competence contributes on the new object. Caveat: the two inherited checkpoints also differ in their learner counters and temperature (SAC0001 starts the actor-step ramp at 0.4 of 1e-4 and alpha at 3e-4 raw-equivalent, 0082 at the full step and 6e-3), so the pair attributes the joint effect of weights, ramp and temperature.

Settings that differ from the SAC0004 baseline: `{"prefix_set": [8], "demonstration_include_mpc": false, "actor_bc_lambda": 100.0, "online_policy_interactions": 4000, "checkpoints": [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], "actor_lr_override": 0.0001, "eval_stochastic_trials": 3, "actor_bc_std_target": 0.01, "prefix_sources": ["cylinder_demonstrations_0001", 120], "suffix_set": [1, 2, 3, 4, 5, 6, 7, 8], "demonstration_suffix_rows": 8, "eval_source_count": 2, "holdout_source_count": 3, "suffix_weights": [2, 2, 2, 2, 1, 1, 1, 1], "env_overrides": {"object_shape": "circular"}, "eval_recorded_prefixes": false, "eval_cold_every_point": true}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds match SAC0004 (warm-up and on-policy pretraining draws reproduce it exactly when the collection settings coincide). Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0090_cyl_ms_from_sac0001/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0090_cyl_ms_from_sac0001/run.py --execute > artifacts/bootstrap_variant_0090_cyl_ms_from_sac0001/run.log 2>&1
```
