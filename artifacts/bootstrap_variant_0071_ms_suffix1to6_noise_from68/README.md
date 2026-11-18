# Variant 71 `ms_suffix1to6_noise_from68`; prepared, not executed

As ms_suffix1to6_from68 with sigma .015 handoff noise for suffixes 3-6.

Settings that differ from the SAC0004 baseline: `{"prefix_set": [6, 7, 8], "demonstration_include_mpc": false, "actor_bc_lambda": 100.0, "handoff_noise_sigma": 0.015, "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "initial_from": ["bootstrap_variant_0068_ms_suffix123_from61", "online_2000"], "actor_lr_override": 0.0001, "actor_bc_std_target": 0.01, "prefix_sources": ["mpc_cold_demonstrations_0001", 25], "suffix_set": [1, 2, 3, 4, 5, 6], "demonstration_suffix_rows": 6, "handoff_noise_suffixes": [3, 4, 5, 6], "holdout_source_count": 2}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds match SAC0004 (warm-up and on-policy pretraining draws reproduce it exactly when the collection settings coincide). Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0071_ms_suffix1to6_noise_from68/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0071_ms_suffix1to6_noise_from68/run.py --execute > artifacts/bootstrap_variant_0071_ms_suffix1to6_noise_from68/run.log 2>&1
```
