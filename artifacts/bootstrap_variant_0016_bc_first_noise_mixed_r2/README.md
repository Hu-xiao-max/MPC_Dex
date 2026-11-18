# Variant 16 `bc_first_noise_mixed_r2`; prepared, not executed

Seed replicate of bc_first_noise_mixed.

Settings that differ from the SAC0004 baseline: `{"prefix_set": [6, 7, 8], "demonstration_include_mpc": false, "actor_bc_lambda": 10.0, "actor_bc_rows": "first", "handoff_noise_sigma": 0.025, "handoff_noise_prefixes": [7, 8], "seed_matched": false, "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000]}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds are fresh. Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0016_bc_first_noise_mixed_r2/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0016_bc_first_noise_mixed_r2/run.py --execute > artifacts/bootstrap_variant_0016_bc_first_noise_mixed_r2/run.log 2>&1
```
