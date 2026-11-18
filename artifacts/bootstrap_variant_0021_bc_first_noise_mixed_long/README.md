# Variant 21 `bc_first_noise_mixed_long`; prepared, not executed

bc_first_noise_mixed with twice the online budget.

Settings that differ from the SAC0004 baseline: `{"prefix_set": [6, 7, 8], "demonstration_include_mpc": false, "actor_bc_lambda": 10.0, "actor_bc_rows": "first", "handoff_noise_sigma": 0.025, "handoff_noise_prefixes": [7, 8], "online_policy_interactions": 4000, "checkpoints": [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds match SAC0004 (warm-up and on-policy pretraining draws reproduce it exactly when the collection settings coincide). Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0021_bc_first_noise_mixed_long/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0021_bc_first_noise_mixed_long/run.py --execute > artifacts/bootstrap_variant_0021_bc_first_noise_mixed_long/run.log 2>&1
```
