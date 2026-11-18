# Variant 15 `bc_first_noise_mixed`; prepared, not executed

Wave-1 winners combined: first-row BC (recorded_bc10_first gave prefix-6 deterministic passes with full 7/8 retention) plus sigma .025 handoff noise with mixed prefixes (handoff_noise_mixed reached prefix-6 stochastic 5/5 at online 1000). Checkpoints every 250 to resolve the drift seen after 1000.

Settings that differ from the SAC0004 baseline: `{"prefix_set": [6, 7, 8], "demonstration_include_mpc": false, "actor_bc_lambda": 10.0, "actor_bc_rows": "first", "handoff_noise_sigma": 0.025, "handoff_noise_prefixes": [7, 8], "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000]}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds match SAC0004 (warm-up and on-policy pretraining draws reproduce it exactly when the collection settings coincide). Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0015_bc_first_noise_mixed/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0015_bc_first_noise_mixed/run.py --execute > artifacts/bootstrap_variant_0015_bc_first_noise_mixed/run.log 2>&1
```
