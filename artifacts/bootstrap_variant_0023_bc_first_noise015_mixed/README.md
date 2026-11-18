# Variant 23 `bc_first_noise015_mixed`; prepared, not executed

Narrower handoff noise (sigma .015) with first-row BC and mixed prefixes: the basin around recorded control 7 is about L2 .02-.05 wide, so .015 per dimension (L2 about .06) may widen it without swamping the suffix.

Settings that differ from the SAC0004 baseline: `{"prefix_set": [6, 7, 8], "demonstration_include_mpc": false, "actor_bc_lambda": 10.0, "actor_bc_rows": "first", "handoff_noise_sigma": 0.015, "handoff_noise_prefixes": [7, 8], "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000]}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds match SAC0004 (warm-up and on-policy pretraining draws reproduce it exactly when the collection settings coincide). Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0023_bc_first_noise015_mixed/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0023_bc_first_noise015_mixed/run.py --execute > artifacts/bootstrap_variant_0023_bc_first_noise015_mixed/run.log 2>&1
```
