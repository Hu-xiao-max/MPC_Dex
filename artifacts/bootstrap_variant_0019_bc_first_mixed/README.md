# Variant 19 `bc_first_mixed`; prepared, not executed

First-row BC with mixed prefixes and no handoff noise: isolates the contribution of noise in bc_first_noise_mixed.

Settings that differ from the SAC0004 baseline: `{"prefix_set": [6, 7, 8], "demonstration_include_mpc": false, "actor_bc_lambda": 10.0, "actor_bc_rows": "first", "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000]}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds match SAC0004 (warm-up and on-policy pretraining draws reproduce it exactly when the collection settings coincide). Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0019_bc_first_mixed/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0019_bc_first_mixed/run.py --execute > artifacts/bootstrap_variant_0019_bc_first_mixed/run.log 2>&1
```
