# Variant 06 `recorded_bc10_isolated_s6`; prepared, not executed

The handoff becomes a BC-only state: no on-policy episode starts at prefix 6 and demonstration rows are excluded from the SAC actor/temperature losses, so only the supervised term shapes the mean at obs6 while SAC on prefix-7/8 rows maintains the suffix; prefix 6 is still evaluated.

Settings that differ from the SAC0004 baseline: `{"prefix_set": [7, 8], "demonstration_include_mpc": false, "actor_bc_lambda": 10.0, "actor_on_demonstration_states": false}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds match SAC0004 (warm-up and on-policy pretraining draws reproduce it exactly when the collection settings coincide). Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0006_recorded_bc10_isolated_s6/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0006_recorded_bc10_isolated_s6/run.py --execute > artifacts/bootstrap_variant_0006_recorded_bc10_isolated_s6/run.log 2>&1
```
