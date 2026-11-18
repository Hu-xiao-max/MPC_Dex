# Variant 02 `recorded_only_control`; prepared, not executed

Single-factor control: SAC0004 with the five MPC rows removed. Read-only probes show the MPC continuation is unfinishable by this actor from any point, so removing it should stop the drift toward the MPC region; retention loss is still expected (stream holds only prefix-6 failures).

Settings that differ from the SAC0004 baseline: `{"demonstration_include_mpc": false}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds match SAC0004 (warm-up and on-policy pretraining draws reproduce it exactly when the collection settings coincide). Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0002_recorded_only_control/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0002_recorded_only_control/run.py --execute > artifacts/bootstrap_variant_0002_recorded_only_control/run.log 2>&1
```
