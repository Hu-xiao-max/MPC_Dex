# Variant 12 `no_actor_on_demo`; prepared, not executed

SAC actor and temperature losses restricted to on-policy rows; the critic still learns from demonstration rows. Tests whether actor updates on demonstration states drive the drift.

Settings that differ from the SAC0004 baseline: `{"actor_on_demonstration_states": false}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds match SAC0004 (warm-up and on-policy pretraining draws reproduce it exactly when the collection settings coincide). Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0012_no_actor_on_demo/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0012_no_actor_on_demo/run.py --execute > artifacts/bootstrap_variant_0012_no_actor_on_demo/run.log 2>&1
```
