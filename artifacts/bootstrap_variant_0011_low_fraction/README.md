# Variant 11 `low_fraction`; prepared, not executed

Only the online demonstration share drops from .3 to .05 (pretraining share unchanged), testing whether repeating 8 rows about 9.6 times per online batch destabilizes the actor.

Settings that differ from the SAC0004 baseline: `{"online_offline_fraction": 0.05}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds match SAC0004 (warm-up and on-policy pretraining draws reproduce it exactly when the collection settings coincide). Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0011_low_fraction/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0011_low_fraction/run.py --execute > artifacts/bootstrap_variant_0011_low_fraction/run.log 2>&1
```
