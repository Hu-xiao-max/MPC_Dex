# Variant 08 `dart_recipe`; prepared, not executed

Combination judged jointly necessary by the panel: BC to recorded control 7 fixes the mean at obs6, sigma .05 handoff noise makes the continuation robust to policy-scale deviations, mixed prefixes preserve the suffix, and a 5% online demonstration share limits repetition; pretraining share stays .3.

Settings that differ from the SAC0004 baseline: `{"prefix_set": [6, 7, 8], "demonstration_include_mpc": false, "online_offline_fraction": 0.05, "actor_bc_lambda": 10.0, "handoff_noise_sigma": 0.05, "handoff_noise_prefixes": [7, 8]}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds match SAC0004 (warm-up and on-policy pretraining draws reproduce it exactly when the collection settings coincide). Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0008_dart_recipe/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0008_dart_recipe/run.py --execute > artifacts/bootstrap_variant_0008_dart_recipe/run.log 2>&1
```
