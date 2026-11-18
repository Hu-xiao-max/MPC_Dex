# Variant 07 `handoff_noise_mixed`; prepared, not executed

Mastered competence is a point (basin about L2 .02-.05 around recorded control 7); executing the last prefix control with N(0,.025^2) action noise on prefixes 7/8 trains the suffix on a neighbourhood so that the stochastic policy's own deviation (about .27) stops capping prefix-6 success.

Settings that differ from the SAC0004 baseline: `{"prefix_set": [6, 7, 8], "demonstration_include_mpc": false, "handoff_noise_sigma": 0.025, "handoff_noise_prefixes": [7, 8]}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds match SAC0004 (warm-up and on-policy pretraining draws reproduce it exactly when the collection settings coincide). Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0007_handoff_noise_mixed/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0007_handoff_noise_mixed/run.py --execute > artifacts/bootstrap_variant_0007_handoff_noise_mixed/run.log 2>&1
```
