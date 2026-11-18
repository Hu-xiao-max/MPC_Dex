# Variant 01 `mixed_prefix`; prepared, not executed

Forgetting is critic-led when the stream holds only prefix-6 failures; sampling the collection prefix uniformly from {6,7,8} keeps mastered states in replay. Keeps the 8-row MPC+recorded buffer for attribution against the recorded-only variants.

Settings that differ from the SAC0004 baseline: `{"prefix_set": [6, 7, 8]}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds match SAC0004 (warm-up and on-policy pretraining draws reproduce it exactly when the collection settings coincide). Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0001_mixed_prefix/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0001_mixed_prefix/run.py --execute > artifacts/bootstrap_variant_0001_mixed_prefix/run.log 2>&1
```
