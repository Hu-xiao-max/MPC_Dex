# Variant 52 `chain_p4_pinned_noise015_from48`; prepared, not executed

Next reverse-curriculum step: prefix 4 inherited from the prefix-5 stage (online 2,000: prefix-5 deterministic pass, 3/10 stochastic; prefix 6 9/10; 7-8 10/10), with recorded controls 5-7 pinned at obs4-obs6 and sigma .015 noise on prefixes 5-8.

Settings that differ from the SAC0004 baseline: `{"prefix_controls": 4, "prefix_set": [4, 5, 6, 7, 8], "demonstration_include_mpc": false, "demonstration_recorded_rows": [4, 9], "actor_bc_lambda": 100.0, "actor_bc_rows": "first_k", "handoff_noise_sigma": 0.015, "handoff_noise_prefixes": [5, 6, 7, 8], "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "initial_from": ["bootstrap_variant_0048_chain_p5_pinned_noise015_from39", "online_2000"], "actor_bc_first_rows": 3, "eval_stochastic_trials": 10, "actor_bc_std_target": 0.01}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds match SAC0004 (warm-up and on-policy pretraining draws reproduce it exactly when the collection settings coincide). Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0052_chain_p4_pinned_noise015_from48/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0052_chain_p4_pinned_noise015_from48/run.py --execute > artifacts/bootstrap_variant_0052_chain_p4_pinned_noise015_from48/run.log 2>&1
```
