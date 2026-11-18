# Variant 67 `chain_p3_cont2_from61`; prepared, not executed

Third stage-length at prefix 3 from the best 30-trial prefix-3 checkpoint so far (0061 online_1750: prefix-3 8/30 D, prefix-4 16/30 D, prefix-5 16/30, prefix-6 28/30, prefixes 7-8 30/30): prefix 3 rose from 3/30 to 8/30 over one continuation.

Settings that differ from the SAC0004 baseline: `{"prefix_controls": 3, "prefix_set": [3, 4, 5, 6, 7, 8], "demonstration_include_mpc": false, "demonstration_recorded_rows": [3, 9], "actor_bc_lambda": 100.0, "actor_bc_rows": "first_k", "handoff_noise_sigma": 0.015, "handoff_noise_prefixes": [4, 5, 6, 7, 8], "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "initial_from": ["bootstrap_variant_0061_chain_p3_cont_from59", "online_1750"], "actor_bc_first_rows": 4, "actor_lr_override": 0.0001, "eval_stochastic_trials": 10, "actor_bc_std_target": 0.01}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds match SAC0004 (warm-up and on-policy pretraining draws reproduce it exactly when the collection settings coincide). Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0067_chain_p3_cont2_from61/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0067_chain_p3_cont2_from61/run.py --execute > artifacts/bootstrap_variant_0067_chain_p3_cont2_from61/run.log 2>&1
```
