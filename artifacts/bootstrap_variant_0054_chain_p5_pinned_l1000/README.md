# Variant 54 `chain_p5_pinned_l1000`; prepared, not executed

Ten times stronger pin at obs5/obs6: the recorded control 6 basin is narrower than .01 while the pinned mean sits .01-.03 away.

Settings that differ from the SAC0004 baseline: `{"prefix_controls": 5, "prefix_set": [5, 6, 7, 8], "demonstration_include_mpc": false, "demonstration_recorded_rows": [5, 9], "actor_bc_lambda": 1000.0, "actor_bc_rows": "first_k", "handoff_noise_sigma": 0.015, "handoff_noise_prefixes": [6, 7, 8], "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "initial_from": ["bootstrap_variant_0039_bc_first_l100_std01", "online_2000"], "actor_bc_first_rows": 2, "eval_stochastic_trials": 10, "actor_bc_std_target": 0.01}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds match SAC0004 (warm-up and on-policy pretraining draws reproduce it exactly when the collection settings coincide). Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0054_chain_p5_pinned_l1000/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0054_chain_p5_pinned_l1000/run.py --execute > artifacts/bootstrap_variant_0054_chain_p5_pinned_l1000/run.log 2>&1
```
