# Variant 57 `chain_p4_pinned_lowlr_from56`; prepared, not executed

Prefix 4 inherited from the prefix-5 stage with the smaller actor step (online 2,000 in-run: prefix-5 deterministic pass and 7/10, prefix-6 10/10, 7-8 10/10), pinning recorded controls 5-7 at obs4-obs6, sigma .015 noise on prefixes 5-8, actor step 1e-4.

Settings that differ from the SAC0004 baseline: `{"prefix_controls": 4, "prefix_set": [4, 5, 6, 7, 8], "demonstration_include_mpc": false, "demonstration_recorded_rows": [4, 9], "actor_bc_lambda": 100.0, "actor_bc_rows": "first_k", "handoff_noise_sigma": 0.015, "handoff_noise_prefixes": [5, 6, 7, 8], "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "initial_from": ["bootstrap_variant_0056_chain_p5_pinned_lowlr", "online_2000"], "actor_bc_first_rows": 3, "actor_lr_override": 0.0001, "eval_stochastic_trials": 10, "actor_bc_std_target": 0.01}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds match SAC0004 (warm-up and on-policy pretraining draws reproduce it exactly when the collection settings coincide). Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0057_chain_p4_pinned_lowlr_from56/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0057_chain_p4_pinned_lowlr_from56/run.py --execute > artifacts/bootstrap_variant_0057_chain_p4_pinned_lowlr_from56/run.log 2>&1
```
