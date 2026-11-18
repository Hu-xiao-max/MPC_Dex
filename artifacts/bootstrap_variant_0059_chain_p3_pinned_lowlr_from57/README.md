# Variant 59 `chain_p3_pinned_lowlr_from57`; prepared, not executed

Prefix 3 inherited from the prefix-4 stage (0057 online_2000 in-run: prefix-4 deterministic pass and 3/10, prefix-5 8/10, prefixes 6-8 10/10), recorded controls 4-7 pinned at obs3-obs6, sigma .015 noise on prefixes 4-8, actor step 1e-4.

Settings that differ from the SAC0004 baseline: `{"prefix_controls": 3, "prefix_set": [3, 4, 5, 6, 7, 8], "demonstration_include_mpc": false, "demonstration_recorded_rows": [3, 9], "actor_bc_lambda": 100.0, "actor_bc_rows": "first_k", "handoff_noise_sigma": 0.015, "handoff_noise_prefixes": [4, 5, 6, 7, 8], "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "initial_from": ["bootstrap_variant_0057_chain_p4_pinned_lowlr_from56", "online_2000"], "actor_bc_first_rows": 4, "actor_lr_override": 0.0001, "eval_stochastic_trials": 10, "actor_bc_std_target": 0.01}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds match SAC0004 (warm-up and on-policy pretraining draws reproduce it exactly when the collection settings coincide). Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0059_chain_p3_pinned_lowlr_from57/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0059_chain_p3_pinned_lowlr_from57/run.py --execute > artifacts/bootstrap_variant_0059_chain_p3_pinned_lowlr_from57/run.log 2>&1
```
