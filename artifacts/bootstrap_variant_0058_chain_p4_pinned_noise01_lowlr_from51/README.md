# Variant 58 `chain_p4_pinned_noise01_lowlr_from51`; prepared, not executed

Prefix 4 inherited from the best 30-trial prefix-5 checkpoint (0051 online_1750: prefix-5 22/30 D, prefix-6 25/30 D, prefix-7/8 30/30), sigma .01 noise on prefixes 5-8, recorded controls 5-7 pinned, smaller actor step against the disruption of later prefixes seen in 0052.

Settings that differ from the SAC0004 baseline: `{"prefix_controls": 4, "prefix_set": [4, 5, 6, 7, 8], "demonstration_include_mpc": false, "demonstration_recorded_rows": [4, 9], "actor_bc_lambda": 100.0, "actor_bc_rows": "first_k", "handoff_noise_sigma": 0.01, "handoff_noise_prefixes": [5, 6, 7, 8], "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "initial_from": ["bootstrap_variant_0051_chain_p5_pinned_noise01_from39", "online_1750"], "actor_bc_first_rows": 3, "actor_lr_override": 0.0001, "eval_stochastic_trials": 10, "actor_bc_std_target": 0.01}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds match SAC0004 (warm-up and on-policy pretraining draws reproduce it exactly when the collection settings coincide). Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0058_chain_p4_pinned_noise01_lowlr_from51/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0058_chain_p4_pinned_noise01_lowlr_from51/run.py --execute > artifacts/bootstrap_variant_0058_chain_p4_pinned_noise01_lowlr_from51/run.log 2>&1
```
