# Variant 62 `chain_p3_std005_l300_from59`; prepared, not executed

Tighter pin (lambda 300, std .005) at obs3-obs6 continuing from the prefix-3 stage: deterministic prefix-3 trials pass at some checkpoints while stochastic ones fail, pointing at spread and mean precision at the knife-edge control 4.

Settings that differ from the SAC0004 baseline: `{"prefix_controls": 3, "prefix_set": [3, 4, 5, 6, 7, 8], "demonstration_include_mpc": false, "demonstration_recorded_rows": [3, 9], "actor_bc_lambda": 300.0, "actor_bc_rows": "first_k", "handoff_noise_sigma": 0.015, "handoff_noise_prefixes": [4, 5, 6, 7, 8], "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "initial_from": ["bootstrap_variant_0059_chain_p3_pinned_lowlr_from57", "online_2000"], "actor_bc_first_rows": 4, "actor_lr_override": 0.0001, "eval_stochastic_trials": 10, "actor_bc_std_target": 0.005}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds match SAC0004 (warm-up and on-policy pretraining draws reproduce it exactly when the collection settings coincide). Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0062_chain_p3_std005_l300_from59/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0062_chain_p3_std005_l300_from59/run.py --execute > artifacts/bootstrap_variant_0062_chain_p3_std005_l300_from59/run.log 2>&1
```
