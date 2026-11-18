# Variant 68 `ms_suffix123_from61`; prepared, not executed

Diverse handoff states: each episode replays a random verified cold-reset MPC demonstration up to its last 1-3 controls and the policy finishes; the last three rows of every demonstration are pinned (mean and .01 spread) and serve as critic demonstrations. Inherits the chained checkpoint that already solves recorded prefixes 4-8; recorded prefixes 6-8 are evaluated for retention.

Settings that differ from the SAC0004 baseline: `{"prefix_set": [6, 7, 8], "demonstration_include_mpc": false, "actor_bc_lambda": 100.0, "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "initial_from": ["bootstrap_variant_0061_chain_p3_cont_from59", "online_1750"], "actor_lr_override": 0.0001, "actor_bc_std_target": 0.01, "prefix_sources": ["mpc_cold_demonstrations_0001", 15], "suffix_set": [1, 2, 3], "demonstration_suffix_rows": 3}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds match SAC0004 (warm-up and on-policy pretraining draws reproduce it exactly when the collection settings coincide). Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0068_ms_suffix123_from61/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0068_ms_suffix123_from61/run.py --execute > artifacts/bootstrap_variant_0068_ms_suffix123_from61/run.log 2>&1
```
