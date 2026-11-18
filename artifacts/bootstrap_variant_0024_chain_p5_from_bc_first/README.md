# Variant 24 `chain_p5_from_bc_first`; prepared, not executed

Chained stage: inherit the recorded_bc10_first final checkpoint (prefix-6 deterministic pass, stochastic 3/5, prefix-7/8 retention 5/5), move the handoff one recorded control earlier with first-row BC toward recorded control 6 at obs5, and keep prefixes 6-8 in the stream against forgetting.

Settings that differ from the SAC0004 baseline: `{"prefix_controls": 5, "prefix_set": [5, 6, 7, 8], "demonstration_include_mpc": false, "demonstration_recorded_rows": [5, 9], "actor_bc_lambda": 10.0, "actor_bc_rows": "first", "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "initial_from": ["bootstrap_variant_0004_recorded_bc10_first", "online_2000"]}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds match SAC0004 (warm-up and on-policy pretraining draws reproduce it exactly when the collection settings coincide). Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0024_chain_p5_from_bc_first/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0024_chain_p5_from_bc_first/run.py --execute > artifacts/bootstrap_variant_0024_chain_p5_from_bc_first/run.log 2>&1
```
