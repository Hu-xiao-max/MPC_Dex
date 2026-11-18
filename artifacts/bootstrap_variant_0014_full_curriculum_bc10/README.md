# Variant 14 `full_curriculum_bc10`; prepared, not executed

Reverse curriculum over every handoff state of the recorded trajectory at once, with the whole recorded trajectory as demonstrations and a supervised pull; the only variant that collects cold-start episodes and could show cold progress.

Settings that differ from the SAC0004 baseline: `{"prefix_set": [0, 1, 2, 3, 4, 5, 6, 7, 8], "demonstration_include_mpc": false, "demonstration_recorded_rows": [0, 9], "actor_bc_lambda": 10.0}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds match SAC0004 (warm-up and on-policy pretraining draws reproduce it exactly when the collection settings coincide). Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0014_full_curriculum_bc10/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0014_full_curriculum_bc10/run.py --execute > artifacts/bootstrap_variant_0014_full_curriculum_bc10/run.log 2>&1
```
