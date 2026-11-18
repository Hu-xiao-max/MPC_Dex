# Variant 72 `ms_suffix4to10_from70`; prepared, not executed

Third multi-source stage: suffix lengths 4-10 (demonstrations of at most 10 controls become cold-start episodes), last ten rows pinned, every available demonstration except two held-out ones used for training (the suffix-1-3 checkpoint memorized per-trajectory suffixes: 61/120, 0/120, 3/120 on unseen demonstrations), lighter in-run evaluation (two training sources plus the two held-out, three stochastic trials) with the 30-trial evaluator for the rest. Inherits the suffix-1-6 stage at online 500 (its best in-run point) rather than waiting for its slow completion.

Settings that differ from the SAC0004 baseline: `{"prefix_set": [8], "demonstration_include_mpc": false, "actor_bc_lambda": 100.0, "checkpoints": [250, 500, 750, 1000, 1250, 1500, 1750, 2000], "initial_from": ["bootstrap_variant_0070_ms_suffix1to6_from68", "online_500"], "actor_lr_override": 0.0001, "eval_stochastic_trials": 3, "actor_bc_std_target": 0.01, "prefix_sources": ["mpc_cold_demonstrations_0001", 60], "suffix_set": [4, 5, 6, 7, 8, 9, 10], "demonstration_suffix_rows": 10, "eval_source_count": 2, "holdout_source_count": 2}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds match SAC0004 (warm-up and on-policy pretraining draws reproduce it exactly when the collection settings coincide). Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0072_ms_suffix4to10_from70/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0072_ms_suffix4to10_from70/run.py --execute > artifacts/bootstrap_variant_0072_ms_suffix4to10_from70/run.log 2>&1
```
