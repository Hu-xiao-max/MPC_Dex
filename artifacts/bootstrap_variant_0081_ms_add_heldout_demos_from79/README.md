# Variant 81 `ms_add_heldout_demos_from79`; prepared, not executed

Control for the DAgger stage: the same continuation of 0079 online_4000 with the previously held-out demonstrations (seeds 194-196) simply added as ordinary training sources, without labels at policy-visited states; only seed_197 stays held out. Separates the effect of labelling policy-visited states from the effect of adding those three trajectories.

Settings that differ from the SAC0004 baseline: `{"prefix_set": [8], "demonstration_include_mpc": false, "actor_bc_lambda": 100.0, "online_policy_interactions": 4000, "checkpoints": [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], "initial_from": ["bootstrap_variant_0079_ms_suffix1to8_weighted_from74", "online_4000"], "actor_lr_override": 0.0001, "eval_stochastic_trials": 3, "actor_bc_std_target": 0.01, "prefix_sources": ["mpc_cold_demonstrations_0001", 120], "suffix_set": [1, 2, 3, 4, 5, 6, 7, 8], "demonstration_suffix_rows": 8, "suffix_weights": [2, 2, 2, 2, 1, 1, 1, 1], "holdout_source_names": ["seed_197"], "eval_source_names": ["seed_100", "seed_101", "seed_194", "seed_195", "seed_196"]}`. Everything else (SAC0001 checkpoint, 256 warm-up episodes, entropy-free demonstration pretraining, 500 + 2,000 updates, evaluation schedule) matches SAC0004; see `protocol.json`. Seeds match SAC0004 (warm-up and on-policy pretraining draws reproduce it exactly when the collection settings coincide). Template: `bootstrap_variant_template_0001`.

```bash
.venv/bin/python -m pytest -q artifacts/bootstrap_variant_0081_ms_add_heldout_demos_from79/test_protocol.py
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_variant_0081_ms_add_heldout_demos_from79/run.py --execute > artifacts/bootstrap_variant_0081_ms_add_heldout_demos_from79/run.log 2>&1
```
