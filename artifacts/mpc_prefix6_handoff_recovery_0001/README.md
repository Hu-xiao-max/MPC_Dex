# Prefix-6 handoff MPC recovery; prepared, not executed

This stage follows the failed [prefix-6 warm-up gate](../bootstrap_suffix_sac_0002/findings.md): the trained SAC0001 checkpoint dropped the object in all 262 physical attempts from the six-control handoff, so it cannot supply positive examples there. Route selected by root: obtain verified non-policy correction labels from the same handoff state with the bounded corrected native MPC recipe, before any new learning budget.

Preparation copies the byte-identical numerical source snapshot of `bootstrap_suffix_sac_0002`, its recorded prefix transitions and physics, the gate evidence, the unchanged measured 40 mm diagnostic PCA prior, the archived `ProjectedReferenceCEM` recipe and audit helper from `mpc_common_noise_recovery_0001`, and the single-branch clock recovery reports as reference. Every copied input is hash-bound in `input_provenance.json` and frozen in `execution_manifest.json`. The native build cache is copied only to avoid recompilation and is excluded from the freeze; its build and binary hashes are recorded in the report at runtime.

The single prescribed branch:

1. Cold reset with physical seed 42, then physically re-execute the six recorded controls. Observations, actions, rewards, flags, qpos/qvel, targets, complete integration states and all 300 prefix physics samples must match the source bit-for-bit. Handoff is 0.6 s at 5.258153506 force-qualified rad.
2. At most twenty corrected native MPC controls with the unchanged recipe: eight workers, 350 candidates, ten elites, three iterations, six knots over .5 s, .1 s controls, retained progress, perpendicular weight 100, position weight 1000, principal weight 1, CEM seed 42, .98 search bound with warm nominal and projected current-target reference candidates. Live physics is never reset or clipped; every planning call must leave the live integration state unchanged.
3. Stop at the first environment boundary, a no-feasible-plan result or the budget. Independent cold-reset replay of every executed control, with a separate force audit, must match bit-for-bit.
4. If executed MPC controls exist, rows after the prefix are packaged as `actual_mpc_correction_pairs.npz` with exact row provenance; prefix rows are never expert labels and no successor action is fabricated.

Disclosure: the recorded controls 7–9 of the same source trajectory already complete the turn from this handoff. MPC therefore supplies an independent feedback-derived label set rather than proof that a positive action exists, and the report records the distance between each correction and the recorded control at the same index. No policy inference, learning update, post-success hold, alternate prefix or parameter sweep is part of this stage. Its output is candidate seed data for a separately reviewed prefix-6 curriculum stage; it is not primitive-eligible and adds no main demonstration.

```bash
.venv/bin/python -m pytest -q artifacts/mpc_prefix6_handoff_recovery_0001/test_protocol.py
.venv/bin/python artifacts/mpc_prefix6_handoff_recovery_0001/prepare.py
# Only on separate explicit execution authorization:
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python artifacts/mpc_prefix6_handoff_recovery_0001/run.py --execute > artifacts/mpc_prefix6_handoff_recovery_0001/run.log 2>&1
```
