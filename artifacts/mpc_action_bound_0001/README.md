# Diagnostic normalized-action search bound

This artifact verifies the optional `--action-bound` in `scripts/probe_mpc.py`. It changes only the diagnostic planner's joint-target search limits. The environment's action range, physics, task validation, production planner, and paper configuration are unchanged.

The requested bound was `0.98`; the effective float32 bound was `0.9799998998641968`. The inward rounding uses the float32 grid for the environment's `action + 1` conversion, so both physical endpoints remain fixed under action/target round trips. The report records both bounds, all sixteen search limits, and the independent diagnostic source choice. Initialization rejects an executed target outside these limits rather than clipping it.

Exactly one native CEM plan and one physical control step ran:

```sh
.venv/bin/python scripts/probe_mpc.py \
  --config artifacts/mpc_matched40_reference_0001/source/config.json \
  --trajectory artifacts/mpc_matched40_reference_0001/source/input_trajectory.npz \
  --output artifacts/mpc_action_bound_0001 \
  --steps 1 --seed 42 --workers 1 --backend native \
  --progress retained --perpendicular-weight 100 \
  --reference-candidates --action-bound .98
```

The measured 40 mm prior produced six PCA components. The unchanged candidate budget was 350 per iteration, three iterations: 175 exploration, 173 random refinement, and two deterministic references. There were 37 feasible candidates (6, 14, and 17 per iteration). The selected exact hold cost was `19.7190358373712`; its maximum normalized action magnitude was `0.6794583797454834`, within the narrowed search range. The actual step retained `0.0013055022887588665` radians with four finger contacts, zero palm force, and no drop. The run took about 8.13 seconds. This one-step check is not a full-turn or policy-success claim.

`verify.py` and `verification.json` preserve the additional checks, which ran no further CEM plans or physics steps. Default bound 1 reproduces the archived prior probe's limits, random samples, and decoded commands exactly with reference candidates enabled/disabled and slew limits absent/set to 0.02. Narrowed endpoint conversions, sampled commands, exact holds, invalid-bound rejection, initial-target rejection, and source hashes also passed. `source/` snapshots the code, configuration, and measured prior used in the physical run.
