# Optional native exact rollouts

The optional backend moves the Python evaluator's physics loop into C++ while
retaining the existing vectorized Python scoring. It does not change the model,
physics timestep, full-data synchronization, candidate constraints, or progress
mode. Both backends enforce rotation without force-bearing palm support. The
original evaluator remains the default.

For the main `collect` and `train` commands, add this top-level field to the
existing JSON configuration:

```json
{
  "planner_backend": "native"
}
```

Omitting the field, or setting it to `"python"`, keeps the exact Python
evaluator. Explicit `"native"` requires the compiled backend and fails if it
cannot initialize; the CLI never silently falls back. This field changes only
the transition implementation. Existing planner budgets, cost settings and
physical constraints are preserved.

Before the first plan, both commands write `planner_backend` and
`planner_runtime` into `run.json`. The runtime record includes the actual
evaluator backend/class, seed, resolved CEM/cost configurations, worker count,
environment contract, numerical source and lockfile hashes, dependency
versions, and native build provenance when used. Initialization failures mark
the run failed and record the error instead of leaving a running manifest.

For direct Python use:

```python
from mpc_dex.native_rollouts import NativeMujocoRolloutEvaluator
from mpc_dex.rollouts import RolloutCostConfig

evaluator = NativeMujocoRolloutEvaluator(
    env, timestep_seconds=.01,
    config=RolloutCostConfig(workers=8, rotation_progress='retained'),
    require_native=True,
)
try:
    costs = evaluator(sequences)  # finite joint targets, shape [N,H,16]
    print(evaluator.last_diagnostics)
finally:
    evaluator.close()
```

`require_native=False` is the default. An unavailable compiler, unsupported
platform/ABI, plugins, or global callbacks selects the original exact Python
implementation and records the reason. `require_native=True` raises instead.
The current build route supports 64-bit Linux, using the installed MuJoCo wheel's
own headers and shared library; it adds no Python dependency.

Each worker owns a typed MuJoCo data object and begins each candidate with
`mj_copyData` from the live full-data snapshot. Its native loop executes
`mj_step` followed by `mj_forward` at every physics timestep, writes the complete
`mjSTATE_FULLPHYSICS` state and distinct-finger count, and stops at the first
divergence, table contact, force-bearing palm support, workspace violation, or
inherited contact-loss deadline. After divergence checks, rejection precedence
is table, palm support, workspace, then contact loss. For rotation, palm support
is the sum of positive normal forces returned by `mj_contactForce` for contacts
between the object and a geom whose body is the palm. It must be no greater than
`EnvConfig.palm_support_force_tolerance` (default `1e-6` N). Mere geometric
proximity with zero force is allowed. A preexisting sticky episode
`_palm_support_violation` rejects every candidate before any physics step;
reorientation does not enforce this rotation-specific constraint.

Unexecuted state tails retain the original initial state. Near the
workspace boundary, where C++ and NumPy arithmetic could differ by rounding,
the worker restarts that candidate and the remaining chunk through the original
Python loop. This preserves the exact Python comparison without a tolerance in
the physical constraint.

Python checks array dimensions, dtype, contiguity, alignment, writable output
buffers, typed owner addresses and each data object's model ownership. Owners
remain strongly referenced throughout each synchronous native call. The ctypes
call releases the GIL so the existing worker threads can run native transitions
in parallel. As with the original evaluator, do not mutate the environment,
model, callbacks or evaluator concurrently with planning.

Compiled files and JSON build provenance live in
`artifacts/native_rollouts/<build-key>/`. Provenance includes MuJoCo version,
header/runtime ABI agreement, compiler/version/flags and source/header/library/
binary SHA-256 hashes. Transition ABI version 2 adds the rotation-only palm
constraint, initial violation flag and force tolerance; both loaded-library ABI
and MuJoCo ABI must agree with the typed Python interface. The changed source
hash creates a separate build cache entry. Builds disable fast-math and
floating-point contraction.
`native_provenance` and `last_diagnostics['native_build']` expose this metadata;
`native_boundary_fallback_candidates` counts candidates delegated to Python for
exact boundary handling.

## Validation and measured runtime

Run the exact equivalence/rejection tests:

```bash
uv run --frozen python -m pytest -q tests/test_native_rollouts.py tests/test_rollout_constraints.py
```

The 48 tests passed on the installed MuJoCo 3.14.0 environment. Native tests
compare full per-step state arrays, distinct-finger counts, full final
integration states, costs, executed step counts and rejection reasons bit for
bit against Python. Cases include stable grasps with external forces, varying
joint targets, one-finger contact, airborne spin, an inherited 48 ms contact-loss
timer, table contact, workspace escape, divergence and exact workspace boundary
values plus their adjacent representable floats. Palm tests distinguish loaded
support from zero-force proximity, preserve sticky episode history, verify
palm-before-workspace rejection and the reorientation exemption, and compare
the exact force threshold plus its adjacent representable floats. Live
integration state remains unchanged. Raw and retained progress scoring are both
covered.

Reproduce the benchmark:

```bash
uv run --frozen python -m scripts.benchmark_native_rollouts --workers 8 --output artifacts/native_rollouts/benchmark_no_palm_support.json
```

The following medians are from three complete scoring calls per backend on
2026-09-23, with eight workers, the stable saved grasp, and force-based palm
rejection enabled. Compilation and worker startup are excluded. All candidates
executed their entire horizon, with zero rejections, and returned exactly equal
costs.

| Candidates × horizon | Physics steps per evaluation | Python | Native | Speedup |
| --- | ---: | ---: | ---: | ---: |
| 100 × 0.1 s | 5,000 | 0.1567 s | 0.0598 s | 2.62× |
| 350 × 0.5 s | 87,500 | 2.6424 s | 1.3987 s | 1.89× |

Three large native evaluations require approximately 4.20 seconds before CEM
optimizer overhead. This does **not** meet a 10 Hz planning deadline. Runtime
depends on available CPU capacity; these are observed timings, not a guaranteed
deadline. The current ABI-2 build, constraint protocol and individual timings
are recorded in `artifacts/native_rollouts/benchmark_no_palm_support.json`.
`benchmark_final.json` and `benchmark.json` are historical ABI-1 measurements
from before force-based palm rejection and must not be reported as measurements
of the current constraint implementation.

The evaluator converts each proposed joint-target sequence through the same float32 normalized-action mapping used by execution before simulating it. This conversion is shared by the Python and native backends. An independent diagnostic found target differences of about 8.4e−8 rad could change a predicted contact-trajectory cost by 1.4; three selected trajectories match actual qpos/qvel exactly after conversion. The simulator and CEM sampling budgets are unchanged. See `artifacts/planner_action_mapping_0003/findings.md` and the physical regression in `tests/test_rollout_action_mapping.py`. Earlier benchmarks and probes retain their archived pre-conversion sources.
