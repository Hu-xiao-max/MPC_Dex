This diagnostic compares the original evaluator against the actual float32 normalized-action execution path. It uses 350 identical CEM candidates at each of three physically replayed states, with no changes to the simulator.

| Prior controls | Maximum target rounding (rad) | Largest shared-feasible cost difference | Feasible raw / mapped | Best candidate raw / mapped |
| --- | --- | --- | --- | --- |
| 0 | 8.38e-08 | 5.65035e-05 | 6 / 6 | 348 / 348 |
| 5 | 8.42e-08 | 2.9232e-05 | 13 / 13 | 186 / 186 |
| 15 | 8.37e-08 | 1.40054 | 109 / 109 | 246 / 246 |

The sampled best indices and feasibility decisions remain unchanged in these three batches. Other costs can nevertheless change materially. For each selected feasible candidate, simulating the mapped targets exactly reproduces the actual qpos/qvel endpoint; simulating the original unrounded targets does not. This establishes a conversion mismatch, without attributing all learning failures to it.

The evaluator now applies the exact environment action conversion before simulation. The native backend inherits the same conversion. The environment, physical actuator limits, CEM sample budget and scoring equation are unchanged. The new physical regression checks both backends against actual normalized-action execution, including integration inputs, acceleration and contact partners.

The original failing test output is preserved in regression_before_fix.txt. That initial test compared full serialized MjData, which also contains profiling/scratch memory; the final regression compares relevant physical fields exactly. Source snapshots and run.py describe the pre-fix experiment. Earlier failed preparation attempts 0001/0002 stopped before producing data and remain preserved.

All 59 targeted rollout, native-backend and CLI tests pass after the correction. Two existing exact-threshold fixtures now measure workspace/palm boundaries using the actual converted command; their equal/adjacent-float rejection expectations remain strict.
