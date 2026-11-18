# One post-hoc maximum-boundary hold branch

This explicitly post-hoc branch selected **control boundary 26**, the maximum force-qualified angle among the existing 28 recorded boundaries. At 2.6 s it held 6.260594098 rad, with three loaded fingers and tilt 0.192784449 rad. Selection was recorded before execution in selection.json; no other prefix or target was tried.

The cold-reset prefix matched all 26 recorded transitions and 1,300 physics rows bit-for-bit, verified before issuing the first extra hold. The exact float32 command from control 26 was then held for the entire authorized 20-control (2 s) budget, without replanning, blending, clipping, state edits, or target changes.

**No strict full turn occurred.** Final force-qualified rotation was 5.962360321 rad, with two loaded fingers and tilt 0.264846964 rad. The whole-episode peak remained 6.265469558 rad from the original prefix, below 2π. No drop or palm/table/other support occurred. The success-conditioned independent replay and extra 2 s aftermath hold were therefore not run.

Only this one physical branch executed. The final immutable-input assertion initially included the mutable run.log and failed after all physical outputs were saved. finalize.py verified that run.log was the sole changed entry, checked all saved prefix/action/target data, and produced the final report without rerunning physics. Original execution code/log/input hashes are preserved. This is partial diagnostic data, with no policy, SAC, primitive, or production-success claim.
