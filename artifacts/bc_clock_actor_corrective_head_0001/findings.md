# One86-pair frozen-head policy audit

One float64 minimum-norm residual solve preserved the final86-pair61D BC trunk and all non-head learner/optimizer values. The86x257 design has full row rank and condition8.35169e7; correction norm7.07181, maximum coefficient0.756758, and final headweight norm7.70553. The ideal maximum residual is4.33e-15. Actual CUDA singleton float32 label fit has maximum actionerror4.76837e-7, including all original/noisy-prefix/actual-policy/MPC groups. There were no alternate solves, regularization sweeps or BC epochs.

After numerical review, exactly one strict deterministic actual-policy audit ran with unchanged physics and the artifact-local actual elapsed-time feature. It dropped at control19, without a qualified full turn or palm support. Per2ms peak force-qualified angle was5.644944168rad at1.370s; final angle was−0.738627121rad. No stochastic test or post-success continuation ran.

The reset and all19control boundaries preserve complete MuJoCo integration states; every saved qpos/qvel/ctrl/time agrees with those states. Candidate inference on every saved visited observation reproduces the executed float32 actions bit for bit. Comparison against nominal9 and the successful recovered14 records is saved without rerunning physics. Using the explicitly descriptive threshold of normalized action error>0.001, the first nominal divergence is control9 at0.800s: maxactionerror0.0020391047, observationL22.75607243. Other thresholds and semantic observation groups are retained; elapsed-time equality is checked separately.

All prior artifacts are unchanged. Counters remain BC12000/critic0/SAC0. Neither the successful teacher recovery nor this failed learned execution supplies a SAC/main60D primitive or robustness claim.
