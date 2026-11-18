# Paired critic fit with training-derived output-bias initialization

The separate centered500-update diagnostic substantially improves training fit relative to the original uncentered500-update fit. It does not establish trustworthy action gradients or policy improvement. The original artifact remains unchanged, and neither actor nor alpha was updated.

The only initialization change was to add the mean of the256 verified TRAIN targets,−75.8250227572, to each critic and target critic's final bias. Every other initial tensor and all optimizer states match the original initial checkpoint. The same256 actual pre-update rows, complete-terminal soft-Q targets,500 minibatches, NumPyseed17051, Adam LR3e−4, and Polyak tau.01 were used. All500 minibatch index arrays are saved and verified. No held-out value or action selected the center. No physics was rerun for this paired fit.

| Initialization/fit | Train RMSE Q1/Q2 | Held-out initial bias Q1/Q2 | Held-out initial RMSE Q1/Q2 |
|---|---|---|---|
| Uncentered,500 updates |55.219 /54.949|+36.805 /+36.412|44.166 /43.839|
| Centered,zero updates |40.073 /40.068|−0.846 /−1.177|24.436 /24.443|
| Centered,500 updates |22.468 /22.144|−15.114 /−13.692|37.143 /34.723|

After centered500, full-training biases are+2.349/+2.183. Q outputs span−110.204..−44.111 and−110.661..−43.960, compared with almost constant−38 for the uncentered fit. Exact current global output lower bounds are−116.606/−117.225;19.531% of train targets remain below them, versus89.063% without centering. Output-head scale still limits part of the lower tail, but the bulk of the target range is now represented. Mean absolute penultimate features are.714/.707; output-head mean absolute weight changes from centered initialization are.12920/.12942.

The cold-state distribution matters when interpreting the held-out comparison:

| Recorded target population | N | Mean | Population variance | Median |10th–90th percentiles |
|---|---:|---:|---:|---:|---|
| All fitting rows |256|−75.825|1601.720|−57.637|−135.404..−42.198|
| Fitting episode-initial rows |78|−91.327|1741.050|−91.762|−138.022..−41.662|
| Held-out episode-initial rows |5|−74.594|596.056|−64.276|−104.731..−52.327|

The78 training resets have one identical initial observation but distinct sampled actions and future noise. Centered500 predicts their initial values with means−89.708/−89.301 and variances485.450/516.754; episode-initial training RMSE is28.021/27.703. On the five held-out actions, prediction means are−89.708/−88.286 and variances486.801/406.258. Thus the shift from the all-row center toward−90 is consistent with the TRAIN episode-initial return distribution. Worse five-episode error than the centered zero-update baseline alone does not prove overfitting; finite sampled returns, different actions, and future policy noise remain unresolved. Full quantiles and prediction distributions are in `return_distribution_comparison.json`.

Read-only gradient probes use the **same256 actual observations and actually executed actions**, regenerated bit-for-bit from saved RNG with the unchanged actor. Centering alone leaves all derivatives exactly unchanged. After500, the uncentered versus centered negative-Q gradient norms are:

| Derivative | Uncentered500 | Centered500 |
|---|---:|---:|
| Actor shared trunk |.00613|2332.073|
| Actor mean head |.00730|558.280|
| Actor std head |.000561|5.695|
| Actual action tensor |.000952|17.422|

Entropy derivatives remain identical because actor/std/alpha are unchanged. These large fitted derivative differences are not evidence that the directions improve real returns. No actor update or unfreeze was performed.

The actor, all its trunk/mean/std tensors, alpha, BC/actor/alpha optimizer states, and stage counters remain bit-exact. BC12000, pretrain0, online0 persist; custom critic-calibration updates500 are separately labeled. All20 critic Adam parameter states are at step500. Held-out actions/log probabilities and final RNG match the original frozen policy exactly. The checkpoint is diagnostic and ineligible for SAC primitive or main-stage promotion. No success or robustness-rate claim follows from this fit.
