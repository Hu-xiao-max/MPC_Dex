# Frozen-policy critic calibration:500 updates

The authorized empirical-return fit improved held-out value error while leaving the successful BC actor unchanged. It did **not** fit the training targets well within500 updates. The measured residual is substantially explained by a restricted critic output range at this budget; it is not evidence that the terminal targets are wrong or that the main problem is held-out generalization.

The exact first256 online controls from bootstrap0005 were generated before its first SAC update. Starting from its saved Torch/CUDA RNG, `actor.sample` reproduces every action and supplies the actual raw-sample log probability without inverse-tanh/clamping. Physical re-execution matches initial integration state, every observation/next observation/flag, and every rounded replay reward;12,800 physics states are retained. These rows form78 complete true-terminal episodes. All256 are eligible, none need a tail, and the nine MPC seed rows are excluded.

Targets use actual re-executed float64 rewards and frozen alpha≈.05: terminal G=r; otherwise G[t]=r[t]+.995*(G[t+1]−alpha*logpi[t+1]). Exactly500 uniform256-row minibatch critic updates used existing Adam LR3e−4, NumPyseed17051, and target Polyak tau.01. There is no bootstrapped return target and no actor or alpha update. Held-out seeds17041–17045 were assessed only and never fitted.

| Measure | Critic1 before | Critic1 after500 | Critic2 before | Critic2 after500 |
|---|---:|---:|---:|---:|
| Held-out initial-action mean bias |74.979|36.805|74.648|36.412|
| Held-out initial-action RMSE |78.856|44.166|78.539|43.839|
| Training mean bias |75.981|38.057|75.886|37.666|
| Training RMSE |85.900|55.219|85.814|54.949|

Training targets span−195.512 to+6.551, mean−75.825. After500, critic outputs on training rows span only−37.789..−37.576 and−38.182..−37.970. For a tanh feature vector h and final linear layer w·h+b, the exact current global lower bound is b−||w||1. These bounds are−43.442 and−43.925;89.0625% of training targets lie below them. Even arbitrary bounded penultimate features cannot reduce training RMSE below50.780/50.452 with these final output weights. The residual is therefore already large on training data and constrained by the current output-head scale.

Output-head mean absolute weight changes are.13939/.13913 (maximum.15824/.15916), with bias changes−.12994/−.12982. Mean absolute penultimate features rise from≈.56 to≈.869; none exceed.99 in magnitude. The proposed rough500×3e−4×256≈38.4 scale is explanatory intuition, not a rigorous Adam update bound. Exact measured head bounds are recorded in `capacity_analysis.json`.

Actor trunk, mean, std, alpha, their optimizers, and counters remain bit-exact: BC12000, pretrain0, online0. Custom critic-calibration updates500 are separately recorded. Reconstructing all held-out actions/log probabilities with the trainer's batch(1,60) shape and saved RNG also matches exactly. This preserves the original deterministic policy identity but adds no actor learning, successful stochastic episodes, SAC-success claim, or primitive eligibility. The diagnostic checkpoint is explicitly ineligible.

An independent reviewer verified original-data binding, terminal-return arithmetic, non-overlap of fitting and held-out(obs,action) rows, and all20 critic Adam parameter states atstep500. Source code, inputs, reconstructed controls/physics, targets, learned critic snapshot, fit metrics, capacity inspection, runtime, and verification are archived. No adaptive extension was performed. A separately authorized centered-initialization comparison will have its own artifact and will not change this result.
