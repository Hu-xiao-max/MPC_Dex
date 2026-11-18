# Main online training continuation

`train --resume RUN --output RUN` continues the latest committed main-training
generation. Supply the original `--config`, `--primitive`, `--replay`, `--seed`,
`--device`, `--batch-size` and `--utd`. The original fresh pretrained input can
be inferred; it and its metadata must remain available and unchanged. `--steps`
is a cumulative target and may be extended. `--checkpoint-every` defaults to
100 completed transitions and may change without changing the learned process.

Fresh training still requires the existing 12,000 BC / 50,000 critic update gate
and the fixed replay containing at least 20 completed MPC episodes. Resume does
not admit bootstrap or diagnostic recordings into that stage.

Each generation contains the learner and all optimizers, online and fixed
offline replay including sampler RNGs and unfinished episode bookkeeping, CEM
warmstart arrays and RNG, controller scheduler, Python/NumPy/Torch RNGs, full
MuJoCo binary `mjData`, and the environment's reward, contact-loss, retained
rotation and sticky palm-support state. Loading copies the saved live simulator
state without an additional physics step or `mj_forward` call. The saved current
observation preserves its finite-difference velocity components.

Files are flushed and hashed before the generation directory is published.
Only atomic replacement of `latest.json` commits it. Resume verifies every hash
before deserializing and rejects changed configurations, input artifacts, model
contracts, source hashes, numerical versions or relevant runtime settings. The
last two generations are retained. An exclusive OS lock prevents concurrent
processes from advancing the same run; its file may remain after exit and is
not a liveness indicator. This lock uses Linux/Unix `flock`.

SIGINT and SIGTERM request a stop after the current transition and **all** of
its UTD updates. Unexpected exceptions or forced termination resume from the
last committed generation; work since that boundary is discarded, then repeated
from the saved RNGs and optimizer state. No partially updated transition is
declared resumable. Uncommitted log tails are retained separately as
`metrics.uncommitted.*.jsonl` before the main metrics log continues.

Top-level `checkpoint.pt` and `online_replay.npz` are convenient exports, not
resume inputs. At a completed interaction budget, the public replay preserves
the previous budget-truncated export behavior. The committed internal replay
keeps an actual unfinished episode open, allowing a later budget extension to
execute its next control and record the real successor action.
Resuming at the already completed cumulative budget republishes the committed
state and repairs public exports, without advancing the environment or learner.

Small CPU tests compare interrupted and uninterrupted real MuJoCo transitions,
stochastic policy actions, learner/optimizer tensors, replay sampling, planner
state and resets bit for bit. MuJoCo profiling timers and unused solver scratch
memory differ between separate executions; all snapshot bytes are restored
exactly before continuation. CUDA RNG state is retained, but these tests do not
establish bit-for-bit determinism for CUDA kernels or a changed machine. These
are persistence tests, not successful-rotation or training-performance evidence.
