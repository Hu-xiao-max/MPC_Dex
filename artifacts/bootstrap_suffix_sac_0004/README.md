# Prefix-6 SAC with entropy-free demonstration targets; prepared, not executed

[SAC0003](../bootstrap_suffix_sac_0003/findings.md) mixed 8 verified off-policy demonstration rows into the unchanged SoftSARSA pretraining. Its entropy term −α log π(a′|s′), evaluated on recorded MPC next actions whose actor log density reaches −155,331, inflated those rows' targets from about 1.1 to 29.4; the critic fitted the inflation, the actor drifted, and prefix-7 retention fell from 6/6 to 0/6. This stage repeats SAC0003 from the same SAC0001 checkpoint with exactly one objective change plus denser retention monitoring.

**Objective change.** Critic pretraining uses a runner-local `entropy_masked_pretrain_step` that reproduces the archived `pretrain_critic_step` (same eligibility rule, critic step, Polyak update and counter) except that demonstration rows bootstrap entropy-free: target = r + γ · min Q′(s′, a′_recorded). On-policy rows keep the unchanged SoftSARSA target with α = .0001. A unit test verifies bit-identical critic and target-critic parameters against the archived step when no demonstration rows are present. Online SAC updates are the unchanged `SACAgent.update`, which samples successor actions from the current actor for every row and therefore never evaluates the density of demonstration actions.

**Everything else as SAC0003.** Inherit the entire SAC0001 final checkpoint (SHA `e4a73d89…`; SAC0003 checkpoints are not used). Fresh reset, fresh online replay, new seed 17491; 256 frozen-actor prefix-6 warm-up episodes (successes recorded, not required); the identical immutable 8-row demonstration buffer (5 MPC corrections and recorded controls 7–9, both terminal successes from the same handoff observation) mixed into every batch at the main-method 30/70 split via a mask-carrying reproduction of `sample_mixed`; 500 critic updates, then 2,000 UTD-1 SAC interactions/updates; the same evaluation schedule with seeds 17501–17525, except that prefix-7 retention is evaluated at every checkpoint rather than only baseline and final. There is no retention hard stop: the stage runs to its budget, and retention failures are reported. Demonstration calibration at each point now compares critic values against exactly the targets those rows were trained toward.

Confounds unchanged from SAC0003 and disclosed: heavy repetition of 8 rows per batch, and the changed reset curriculum. All assisted and demonstration-mixed data remain ineligible for primitive export.

```bash
.venv/bin/python artifacts/bootstrap_suffix_sac_0004/prepare.py
.venv/bin/python -m pytest -q artifacts/bootstrap_suffix_sac_0004/test_protocol.py
# Only on separate explicit execution authorization:
MUJOCO_GL=egl .venv/bin/python artifacts/bootstrap_suffix_sac_0004/run.py --execute > artifacts/bootstrap_suffix_sac_0004/run.log 2>&1
```
