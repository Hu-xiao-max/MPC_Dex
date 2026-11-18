# Proposed next-stage continuation; preparation only

This is a pre-launch protocol snapshot. No physics, policy inference or training was executed during preparation. Execution, if separately authorized, is recorded by `runtime.json`, checkpoint evidence and the terminal `summary.json`. Root selected prefix 6 as a **candidate for gated warm-up** after the frozen descending screen failed its deterministic trial and all five stochastic trials at prefix 6; prefixes 5–1 were not tested. Zero out of five does not prove that useful action support is absent. The completed screen must be bound before the stage is frozen.

Load the entire final SAC0001 checkpoint: BC 12000, critic pretraining 500, online 2000. Preserve actor, critics, target critics, all optimizer moments and step counts, learned standard deviation, learned normalized alpha (0.00030585992499254644), configuration and cumulative actor learning-rate ramp. Initialization is checked against the source payload, including optimizer state. This is a new curriculum stage with a fresh physical reset, replay and RNG, not exact resumption of the prior unfinished environment episode.

At every reset, physically replay the immutable original six recorded prefix controls and verify observations, actions, rewards, flags, complete integration states and 2 ms physics against the saved source. All task history, previous commands, finite-difference observation context, accumulated progress and sticky failures stay intact. The 100-control horizon includes the prefix. Keep raw environment rewards in all physical evidence and replay; divide only sampled update-batch rewards by 200. Retain the learned online temperature and the existing SoftSARSA alpha .0001. No network or temperature is reinitialized, and no clock input is added.

Use a new empty stage-specific replay. Copy the entire 2808-row prior replay solely as immutable lineage, including its one open final row. Never import that row, change its flags, mark it terminal, or connect it to a new reset. Fresh replay supplies actual current-stage successors under the frozen inherited actor for SoftSARSA, and avoids prior-stage transitions dominating the small new-stage collection. Possible forgetting is measured with separate prefix-7 retention evaluations.

The single prescribed path is:

1. Collect exactly 256 complete environment episodes of actual stochastic suffix control under the frozen inherited actor, with new sampling seed 17091. Prefix controls never enter replay or count as policy interactions. Save all failures and all direct sampled log probabilities.
2. Require at least one strict environment-and-force-qualified warm-up success. The five-screen outcomes are not the learning gate. If none qualifies, finish with status `warmup_only_no_positive_support`, preserve the complete warm-up checkpoint/data and inherited learner unchanged, and perform no learning or automatic fallback.
3. Only if the gate passes, run exactly 500 new existing SoftSARSA critic updates, then exactly 2000 new UTD1 SAC interactions/updates. Actor and alpha optimizer state are unchanged during the critic-only stage. Final cumulative counters are BC 12000, pretraining 1000, online 4000; critic optimizer step 5000 and actor/alpha step 4000. The first new actor update uses learning rate .00012006 (cumulative update 2001), ending at .00024 (update 4000).

The existing std bounds, two critics, learning rates, batch size 256, discount, Polyak rule and actor warm-up length 5000 remain unchanged. Episode rows record actual cumulative learner counters at each action; per-run new interactions/updates are separate. Online training episodes may span changing policies and are not frozen-policy success evidence.

Baseline before new pretraining and local online checkpoints +500/+1000/+2000 each receive one deterministic selected-phase evaluation, five stochastic phase trials (17201–17205), and one deterministic cold evaluation. Baseline and final also receive prefix-7 retention trials (one deterministic plus stochastic seeds 17221–17225). Five stochastic cold trials (17211–17215) occur only at the final online checkpoint. If the warm-up gate fails, only the baseline set occurs. Evaluations use separate environments, restore all training RNG, and never enter replay. There are no prefix-8 additions, stage advancement, noise sweeps or post-success continuations.

Empirical soft-Q calibration from complete terminal warm-up episodes uses both the actual inherited learned online alpha and the unchanged .0001 pretraining alpha, with explicit units. Exclude any truncated episode; its tail is unknown. Exclude current-action entropy and terminal bootstraps. Held-out baseline phase trajectories are compared before/after critic pretraining but never fitted. The unchanged SoftSARSA method uses inverse-action density with its 1e-6 clamp; empirical targets use the actual sampler-returned density.

Checkpoints preserve the full learner/optimizer state, fresh replay and its RNG/pending action, global RNG, exact live MuJoCo/Python/observer state, and unfinished suffix evidence at completed update boundaries. A final unfinished suffix keeps its real nonterminal flags. The runner provides evidence snapshots, not automatic resume. All assisted data remain ineligible for primitive export; a later qualified trained 60D policy still requires an independent frozen cold-policy collection.

Preparation, freeze and execution remain separate:

```bash
.venv/bin/python artifacts/bootstrap_suffix_sac_0002/prepare.py
# After completed stage binding and root review:
.venv/bin/python artifacts/bootstrap_suffix_sac_0002/prepare.py --finalize-prefix 6
# Only on separate explicit execution authorization:
.venv/bin/python artifacts/bootstrap_suffix_sac_0002/run.py --execute
```
