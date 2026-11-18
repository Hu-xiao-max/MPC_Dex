# Bootstrap0005: bounded SAC run completed without a new full turn

The fresh SAC experiment stopped normally at exactly **2,000 online interactions**, with **1,745 SAC updates** and exit code0. There was no automatic extension. It produced **zero training successes and zero scheduled evaluation successes**. The imported deterministic BC-plus-head-refinement motion was preserved at initialization, but the evaluated SAC checkpoints did not retain its successful first-turn behavior.

## Protocol and provenance

The unchanged40mm absolute-target,60-observation, horizon100 environment was used, with the original rotation reward, gravity/contact dynamics, and strict no-palm/retention criteria. `--initial-actor` imported only the immutable common-continuation actor trunk and mean. Source BC history remains12,000 updates; new BC and critic-pretraining updates are both zero. Critics, target critics, optimizers, and temperature were fresh. One physically validated nine-transition MPC demonstration seeded replay; it is explicitly separate from SAC successes.

The configured learner used256-by-256 hidden layers, two critics/targets, initial latent std0.05, actor learning rate3e-4 with5,000-update warmup, critic learning rate3e-4, temperature learning rate1e-4, initial alpha0.05, discount0.995, tau0.01, and target entropy−16. Other existing bootstrap defaults were unchanged. Seed17031, one environment, batch256, UTD1, and learning_starts256 were fixed. CUDA singleton-environment inference used one Torch intra/inter-op thread, highest float32 matmul precision, and both TF32 flags disabled. The nominal MPC replay received BC0 and critic-pretrain0. There was no main-stage training or count promotion.

All numerical code/assets, actual actor and seed inputs, resolved configuration, launch command, and runtime were archived before the first interaction. The launcher used the frozen reviewed `BootstrapRunner` without numerical edits, subclasses, or monkeypatches; a separate I/O thread copied committed checkpoints. Initial/500/1,000/1,500/2,000 copies are outside rolling cleanup. Final generation00000006 is authoritative, with agent SHA-256 `386be7a4e915a3b54e7bd7fc534c5b9502b84f35fd5e7139f7ec3c6ddeabc220`. Generation00000005 has identical agent bytes and precedes the final status commit.

## Actual outcomes

The run completed644 episodes. All ended in failure:410 were labeled drop and234 palm support; five palm failures also met the drop condition, giving415 physical drops in total. The replay contains2,009 transitions:2,000 actual online interactions plus the nine seed transitions. No successful trajectory export was created. The maximum final retained angle among completed training episodes was4.50636rad; this is an episode-end statistic, not a maximum over unrecorded per-physics training states. Mean completed-episode return was−46.0486. The longest episode lasted20 controls.

| Interaction checkpoint | Actual SAC updates | Scheduled deterministic result | Final retained angle |
|---:|---:|---|---:|
| 500 | 245 | Drop after2 controls, each repeat | −0.394627rad |
| 1,000 | 745 | Drop after2 controls, each repeat | 0.244121rad |
| 1,500 | 1,245 | Palm support after2 controls, each repeat | 0.555800rad |
| 2,000 | 1,745 | Palm support after7 controls, each repeat | 1.003255rad |

Each scheduled evaluation consisted of five identical physical cold resets, and each checkpoint reproduced the same outcome in all five. These are repeated deterministic checks, **not five independently varied robustness trials**, and the four checkpoint sets are different policies. No success from the BC-only phase was counted as SAC-origin; there were no training successes in either phase. The independently audited initial deterministic full turn is explicitly BC-origin because its online update count is zero.

## Independent frozen value audits

The separate native reviewer audited immutable initial,1,000, and2,000 checkpoints with one deterministic strict rollout and the same five held-out stochastic seeds17041–17045, without learning. Initial deterministic execution matched the original nine-control BC trajectory and physics bit-for-bit; all five initial stochastic episodes failed. The1,000- and2,000-step policies also failed all five stochastic samples. These are fixed-seed diagnostics, not estimated robustness rates.

Mean **initial-action** critic-minus-realized-soft-Q biases were approximately(+74.98,+74.65) initially, (+67.76,+68.84) at1,000 interactions, and (+68.05,+69.76) finally. Final critics at these sampled cold-reset actions were positive, about+12.26–+18.66, while realized soft-Q returns averaged−53.628. These measurements differ from the negative training-batch target mean on mixed replay states; neither is a maximum over actions. Each comparison uses the actual matched initial action and a complete terminal rollout with the checkpoint's frozen alpha. All18 independent diagnostic episodes truly terminated, so no unmeasured time-limit tail explains the difference. Policy/action distributions changed across checkpoints, and bias did not monotonically grow; the numerical difference is not a controlled fixed-policy calibration improvement estimate or proof of an SAC implementation error.

The final actor's mean actions changed by up to1.3602 on the reference observations, and latent standard deviations ranged0.05569–0.59550 there (reset range0.05569–0.36185). The final logged target mean was−44.5468, critic loss137.222, and alpha0.05803. This budget produced substantial policy movement without restoring the demonstrated behavior. It does not establish whether a longer run, different data coverage, or another initialization would succeed; none was tested here.

Independent value-audit reports and full actual state/action/force traces are under `value_calibration/`. Their source hashes and the run checkpoints are referenced by `summary.json`. The original seed remains one MPC demonstration; no SAC primitive or main-stage artifact is eligible from this run. All copied input hashes and all captured existing live source/config/asset hashes were verified unchanged immediately after process exit, before root was notified that deferred source edits could resume.
