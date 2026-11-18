# Physical test of the fitted critic direction

The fitted mean-critic gradient predicts that the plus direction is better for all five seeds. Actual matched-noise physics favors plus over minus in only two of these five prescribed comparisons. All ten interventions fail; none completes a full turn. These samples do not establish reliable action directions or justify actor unfreezing.

Exactly one first-action change was applied per branch: the actual original sampled first action plus or minus0.005 times the L2-normalized gradient of the two fitted critics’ mean. The rest of each trajectory uses the unchanged original actor, latent std.05, alpha.05, strict100-control horizon, unchanged force/retention/tilt criteria, and the same seeded Gaussian stream. No training, step-size sweep, or adaptive continuation occurred.

| Seed | Predicted Q(+)-Q(−) | Actual soft return(+)-return(−) | Actual + vs baseline | Actual − vs baseline | + / − controls | + / − failure |
|---|---:|---:|---:|---:|---|---|
| 17041 | +1.546600 | -2.100538 | -2.515965 | -0.415427 | 2 / 2 | drop / drop |
| 17042 | +11.110371 | +16.323257 | +65.531157 | +49.207899 | 2 / 4 | palm_support / drop |
| 17043 | +1.288782 | +6.258227 | +16.603682 | +10.345455 | 3 / 5 | drop / drop |
| 17044 | +5.494324 | -0.776140 | -0.492011 | +0.284129 | 3 / 3 | palm_support / palm_support |
| 17045 | +2.355576 | -15.080573 | -15.879314 | -0.798741 | 5 / 2 | palm_support / palm_support |

Positive predicted deltas are across the same two critics at actual candidate first actions, not a maximum over critics or actions. Actual returns are complete terminal soft-Q returns: discounted rewards minus discounted frozen-alpha log probabilities of actual future policy actions, excluding the first action entropy. The first action is forced; its unexecuted baseline sample and log probability are recorded separately. There were no horizon truncations or invented bootstrap tails.

Both directions improve over the baseline for seeds17042 and17043, despite the opposite Q predictions. Some larger return gains accompany earlier failure rather than a demonstrated task improvement. The maximum force-qualified progress over all branches is only1.04936rad. Thus even a return improvement in these examples is not a successful primitive or policy-only success.

All original baseline actions, actual log probabilities, and final RNG states were reconstructed exactly from archived observations, without rerunning baseline physics. Each physical branch starts from exactly the original integration state. Standardized Gaussian draws match bit-for-bit on steps actually executed by both the baseline and intervention; beyond original termination, interventions simply continue the prescribed seed stream. No baseline post-terminal actions or draws were fabricated. Plus/minus common executed steps also have identical Gaussian draws.

Only physical normalized bounds[-1,1] were available for clipping; no dimension clipped. Achieved radii range0.004999994248–0.005000008762 because actions are float32. Independent re-evaluation verifies every gradient, intervention action, future policy sample/log probability, RNG endpoint, terminal return, and control-boundary physics/integration state. Full commands, observations, rewards, flags, Gaussian draws, physics/forces, checkpoint/config/source copies, and hashes are retained.

These are five fixed-seed paired action interventions, not a robustness-rate estimate. No actor, critic, alpha, optimizer, or source file changed. The original uncentered and centered500-update calibration artifacts remain immutable.
