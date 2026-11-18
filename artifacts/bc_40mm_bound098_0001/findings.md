The action-restricted teacher passes independent validation, but the bounded paper-settings BC baseline does not reproduce its full turn in closed loop.

All 60 recorded controls replay bit-exact in observations, actions, rewards, next observations, qpos/qvel, and all 3,000 physics samples. The final force-qualified angle is 7.727621 rad with no palm contact or drop. The strict first-success prefix contains 22 controls and ends at raw angle 6.509392 rad, force-qualified angle **6.287875 rad**, and three loaded fingers. A separate two-second fixed-target hold passes both after that prefix (final 6.698618 rad) and after all 60 controls (final 7.637516 rad), with no palm/drop in either case.

Every recorded action is **bitwise unchanged by the actual Torch epsilon=.02 clamp**. Maximum magnitude is .9799998998641968. This planner action restriction is an explicit local diagnostic choice; the object, contact model, no-palm threshold, and rotation qualification are unchanged.

The [authors' implementation table](https://real-world-rl-with-mpc.github.io/) specifies BC learning rate .003, batch size 256, 12,000 mean-only updates, epsilon .02, and standard deviation .3 afterward. The cached primary table and repository reference configuration are archived with hashes. The single baseline uses those BC settings, fixed seed 17001, uniform sampling with replacement within the actual 22-transition episode, the existing 60→256→256 actor, CUDA, and one Torch CPU thread. Two unused critics are retained solely for checkpoint compatibility; their count does not alter BC. This is one local simulation episode, not the paper's 20 hardware demonstrations.

| BC updates | Latent target MSE | Teacher action MSE | Maximum action error | Peak force-qualified angle | Strict result |
|---|---:|---:|---:|---:|---|
| 1,000 | .0158217 | .00558018 | .760178 | 1.199156 rad | Palm at .2 s |
| 3,000 | .0156362 | .00519606 | .718364 | 1.025369 rad | Palm at .3 s |
| 6,000 | .0154956 | .00517098 | .719491 | .462512 rad | Palm at .2 s |
| 9,000 | .0153552 | .00524077 | .726506 | .634921 rad | Palm at .2 s |
| 12,000 | 5.29e-12 | 1.97e-12 | 1.35e-5 | 4.142790 rad | Palm at .4 s |

The final actor fits the recorded means accurately but still begins with a 1.350e-5 normalized action error at the exact initial state. After one physical control, its maximum observation error is .00011112. Its second command then differs from the teacher by .00111711, its third by .0179069, and it hits the palm on control four. Thus the remaining fitting error and feedback sensitivity are practically relevant despite the low mean-square loss. The independently performed command-noise sensitivity audit is separate; this baseline ran no noise sweep or alternate optimizer settings.

No clone passed the strict independent full-turn check, so none qualified for a two-second continuous-policy aftermath. No critic, target critic, alpha, or online update occurred; hashes remain unchanged. All five checkpoints, exact sampled minibatch indices, losses, teacher fits, actual policy rollout transitions, and per-physics traces are saved under `bc_paper/`. `audit_report.json` preserves the teacher evidence; `fit_diagnosis.json` preserves actual error growth. Shared numerical sources were not changed. This is a BC diagnostic, not a SAC-trained policy or production primitive.
