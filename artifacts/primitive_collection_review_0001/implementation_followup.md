# Implemented collection and provenance gate

The original review and command templates record the pre-fix state and remain unchanged. The dedicated `scripts/collect_policy_trajectories.py` now copies one compatible checkpoint, sidecar, environment configuration and source snapshot before five actual policy-only attempts. A checkpoint with zero actual online SAC updates is rejected before execution. Initial and terminal measured joint frames, full transitions, MuJoCo integration states, actual reset-relative timestamps, force audits and hashes remain available for every attempt, including failures.

Only five independently qualified episodes from that one checkpoint produce `successful_joint_trajectories.npz`. Extraction and the main planner now recheck the frozen collection contract, immutable evidence identities, measured state consistency, force qualification and derived PCA. Generic evaluation and online training exports remain diagnostic and cannot supply this primitive. The main method's 20-demonstration, 12,000-BC-update and 50,000-critic-pretraining requirements are unchanged.

For a future independently successful strict 40 mm SAC checkpoint, the concrete collection command is:

```sh
.venv/bin/python scripts/collect_policy_trajectories.py \
  --checkpoint SAC40_CHECKPOINT --config SAC40_CONFIG \
  --output artifacts/frozen_sac40_collection_0001 \
  --seed 1000000 --device cuda --torch-threads 1
```

The default is deterministic singleton float32 inference, one intra/inter-op thread and TF32 disabled. `--stochastic` explicitly selects policy sampling. Exit status 2 means the five attempts completed but did not all qualify; it is not an eligible collection. No archive is synthesized from partial episodes. If all five qualify, extract the measured archive directly:

```sh
.venv/bin/python -m mpc_dex extract-primitive \
  --trajectory artifacts/frozen_sac40_collection_0001/successful_joint_trajectories.npz \
  --output artifacts/rotation_primitive.npz
```

The alignment is explicitly `reset_relative_original_samples`, without interpolation, padding or temporal warping. Actual durations and repeated initial states/trajectories are reported. This local sampling convention is not a claim about the unpublished author alignment algorithm or policy robustness.

The real integration in `artifacts/bootstrap_0005_frozen_collection_0001` used final checkpoint SHA-256 `386be7a4e915a3b54e7bd7fc534c5b9502b84f35fd5e7139f7ec3c6ddeabc220` with BC 12,000 / pretrain 0 / actual online SAC 1,745 updates. All five deterministic attempts failed on palm support at control 7, with force-qualified angle 1.0032547032686967 rad and peak palm normal force 1.7862867488820415 N. Every transition and all 350 physics samples in each episode match the independent final-checkpoint audit bit for bit. No eligible archive was exported and no training ran.

The final focused collector/extraction regression run passed 50 tests. Successful assembly and extraction tests are expressly synthetic schema fixtures, not evidence of a physical turn. Their serialized MuJoCo states are internally consistent, and corruption tests reject contradictory state arrays, missing initial state, changed checkpoint/audit provenance, missing episodes, support/contact-loss violations, and tampered evidence or PCA.
