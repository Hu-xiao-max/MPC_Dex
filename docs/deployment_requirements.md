The requested paper describes a robot learning and control system. This workspace runs an independent simulation controller with audited full-turn demonstrations and one successful BC-policy turn. A successful SAC-trained policy, sustained learned control and the requested hardware result remain unverified. Values and unresolved fields are recorded in [paper_reference.json](../configs/paper_reference.json).

Sources checked on 2026-09-23 are the [RL paper, v1](https://arxiv.org/html/2609.14878v1), its [project implementation tables](https://real-world-rl-with-mpc.github.io/), and the [companion MPC paper, v1](https://arxiv.org/html/2609.14868v1). The project HTML checksum is in the JSON because the website is mutable. Each configuration group identifies its source. Null means unresolved or unavailable, never an implicit default.

The published system uses an Allegro hand, a Franka FR3 mounting arm, OptiTrack object pose, and a synchronized MuJoCo model. Continuous rotation targets a 45 mm hexagonal object and sends 16 joint position commands at 10 Hz. Deployment needs the actual hand, tracking and target-machine interface details or an explicitly chosen simulation target. A simulation run cannot establish the paper's real-world outcome. [RL paper, III-A](https://arxiv.org/html/2609.14878v1)

The following dependencies remain to be supplied or independently implemented and verified; the inspected source material does not supply deployable asset files.

| Required input | What must be established before the relevant run |
| --- | --- |
| Runtime source | Repository/version, dependencies, entry point, and supported machine/device interfaces. |
| Trained actor | Checkpoint plus architecture, preprocessing, task identity, and action mapping for policy execution; otherwise the training stages remain necessary. |
| Robot and object model | MuJoCo model and meshes, calibrated object properties, joint names, actuator definitions, limits and gains. |
| Observation/action contract | Joint order/signs, quaternion order/frame, tracking transforms, angle sign/unwrapping, finite-difference timing/filtering, normalization, and normalized-action conversion. |
| Primitive data | PCA basis, mean posture, coefficient ranges and joint ordering, or the original extraction data and procedure needed to regenerate them. |
| Training replay | Recorded transitions and actual next demonstration actions for Soft-SARSA, or a functioning hardware MPC collection path. |
| Task lifecycle | Initial grasp, reset operation, drop detection and success/time-limit semantics, including what is recorded at terminal transitions. |
| Target access | A selected execution host and access to the required hardware or simulation environment. |

Policy inference alone can use a compatible trained actor without running MPC. Reproducing training requires MPC collection, mean-only behavior cloning, frozen-actor Soft-SARSA critic pretraining, and SAC with retained offline replay and temporary MPC guidance. New-object adaptation starts from the learned actor and collects fresh online experience without reusing the nominal replay buffer. The reorientation experiment adds a goal observation; its exact serialized input contract still needs confirmation. [RL paper, II and III-C–D](https://arxiv.org/html/2609.14878v1)

Two implementation details require explicit resolution:

- Target critic selection conflicts across sources. Paper II-B2 describes a minimum over ten target critics and retains that architecture online. The project table specifies two randomly sampled critics per target from the ten-network ensemble. Both descriptions are preserved, and neither is silently selected.
- The paper's observation terms sum to 60 if orientation uses four quaternion components: `16 + 3 + 4 + 1 + 1 + 16 + 3 + 16`. This agrees with the project table. It does not establish quaternion ordering, reference frames, joint order, normalization, or the actual runtime array layout.

The companion controller's published horizon, sample budget and primitive settings are recorded separately. Its rotation primitive was extracted using a 40 mm object; that is different from the RL experiment's 45 mm object. The primitive arrays and calibrated model cannot be reconstructed from their dimensions. Its low-level command and synchronization rates also serve different purposes from the RL policy's 10 Hz decisions. Companion parameters must be checked against the actual RL controller assets. [Companion paper, II–III](https://arxiv.org/html/2609.14868v1)

Deployment can be verified only after the selected target runs the intended task with the resolved observation/action contract and genuine model/checkpoint assets. Evidence should identify versions and assets, show sustained operation at the required control rate, and record policy-only task outcomes. If reproducing the training pipeline, preserve collection, pretraining, replay and scheduling evidence as well. A JSON parse, import test or synthetic rollout proves only that narrow check; it cannot establish trained robot behavior.
