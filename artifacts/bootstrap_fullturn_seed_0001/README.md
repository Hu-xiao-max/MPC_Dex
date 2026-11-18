# Verified bootstrap demonstration seed

This artifact is one actual 40 mm MPC diagnostic demonstration, **not a SAC
policy demonstration, production primitive, or the main method's 20-episode
MPC dataset**. No training was launched while preparing it.

- Input: `artifacts/mpc_matched40_reference_0001/transitions.npz` (60 commands).
- Used: the first 19 commands, ending at the first actual strict full-turn
  success, after 1.9 simulated seconds.
- Unused: the remaining 41 commands. They do not become extra episodes.
- Replay: `replay.npz`, one complete episode with 19 transitions, 18 known
  executed successor actions, and an invalid terminal next action.
- Actual recorded observations, next observations, qpos and qvel reproduced
  with zero error; rewards and endpoint flags were recomputed by the simulator.
- Independent force-qualified angle: 6.630861641641375 rad; three finger
  contacts; final axis tilt 0.10165464363391391 rad. Peak palm, table and other
  normal support forces were all zero; no drop or palm violation occurred.

`seed_config.json` preserves the original strict rotation environment and uses
the absolute joint-target interface. Its provenance records the source config
and the explicit removal policy for `bootstrap.control`; that block was absent
from the supplied strict config at preparation time. The original configuration
is preserved in `original_strict_config.json`.

`replay.npz.metadata.json` contains the model/grasp/source hashes, preparation
versions, per-input first-success boundary and independent audit report.
`replay_audit/episode_000/` preserves the complete original input, executed
prefix with initial state, per-physics-step force audit and commands.

The bootstrap trainer accepts this replay through `bootstrap.seed.replay` in an
absolute-interface config with the matching environment contract. Seed
demonstration counts stay separate from online steps, SAC success counts and
SAC trajectory exports. BC/critic update counts remain an explicit future
training decision.

Replay SHA-256:
`e20aee06f218396486785efeecf98ef45fa16f38e107ee4b6cbf3e2b3979eb4e`
