from pathlib import Path
from dataclasses import asdict, replace
import json,sys
D=Path(__file__).resolve().parent
sys.path.insert(0,str(D/'source'))
import numpy as np
from mpc_dex.env import AllegroRotationEnv,EnvConfig
from mpc_dex.contracts import sha256,environment_contract
from scripts.audit_policy import PHYSICS_COLUMNS,audit_episode,write_json
from scripts.audit_recorded_controls import RecordedPolicy

class RecordingAgent:
    def __init__(self, agent):
        self.agent = agent
        self.obs = []
        self.actions = []

    def act(self, obs, deterministic=True):
        action = self.agent.act(obs, deterministic=deterministic)
        self.obs.append(obs.copy()); self.actions.append(action.copy())
        return action


def run_audit(agent, ec, output, limit):
    output.mkdir()
    env = AllegroRotationEnv(ec)
    policy = RecordingAgent(agent)
    original_step = env.step
    next_obs, rewards, dones, truncations, qpos, qvel, targets = [], [], [], [], [], [], []
    initial = {}
    original_reset = env.reset

    def capture_reset(*args, **kwargs):
        result = original_reset(*args, **kwargs)
        initial.update(qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(),
                       ctrl=env.data.ctrl.copy(), joint_positions=env.data.qpos[env.q_ids].copy())
        return result

    def capture_step(action):
        result = original_step(action)
        ob, reward, done, truncated, info = result
        next_obs.append(ob.copy()); rewards.append(reward); dones.append(done); truncations.append(truncated)
        qpos.append(env.data.qpos.copy()); qvel.append(env.data.qvel.copy()); targets.append(env.data.ctrl.copy())
        return result

    env.reset = capture_reset
    env.step = capture_step
    try:
        result, physics, commands = audit_episode(env, policy, 42, limit, True)
        result.update(production_eligible=False, policy_kind='recorded-control-replay',
                      environment=asdict(ec), model_equality_constraints=int(env.model.neq))
        np.savez_compressed(output / 'transitions.npz', obs=np.asarray(policy.obs),
            actions=np.asarray(policy.actions), next_obs=np.asarray(next_obs), rewards=np.asarray(rewards),
            terminated=np.asarray(dones), truncated=np.asarray(truncations),
            qpos=np.asarray(qpos), qvel=np.asarray(qvel), targets=np.asarray(targets),
            initial_qpos=initial['qpos'], initial_qvel=initial['qvel'], initial_ctrl=initial['ctrl'],
            initial_joint_positions=initial['joint_positions'])
        np.savez_compressed(output / 'physics.npz', samples=physics, columns=np.asarray(PHYSICS_COLUMNS))
        write_json(output / 'commands.json', commands)
        write_json(output / 'report.json', result)
        return result, np.asarray(policy.actions), np.asarray(policy.obs)
    finally:
        env.close()


def full_turn_events(output, ec):
    with np.load(output / 'physics.npz') as archive:
        samples = archive['samples']; columns = archive['columns'].tolist()
    x = {name: samples[:, columns.index(name)] for name in columns}
    failed = np.maximum.accumulate((x['table_contacts'] > 0) | (x['object_displacement'] > .18) |
        (x['contactless_seconds'] >= ec.contact_loss_seconds - 1e-10) |
        (x['palm_support_violation'] > 0))
    externally_supported = np.zeros(len(samples), dtype=bool)
    for name in ['palm_normal_force', 'table_normal_force', 'other_normal_force',
                 'palm_force_norm', 'table_force_norm', 'other_force_norm',
                 'object_applied_wrench_norm', 'object_applied_generalized_force_norm']:
        externally_supported |= np.maximum.accumulate(x[name] > 1e-6)
    eligible = (~failed) & (~externally_supported) & (x['force_retained'] > 0)
    turns = []
    for count in range(1, max(1, int(max(0., x['force_qualified_angle'].max()) // (2*np.pi))) + 1):
        valid = eligible & (x['raw_angle'] >= count*2*np.pi) & (x['force_qualified_angle'] >= count*2*np.pi)
        indices = np.flatnonzero(valid)
        boundaries = np.arange(round(ec.control_seconds/ec.timestep)-1, len(samples), round(ec.control_seconds/ec.timestep))
        valid_boundaries = boundaries[valid[boundaries]]
        if not len(indices):
            continue
        i = int(indices[0]); j = int(valid_boundaries[0]) if len(valid_boundaries) else None
        turns.append(dict(turn=count, first_physics_time_seconds=float(x['time'][i]),
            force_qualified_angle=float(x['force_qualified_angle'][i]), raw_angle=float(x['raw_angle'][i]),
            first_control_boundary_seconds=float(x['time'][j]) if j is not None else None,
            force_bearing_fingers=int(x['force_bearing_fingers'][i]),
            axis_tilt_radians=float(x['axis_tilt_radians'][i])))
    return turns


