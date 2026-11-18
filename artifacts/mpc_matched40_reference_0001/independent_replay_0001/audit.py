from pathlib import Path
from dataclasses import asdict
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


def main():
    ec = EnvConfig(**json.loads((D/'config.json').read_text())['environment'])
    original_report = json.loads((D/'original_report.json').read_text())
    assert environment_contract(ec) == original_report['environment_contract']
    assert not ec.terminate_on_success
    with np.load(D/'original_transitions.npz', allow_pickle=False) as a:
        originals = {key: a[key].copy() for key in a.files}
    actions = originals['actions']
    assert actions.shape == (60, 16)
    report = dict(purpose='Independent replay and clipping sensitivity audit of actual continuous diagnostic MPC commands',
        production_eligible=False, learned_SAC_policy=False, training_updates=0,
        source=str(D.parent/'transitions.npz'), source_sha256=sha256(D/'original_transitions.npz'),
        audit_source_sha256=sha256(__file__), environment_contract=environment_contract(ec),
        settings=dict(seed=42, original_controls=60, posthold_controls=20, epsilons=[.02,1e-6]),
        saturation=dict(exact=int((abs(actions)==1).sum()), total=int(actions.size),
                        fraction=float((abs(actions)==1).mean()),
                        exact_per_joint=(abs(actions)==1).sum(0).tolist()),
        evaluations={})
    runs=[('exact_60', actions), ('posthold_20',np.vstack([actions,np.repeat(actions[-1:],20,axis=0)])),
          ('clipped_002',np.clip(actions,-1+.02,1-.02)),
          ('clipped_0000001',np.clip(actions,-1+1e-6,1-1e-6))]
    for label, commands in runs:
        result, _, _ = run_audit(RecordedPolicy(commands), ec, D/label, len(commands))
        result['full_turn_events'] = full_turn_events(D/label, ec)
        result['ever_force_qualified_full_turn'] = bool(result['full_turn_events'])
        if label=='exact_60':
            with np.load(D/label/'transitions.npz') as replay:
                differences={key:float(np.max(abs(replay[key]-originals[key]))) for key in originals}
            with np.load(D/'original_physics.npz') as a, np.load(D/label/'physics.npz') as b:
                assert np.array_equal(a['columns'],b['columns'])
                differences['physics'] = float(np.max(abs(a['samples']-b['samples'])))
                differences['physics_bitexact'] = bool(np.array_equal(a['samples'],b['samples']))
            result['original_replay_differences']=differences
            assert all(value==0 for key,value in differences.items() if key!='physics_bitexact'), differences
        if label.startswith('clipped'):
            result['changed_scalar_actions']=int((commands!=actions).sum())
            result['max_normalized_action_change']=float(abs(commands-actions).max())
            result['mean_squared_action_change']=float(np.mean((commands-actions)**2))
        if label=='posthold_20':
            result['requested_posthold_seconds']=2.
            result['completed_posthold_seconds']=max(0.,result['simulated_seconds']-6.)
            result['completed_full_posthold']=bool(result['steps']==80 and not result['independent_drop'] and
                                                   not result['independent_palm_support_violation'])
        write_json(D/label/'report.json',result)
        report['evaluations'][label]=result
        write_json(D/'report.json',report)
        print(json.dumps(dict(label=label,result=result)),flush=True)
    report['status']='complete'; write_json(D/'report.json',report)
    original_hashes=json.loads((D/'inputs_manifest.json').read_text())
    assert all(sha256(D/key)==value for key,value in original_hashes.items())
    manifest={str(p.relative_to(D)):sha256(p) for p in sorted(D.rglob('*'))
              if p.is_file() and '__pycache__' not in p.parts and p.name not in ('manifest.json','audit.log')}
    write_json(D/'manifest.json',manifest)


if __name__=='__main__':main()

