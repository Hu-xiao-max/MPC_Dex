"""Collect five force-audited rotation trajectories from one frozen SAC actor.

This companion-prerequisite collector does not train a policy or change the
main method's demonstration/pretraining requirements. All five actual attempts
are retained. Only five qualifying episodes produce a primitive input archive.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
import torch

from mpc_dex.bootstrap_env import environment_for_policy
from mpc_dex.cli import read_config
from mpc_dex.contracts import canonical, environment_contract, metadata_path, sha256, validate_checkpoint
from mpc_dex.env import ASSET_ROOT, JOINT_NAMES, ROTATION_VALIDATION
from mpc_dex.learning import SACAgent
from scripts.audit_policy import PHYSICS_COLUMNS, audit_episode, snapshot_checkpoint, write_json

COLLECTION_KIND = 'frozen-simulation-sac-rotation-trajectories'
EPISODES = 5


def checkpoint_counters(checkpoint, metadata):
    """Read authenticated payload counters, never infer SAC from a stage label."""
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    counters = {name: payload.get(name) for name in ('bc_updates', 'pretrain_updates', 'online_updates')}
    if any(type(value) is not int or value < 0 for value in counters.values()):
        raise ValueError('Checkpoint requires nonnegative integer learner counters')
    if counters['online_updates'] == 0:
        raise ValueError('A SAC primitive requires actual positive online_updates; BC-only actors are ineligible')
    if payload.get('schema_version') != 1 or payload.get('obs_dim') != 60 or payload.get('action_dim') != 16:
        raise ValueError('Rotation checkpoint requires observation/action dimensions 60/16')
    if canonical(payload.get('config')) != metadata.get('learner_config'):
        raise ValueError('Checkpoint learner configuration disagrees with metadata')
    if not payload.get('std_initialized'):
        raise ValueError('Checkpoint stochastic policy standard deviation is not initialized')
    if not all(torch.isfinite(value).all().item() for value in payload['actor'].values()):
        raise ValueError('Checkpoint actor contains nonfinite parameters')
    return counters


def qualifies(result):
    return bool(result.get('environment_success') is True
        and result.get('finger_supported_rotation_success') is True
        and result.get('independent_rotation_success') is True
        and result.get('independent_drop') is False
        and result.get('environment_drop') is False
        and result.get('independent_palm_support_violation') is False
        and result.get('no_external_support') is True
        and result.get('terminated') is True and result.get('truncated') is False
        and result.get('environment_failure_reason') is None
        and result.get('model_equality_constraints') == 0
        and result.get('final_force_qualified_angle', -np.inf) >= 2 * np.pi
        and result.get('final_force_bearing_fingers', 0) >= 2)


def _integration(env):
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(env.model, spec))
    mujoco.mj_getState(env.model, env.data, state, spec)
    return state


def run_episode(agent, ec, interface, output, seed, deterministic):
    """Record initial/post-step state while the existing independent auditor runs."""
    output.mkdir()
    torch.manual_seed(seed)
    if agent.device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    rng = {'cpu': torch.get_rng_state()}
    if agent.device.type == 'cuda':
        rng['cuda'] = torch.cuda.get_rng_state_all()
    torch.save(rng, output / 'torch_rng_initial.pt')
    env = environment_for_policy(ec, interface)
    initial, joint_positions, control_times = {}, [], []
    records = {name: [] for name in ('obs', 'actions', 'next_obs', 'rewards', 'terminated',
               'truncated', 'qpos', 'qvel', 'targets', 'integration_state')}
    reset_original, step_original = env.reset, env.step
    observer = None

    def reset(*args, **kwargs):
        result = reset_original(*args, **kwargs)
        initial.update(qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(), ctrl=env.data.ctrl.copy(),
                       integration_state=_integration(env), time=float(env.data.time))
        joint_positions.append(env.data.qpos[env.q_ids].copy())
        control_times.append(0.)
        return result

    def step(action):
        nonlocal observer
        observer = getattr(env.finger_contacts, '__self__', None)
        result = step_original(action)
        obs, reward, terminated, truncated, _ = result
        for name, value in dict(next_obs=obs.copy(), rewards=reward, terminated=terminated,
                truncated=truncated, qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(),
                targets=env.data.ctrl.copy(), integration_state=_integration(env)).items():
            records[name].append(value)
        joint_positions.append(env.data.qpos[env.q_ids].copy())
        control_times.append(float(env.data.time) - initial['time'])
        return result

    class RecordingPolicy:
        def act(self, observation, deterministic=True):
            action = agent.act(observation, deterministic=deterministic)
            records['obs'].append(observation.copy())
            records['actions'].append(action.copy())
            return action

    env.reset, env.step = reset, step

    def save_transitions():
        np.savez_compressed(output / 'transitions.npz',
            **{name: np.asarray(values) for name, values in records.items()},
            **{'initial_' + name: value for name, value in initial.items()},
            joint_positions=np.asarray(joint_positions), control_times=np.asarray(control_times))

    try:
        result, physics, commands = audit_episode(env, RecordingPolicy(), seed, ec.horizon_steps, deterministic)
        result.update(model_equality_constraints=int(env.model.neq), policy_only=True,
                      policy_interface=interface, environment_contract=environment_contract(ec))
        result['eligible_for_primitive'] = qualifies(result)
        save_transitions()
        np.savez_compressed(output / 'physics.npz', samples=physics, columns=np.asarray(PHYSICS_COLUMNS))
        write_json(output / 'commands.json', commands)
        write_json(output / 'audit.json', result)
        rng = {'cpu': torch.get_rng_state()}
        if agent.device.type == 'cuda':
            rng['cuda'] = torch.cuda.get_rng_state_all()
        torch.save(rng, output / 'torch_rng_final.pt')
        return result
    except BaseException as error:
        save_transitions()
        rows = getattr(observer, 'rows', [])
        np.savez_compressed(output / 'physics.npz', samples=np.asarray(rows), columns=np.asarray(PHYSICS_COLUMNS))
        write_json(output / 'audit.json', dict(status='exception', error=str(error), eligible_for_primitive=False,
                   completed_controls=len(records['rewards']), attempted_actions=len(records['actions'])))
        raise
    finally:
        env.close()


def episode_record(output, episode, result, checkpoint_hash, counters, contract):
    directory = output / f'episode_{episode:03d}'
    with np.load(directory / 'transitions.npz', allow_pickle=False) as data:
        frames, times = data['joint_positions'].copy(), data['control_times'].copy()
        steps = len(data['actions'])
        if frames.shape != (steps + 1, 16) or times.shape != (steps + 1,):
            raise ValueError('Episode recording lacks initial or terminal measured joint frames')
        if not np.isfinite(frames).all() or not np.isfinite(times).all() or times[0] != 0 or np.any(np.diff(times) <= 0):
            raise ValueError('Episode frames and real reset-relative times must be finite and ordered')
        if steps != result['steps']:
            raise ValueError('Recorded transition count differs from the independent audit')
        np.testing.assert_array_equal(data['obs'][1:], data['next_obs'][:-1])
    evidence = {}
    for name, filename in [('transitions', 'transitions.npz'), ('physics', 'physics.npz'), ('audit', 'audit.json')]:
        path = directory / filename
        evidence[name] = {'path': str(path.relative_to(output)), 'sha256': sha256(path)}
    record = dict(episode=episode, seed=result['seed'], success=result['environment_success'],
        drop=result['environment_drop'], palm_support_violation=result['independent_palm_support_violation'],
        peak_palm_support_force=max(result['initial_palm_normal_force'], result['peak_support_normal_forces']['palm']),
        failure_reason=result['environment_failure_reason'], angle=result['final_raw_angle'],
        retained_angle=result['final_qualified_angle'], finger_contacts=result['final_finger_contacts'],
        axis_tilt_radians=result['final_axis_tilt_radians'], steps=steps, frames=len(frames),
        duration_seconds=float(times[-1]), control_times=times.tolist(), checkpoint_sha256=checkpoint_hash,
        online_updates=counters['online_updates'], learner_counters=counters, environment_contract=contract,
        policy_interface=result['policy_interface'], policy_only=True, domain_randomization=False,
        force_qualified_angle=result['final_force_qualified_angle'], force_bearing_fingers=result['final_force_bearing_fingers'],
        no_external_support=result['no_external_support'], independent_drop=result['independent_drop'],
        model_equality_constraints=result['model_equality_constraints'],
        finger_supported_rotation_success=result['finger_supported_rotation_success'],
        eligible_for_primitive=qualifies(result), evidence=evidence,
        initial_physics_sha256=result['initial_physics_sha256'],
        measured_joint_trajectory_sha256=hashlib.sha256(frames.astype('<f8').tobytes()).hexdigest())
    return record, frames, times


def export_if_eligible(output, metadata, frames):
    """No archive is emitted for incomplete, mixed-checkpoint, or failed attempts."""
    records = metadata['episodes']
    if len(records) != EPISODES or not all(record['eligible_for_primitive'] for record in records):
        return False
    identity = metadata['checkpoint']['sha256']
    counters = metadata['learner_counters']
    if counters['online_updates'] <= 0 or any(record['checkpoint_sha256'] != identity
            or record['learner_counters'] != counters or record['online_updates'] != counters['online_updates']
            or record['environment_contract'] != metadata['environment_contract'] for record in records):
        raise ValueError('Primitive trajectories must share one frozen SAC checkpoint and contract')
    for record, measured in zip(records, frames):
        for evidence in record['evidence'].values():
            if sha256(output / evidence['path']) != evidence['sha256']:
                raise ValueError('Episode evidence changed before primitive export')
        with np.load(output / record['evidence']['transitions']['path'], allow_pickle=False) as source:
            np.testing.assert_array_equal(measured, source['joint_positions'])
            np.testing.assert_array_equal(record['control_times'], source['control_times'])
    metadata = dict(metadata, successful_episodes=EPISODES, eligible_for_primitive=True)
    from mpc_dex.cli import validate_rotation_trajectories
    from mpc_dex.trajectory_contracts import validate_frozen_policy_collection
    pooled = np.concatenate(frames)
    validate_rotation_trajectories(pooled, metadata)
    validate_frozen_policy_collection(pooled, metadata, output)
    np.savez_compressed(output / 'successful_joint_trajectories.npz', joint_positions=pooled,
                        metadata=np.asarray(json.dumps(metadata, allow_nan=False)))
    return True


def collect(args):
    config, ec = read_config(args.config)
    if ec.task != 'rotation' or ec.object_shape != 'hexagonal' or ec.object_diameter != .04:
        raise ValueError('Companion primitive collection requires a compatible 40 mm hexagonal rotation environment')
    if not ec.terminate_on_success:
        raise ValueError('Primitive collection requires termination at the first successful full turn')
    if not np.isclose(ec.timestep, .002, rtol=0, atol=1e-12):
        raise ValueError('This collector requires the 2 ms audited physics timestep')
    source_metadata = validate_checkpoint(args.checkpoint, ec)
    counters = checkpoint_counters(args.checkpoint, source_metadata)
    if source_metadata.get('policy_interface', {}).get('kind') not in ('absolute_joint_target', 'bootstrap_joint_increment'):
        raise ValueError('Checkpoint has an unsupported or missing policy interface')
    output = Path(args.output)
    copied, snapshot = snapshot_checkpoint(output, checkpoint=args.checkpoint)
    metadata = validate_checkpoint(copied, ec)
    if checkpoint_counters(copied, metadata) != counters:
        raise ValueError('Checkpoint changed while being snapshotted')
    interface = metadata.get('policy_interface', {'kind': 'absolute_joint_target'})
    source = output / 'source'
    source.mkdir()
    shutil.copytree(ROOT / 'mpc_dex', source / 'mpc_dex', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (source / 'scripts').mkdir()
    for name in ('audit_policy.py', 'collect_policy_trajectories.py'):
        shutil.copyfile(ROOT / 'scripts' / name, source / 'scripts' / name)
    shutil.copytree(ASSET_ROOT, source / 'assets' / 'allegro')
    shutil.copyfile(args.config, output / 'input_config.json')
    if ec.initial_grasp:
        grasp = source / 'assets' / 'initial_grasp.npz'
        shutil.copyfile(ec.initial_grasp, grasp)
        ec = replace(ec, initial_grasp=str(grasp.resolve()))
    config = dict(config, environment=asdict(ec))
    write_json(output / 'config.json', config)
    contract = environment_contract(ec)
    assert contract == metadata['environment_contract']
    source_hashes = {str(path.relative_to(output)): sha256(path) for path in source.rglob('*') if path.is_file()}
    live_sources = {str(path.relative_to(ROOT)): sha256(path) for path in (ROOT / 'mpc_dex').rglob('*.py') if '__pycache__' not in path.parts}
    live_sources.update({str(path.relative_to(ROOT)): sha256(path) for path in ASSET_ROOT.rglob('*') if path.is_file()})
    live_sources.update({f'scripts/{name}': sha256(ROOT / 'scripts' / name) for name in ('audit_policy.py', 'collect_policy_trajectories.py')})
    write_json(output / 'source_hashes.json', source_hashes)
    checkpoint = dict(path='checkpoint.pt', sha256=sha256(copied), metadata_path='checkpoint.pt.metadata.json',
                      metadata_sha256=sha256(metadata_path(copied)))
    agent = SACAgent.load(copied, args.device)
    actor_before = {name: value.detach().cpu().clone() for name, value in agent.actor.state_dict().items()}
    runtime = dict(device=str(agent.device), torch_version=str(torch.__version__), numpy_version=np.__version__,
        mujoco_version=mujoco.__version__, torch_threads=torch.get_num_threads(),
        torch_interop_threads=torch.get_num_interop_threads(), float32_matmul_precision=torch.get_float32_matmul_precision(),
        cuda_matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cuda_version=torch.version.cuda, gpu=torch.cuda.get_device_name(agent.device) if agent.device.type=='cuda' else None,
        inference='singleton float32; saved policy interface; no clipping or smoothing beyond that interface')
    records, measured, timestamps = [], [], []
    report = dict(kind=COLLECTION_KIND, schema_version=1, status='running',
        source='Five actual policy-only attempts from one immutable simulation SAC checkpoint',
        implementation='independent-paper-reproduction', policy_only=True, domain_randomization=False,
        eligible_for_primitive=False, successful_episodes=0, requested_episodes=EPISODES,
        task='rotation', environment=asdict(ec), environment_contract=contract, policy_interface=interface,
        learner_counters=counters, checkpoint=checkpoint, snapshot=snapshot, deterministic=not args.stochastic,
        seed=args.seed, runtime=runtime, joint_order=JOINT_NAMES, units='radians', rotation_validation=ROTATION_VALIDATION,
        alignment=dict(method='reset_relative_original_samples', interpolation=False, temporal_warping=False,
            author_algorithm_known=False, variable_durations_preserved=True),
        episode_timestamps=timestamps, episodes=records, numerical_source_hashes=source_hashes,
        robustness_rate_claim=False, reset_diversity_claim=False,
        duration_note='Actual durations are retained; the published approximate minute is not interpreted or manufactured.',
        collection_script_sha256=sha256(Path(__file__)))
    write_json(output / 'report.json', report)
    try:
        for episode in range(EPISODES):
            result = run_episode(agent, ec, interface, output / f'episode_{episode:03d}', args.seed+episode, not args.stochastic)
            if result['environment_contract'] != contract or result['policy_interface'] != interface:
                raise ValueError('Episode environment or policy interface changed during collection')
            result.update(checkpoint_sha256=checkpoint['sha256'], learner_counters=counters,
                          online_updates=counters['online_updates'])
            write_json(output / f'episode_{episode:03d}' / 'audit.json', result)
            record, frames, times = episode_record(output, episode, result, checkpoint['sha256'], counters, contract)
            records.append(record); measured.append(frames); timestamps.append(times.tolist())
            report.update(completed_episodes=len(records), successful_episodes=sum(row['eligible_for_primitive'] for row in records))
            write_json(output / 'report.json', report)
        if any(sha256(output / name) != digest for name, digest in source_hashes.items()) or any(
                sha256(ROOT / name) != digest for name, digest in live_sources.items()):
            raise ValueError('Numerical sources changed during collection; no primitive archive is eligible')
        if sha256(copied) != checkpoint['sha256'] or sha256(metadata_path(copied)) != checkpoint['metadata_sha256']:
            raise ValueError('Frozen checkpoint changed during collection')
        if any(getattr(agent, name) != value for name, value in counters.items()) or any(
                not torch.equal(value, agent.actor.state_dict()[name].detach().cpu()) for name, value in actor_before.items()):
            raise ValueError('The in-memory actor or learner counters changed during collection')
        report.update(status='complete', unique_initial_physics_states=len({r['initial_physics_sha256'] for r in records}),
            unique_measured_joint_trajectories=len({r['measured_joint_trajectory_sha256'] for r in records}),
            total_simulated_seconds=sum(r['duration_seconds'] for r in records),
            repeated_episodes_note='Five actual executions; identical resets or trajectories are not evidence of diversity or robustness.')
        eligible = export_if_eligible(output, report, measured)
        report['eligible_for_primitive'] = eligible
        if eligible:
            report['trajectory_archive_sha256'] = sha256(output / 'successful_joint_trajectories.npz')
        write_json(output / 'report.json', report)
        write_json(output / 'manifest.json', {str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob('*'))
                   if p.is_file() and '__pycache__' not in p.parts and p.name not in ('manifest.json', 'run.log')})
        return 0 if eligible else 2
    except BaseException as error:
        report.update(status='failed', eligible_for_primitive=False, error=str(error))
        write_json(output / 'report.json', report)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=1000000)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--stochastic', action='store_true', help='Sample the frozen policy; deterministic means are the default')
    parser.add_argument('--torch-threads', type=int, default=1)
    args = parser.parse_args(argv)
    if args.torch_threads < 1:
        parser.error('--torch-threads must be positive')
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return collect(args)


if __name__ == '__main__':
    raise SystemExit(main())
