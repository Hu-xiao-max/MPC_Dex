"""Verify the frozen simulation-SAC evidence used for rotation primitives.

General physical trajectory validation lives in cli.py. This additional gate
binds five measured episodes to one trained checkpoint and their force audits.
It does not turn hashes or synthetic test fixtures into proof of robot behavior.
"""
from pathlib import Path
import json

import numpy as np
import torch
import mujoco

from .contracts import metadata_path, sha256, validate_checkpoint
from .env import EnvConfig, JOINT_NAMES, make_model


COLLECTION_KIND = 'frozen-simulation-sac-rotation-trajectories'


def _evidence_file(root, record, label):
    if not isinstance(record, dict) or set(record) != {'path', 'sha256'}:
        raise ValueError(f'Missing {label} evidence identity')
    name, digest = record['path'], record['sha256']
    if (not isinstance(name, str) or not name or Path(name).is_absolute()
            or not isinstance(digest, str) or len(digest) != 64
            or any(c not in '0123456789abcdef' for c in digest)):
        raise ValueError(f'Invalid {label} evidence identity')
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file() or sha256(path) != digest:
        raise ValueError(f'{label} evidence is missing, changed or outside the collection')
    return path


def validate_frozen_policy_collection(frames, metadata, artifact_root):
    """Require auditable five-trajectory, one-checkpoint simulation SAC data."""
    root = Path(artifact_root).resolve()
    if (metadata.get('kind') != COLLECTION_KIND or metadata.get('schema_version') != 1
            or metadata.get('eligible_for_primitive') is not True
            or metadata.get('policy_only') is not True
            or metadata.get('domain_randomization') is not False):
        raise ValueError('Primitive extraction requires a verified frozen simulation SAC collection')
    records = metadata.get('episodes')
    if metadata.get('successful_episodes') != 5 or not isinstance(records, list) or len(records) != 5:
        raise ValueError('Companion primitive extraction requires five successful frozen-policy trajectories')
    ec = EnvConfig(**metadata['environment'])
    if (ec.task != 'rotation' or ec.object_shape != 'hexagonal' or ec.object_diameter != .04
            or not ec.terminate_on_success):
        raise ValueError('Frozen primitive collection requires the strict 40 mm hexagonal rotation task')
    alignment = metadata.get('alignment', {})
    if (alignment.get('method') != 'reset_relative_original_samples'
            or alignment.get('interpolation') is not False):
        raise ValueError('Collection must identify original measured samples and reset-relative alignment')
    checkpoint = metadata.get('checkpoint', {})
    path = _evidence_file(root, {k: checkpoint.get(k) for k in ('path', 'sha256')}, 'checkpoint')
    sidecar = _evidence_file(root, {'path': checkpoint.get('metadata_path'),
                                  'sha256': checkpoint.get('metadata_sha256')}, 'checkpoint metadata')
    if sidecar != metadata_path(path):
        raise ValueError('Checkpoint sidecar path does not match its checkpoint')
    checkpoint_record = validate_checkpoint(path, ec)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    counters = {key: payload.get(key) for key in ('bc_updates', 'pretrain_updates', 'online_updates')}
    if (any(type(value) is not int or value < 0 for value in counters.values())
            or counters['online_updates'] == 0 or metadata.get('learner_counters') != counters):
        raise ValueError('Collection must preserve actual SAC updates; BC-only histories cannot supply this primitive')
    if (payload.get('schema_version') != 1 or payload.get('obs_dim') != 60 or payload.get('action_dim') != 16
            or metadata.get('policy_interface') != checkpoint_record.get('policy_interface')
            or metadata.get('environment_contract') != checkpoint_record.get('environment_contract')
            or json.loads(json.dumps(payload.get('config'))) != checkpoint_record.get('learner_config')):
        raise ValueError('Collection and frozen checkpoint contracts disagree')
    model = make_model(ec)
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state_size = mujoco.mj_stateSize(model, state_spec)
    state_data = mujoco.MjData(model)
    q_ids = np.array([model.joint(name).qposadr[0] for name in JOINT_NAMES])
    frame_cursor = 0
    evidence_paths = set()
    episode_times = metadata.get('episode_timestamps')
    if not isinstance(episode_times, list) or len(episode_times) != 5:
        raise ValueError('Collection must preserve measured timestamps for all five episodes')
    for index, record in enumerate(records):
        if (record.get('checkpoint_sha256') != checkpoint['sha256']
                or record.get('online_updates') != counters['online_updates']):
            raise ValueError('Trajectory episodes must come from the same frozen trained checkpoint')
        evidence = record.get('evidence', {})
        paths = {name: _evidence_file(root, evidence.get(name), name)
                 for name in ('transitions', 'physics', 'audit')}
        if any(path in evidence_paths for path in paths.values()):
            raise ValueError('Each trajectory needs its own recorded execution evidence')
        evidence_paths.update(paths.values())
        audit = json.loads(paths['audit'].read_text())
        if (audit.get('checkpoint_sha256') != checkpoint['sha256']
                or audit.get('learner_counters') != counters
                or audit.get('online_updates') != counters['online_updates']
                or audit.get('environment_contract') != metadata['environment_contract']
                or audit.get('policy_interface') != metadata['policy_interface']
                or audit.get('policy_only') is not True):
            raise ValueError('Episode audit provenance disagrees with its frozen checkpoint')
        if (audit.get('finger_supported_rotation_success') is not True
                or audit.get('environment_success') is not True
                or audit.get('independent_drop') is not False or audit.get('environment_drop') is not False
                or audit.get('independent_palm_support_violation') is not False
                or audit.get('no_external_support') is not True
                or audit.get('environment_failure_reason') is not None
                or audit.get('terminated') is not True or audit.get('truncated') is not False
                or record.get('model_equality_constraints') != 0
                or record.get('no_external_support') is not True
                or record.get('independent_drop') is not False):
            raise ValueError('Every collected trajectory needs a successful independent force audit')
        steps = record.get('steps')
        if type(steps) is not int or steps < 1 or audit.get('steps') != steps:
            raise ValueError('Trajectory and force-audit control counts disagree')
        with np.load(paths['transitions'], allow_pickle=False) as archive:
            data = {key: archive[key].copy() for key in archive.files}
        expected = {'obs': (steps, 60), 'next_obs': (steps, 60), 'actions': (steps, 16),
                    'joint_positions': (steps + 1, 16), 'control_times': (steps + 1,),
                    'rewards': (steps,), 'qpos': (steps, model.nq), 'qvel': (steps, model.nv),
                    'targets': (steps, model.nu), 'integration_state': (steps, state_size),
                    'initial_qpos': (model.nq,), 'initial_qvel': (model.nv,),
                    'initial_ctrl': (model.nu,), 'initial_integration_state': (state_size,), 'initial_time': ()}
        if any(key not in data or data[key].shape != shape or not np.isfinite(data[key]).all()
               for key, shape in expected.items()):
            raise ValueError('Invalid measured transition or joint-frame dimensions')
        if (np.any(abs(data['actions']) > 1) or not np.array_equal(data['obs'][1:], data['next_obs'][:-1])
                or data.get('terminated', np.empty(0)).shape != (steps,)
                or data.get('truncated', np.empty(0)).shape != (steps,)
                or not np.array_equal(data['terminated'], np.arange(steps) == steps - 1)
                or data['truncated'].any()):
            raise ValueError('Measured episode continuity or terminal boundaries are invalid')
        actual_joints = np.vstack((data['initial_qpos'][q_ids], data['qpos'][:, q_ids]))
        if (not np.array_equal(data['joint_positions'], actual_joints)
                or not np.array_equal(data['obs'][:, :16], actual_joints[:-1].astype(data['obs'].dtype))
                or not np.array_equal(data['next_obs'][:, :16], actual_joints[1:].astype(data['next_obs'].dtype))):
            raise ValueError('Measured joint frames disagree with executed states or observations')
        times = data['control_times']
        if (not np.array_equal(np.asarray(episode_times[index]), times) or times[0] != 0
                or not np.allclose(times, np.arange(steps + 1) * ec.control_seconds, rtol=0, atol=1e-9)
                or not np.isclose(record.get('duration_seconds', np.nan), times[-1], rtol=0, atol=1e-9)):
            raise ValueError('Trajectory timing or alignment disagrees with actual execution')
        for step in range(steps + 1):
            state = data['initial_integration_state'] if step == 0 else data['integration_state'][step - 1]
            mujoco.mj_setState(model, state_data, state, state_spec)
            for name, field in (('qpos', 'qpos'), ('qvel', 'qvel'), ('ctrl', 'targets')):
                expected_value = data['initial_' + name] if step == 0 else data[field][step - 1]
                if not np.array_equal(getattr(state_data, name), expected_value):
                    raise ValueError('Recorded state arrays disagree with the full simulator state')
            if not np.isclose(state_data.time - float(data['initial_time']), times[step], rtol=0, atol=1e-9):
                raise ValueError('Recorded simulator state time disagrees with control timestamps')
        stop = frame_cursor + steps + 1
        if not np.array_equal(frames[frame_cursor:stop], data['joint_positions']):
            raise ValueError('Extraction frames differ from recorded measured joint positions')
        frame_cursor = stop
        with np.load(paths['physics'], allow_pickle=False) as archive:
            samples = archive['samples']; columns = archive['columns'].tolist()
        if (samples.ndim != 2 or samples.shape != (steps * round(ec.control_seconds / ec.timestep), len(columns))
                or len(set(columns)) != len(columns) or not np.isfinite(samples).all()):
            raise ValueError('Independent physics evidence has invalid coverage')
        col = {name: samples[:, i] for i, name in enumerate(columns)}
        support = ('palm_normal_force', 'table_normal_force', 'other_normal_force',
                   'palm_force_norm', 'table_force_norm', 'other_force_norm',
                   'object_applied_wrench_norm', 'object_applied_generalized_force_norm')
        needed = ('time', 'raw_angle', 'qualified_angle', 'force_qualified_angle',
                  'finger_contacts', 'force_bearing_fingers', 'axis_tilt_radians',
                  'retained', 'force_retained', 'twist_increment', 'palm_support_violation',
                  'table_contacts', 'object_displacement', 'contactless_seconds', *support)
        if any(name not in col for name in needed):
            raise ValueError('Independent physics evidence lacks force/support measurements')
        retained = ((col['finger_contacts'] >= 2)
                    & (col['axis_tilt_radians'] <= np.radians(ec.success_axis_tolerance_degrees))
                    & (col['palm_support_violation'] == 0))
        loaded = retained & (col['force_bearing_fingers'] >= 2)
        raw_angle = np.cumsum(col['twist_increment'])
        qualified_angle = np.cumsum(np.where((col['twist_increment'] < 0) | retained, col['twist_increment'], 0.))
        force_angle = np.cumsum(np.where((col['twist_increment'] < 0) | (col['force_retained'] > 0), col['twist_increment'], 0.))
        if (not np.allclose(col['time'], np.arange(1, len(samples) + 1) * ec.timestep, rtol=0, atol=1e-9)
                or not np.array_equal(col['retained'], retained)
                or not np.array_equal(col['force_retained'], loaded)
                or not np.allclose(col['raw_angle'], raw_angle, rtol=0, atol=1e-9)
                or not np.allclose(col['qualified_angle'], qualified_angle, rtol=0, atol=1e-9)
                or not np.allclose(col['force_qualified_angle'], force_angle, rtol=0, atol=1e-9)
                or min(raw_angle[-1], qualified_angle[-1], force_angle[-1]) < 2*np.pi
                or col['force_bearing_fingers'][-1] < 2
                or col['axis_tilt_radians'][-1] > np.radians(ec.success_axis_tolerance_degrees)
                or col['palm_support_violation'].any()
                or col['table_contacts'].any() or np.any(col['object_displacement'] > .18)
                or np.any(col['contactless_seconds'] >= ec.contact_loss_seconds - 1e-10)
                or any(np.max(col[name]) > 1e-6 or np.min(col[name]) < 0 for name in support)):
            raise ValueError('Recorded physics contradicts force-qualified unsupported rotation')
        if (not np.isclose(audit.get('final_force_qualified_angle', np.nan), force_angle[-1], rtol=0, atol=1e-9)
                or not np.isclose(record.get('force_qualified_angle', np.nan), force_angle[-1], rtol=0, atol=1e-9)
                or record.get('force_bearing_fingers') != int(col['force_bearing_fingers'][-1])):
            raise ValueError('Recorded force summaries disagree with physics evidence')
    if frame_cursor != len(frames):
        raise ValueError('Extra joint frames are not backed by the five recorded episodes')
    return metadata


def validate_primitive_source(primitive):
    """Recheck source evidence and PCA before the main MPC collection/training."""
    # Local import keeps the generic numerical PrimitiveBasis independently usable.
    from .cli import validate_rotation_trajectories
    from .primitives import PrimitiveBasis

    provenance = primitive.metadata.get('provenance', {})
    path = Path(provenance.get('trajectory_file', ''))
    if not path.is_file() or sha256(path) != provenance.get('sha256'):
        raise ValueError('Main MPC requires the verified frozen-SAC primitive source file')
    if any((parent / 'invalidated.json').exists() for parent in list(path.resolve().parents)[:3]):
        raise ValueError('The primitive source was invalidated')
    with np.load(path, allow_pickle=False) as archive:
        frames = archive['joint_positions']
        metadata = json.loads(str(archive['metadata']))
    if metadata != provenance.get('record'):
        raise ValueError('Primitive provenance differs from its recorded source')
    validate_rotation_trajectories(frames, metadata)
    validate_frozen_policy_collection(frames, metadata, path.parent)
    expected = PrimitiveBasis.from_trajectories(frames, provenance=provenance)
    if primitive.metadata != expected.metadata or any(
            not np.allclose(getattr(primitive, name), getattr(expected, name), rtol=1e-10, atol=1e-12)
            for name in ('mean', 'basis', 'coefficient_min', 'coefficient_max', 'explained_variance_ratio')):
        raise ValueError('Primitive differs from the 90-percent PCA of its verified source')
    return metadata
