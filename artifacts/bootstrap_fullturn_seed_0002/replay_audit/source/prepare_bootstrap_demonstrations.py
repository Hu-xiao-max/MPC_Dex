"""Re-execute and independently verify absolute-control bootstrap demonstrations.

One input contributes at most one episode, ending at its first genuine retained
full turn. Remaining controls are reported as unused. This is MPC diagnostic
data, not evidence of a SAC policy or the main method's 20-episode dataset.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np

from mpc_dex.bootstrap_env import policy_interface
from mpc_dex.contracts import environment_contract, sha256
from mpc_dex.env import AllegroRotationEnv, ROTATION_VALIDATION
from mpc_dex.replay import ReplayBuffer
from scripts.audit_policy import PHYSICS_COLUMNS, audit_episode
from scripts.audit_recorded_controls import RecordedPolicy
from scripts.train_bootstrap import read_environment


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def _execute(source, ec):
    with np.load(source, allow_pickle=False) as archive:
        recorded = {key: archive[key].copy() for key in archive.files}
    source_controls = recorded['actions']
    if (source_controls.ndim != 2 or source_controls.shape[1] != 16 or not len(source_controls)
            or source_controls.dtype.kind not in 'fiu'
            or not np.isfinite(source_controls).all() or np.any(np.abs(source_controls) > 1)):
        raise ValueError('Expected finite nonempty absolute normalized controls [N,16]')
    # ReplayBuffer and the policy use float32 actions. Canonicalize before any
    # physics or audit, because action+1 is evaluated in the action's dtype by
    # the environment and float64 source controls need not execute identically.
    controls = np.asarray(source_controls, dtype=np.float32)
    action_conversion = dict(source_dtype=str(source_controls.dtype), effective_dtype='float32',
        max_absolute_difference=float(np.max(np.abs(
            source_controls.astype(np.float64)-controls.astype(np.float64)))),
        applied_before_physics=True)
    aliases = {'obs': 'obs', 'observations': 'obs', 'next_obs': 'next_obs',
               'next_observations': 'next_obs', 'qpos': 'qpos', 'qvel': 'qvel'}
    env = AllegroRotationEnv(ec)
    rows = {key: [] for key in ('obs', 'actions', 'rewards', 'next_obs',
                                'terminated', 'truncated', 'qpos', 'qvel')}
    errors = {key: 0. for key in aliases if key in recorded}
    try:
        obs, _ = env.reset(seed=0)
        initial = dict(qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(),
                       ctrl=env.data.ctrl.copy(), observation=obs.copy())
        expected_shapes = {'obs': (len(controls), len(obs)), 'next_obs': (len(controls), len(obs)),
                           'qpos': (len(controls), env.model.nq), 'qvel': (len(controls), env.model.nv)}
        for key in errors:
            if recorded[key].shape != expected_shapes[aliases[key]] or not np.isfinite(recorded[key]).all():
                raise ValueError(f'Invalid recorded {key} array')
        for index, action in enumerate(controls):
            before = obs.copy()
            obs, reward, terminated, truncated, info = env.step(action)
            executed = dict(obs=before, actions=np.asarray(action).copy(), rewards=reward,
                            next_obs=obs.copy(), terminated=terminated, truncated=truncated,
                            qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy())
            for key in errors:
                difference = float(np.max(np.abs(executed[aliases[key]]-recorded[key][index])))
                errors[key] = max(errors[key], difference)
                tolerance = 1e-5 if aliases[key] in {'obs', 'next_obs'} else 1e-8
                if difference > tolerance:
                    raise ValueError(f'Source {key} does not replay at command {index}: {difference}')
            for key in rows:
                rows[key].append(executed[key])
            if terminated or truncated:
                break
        if (not terminated or truncated or not info['success'] or info['drop']
                or info['palm_support_violation'] or info['failure_reason'] is not None
                or min(info['angle'], info['retained_angle']) < 2*np.pi
                or info['finger_contacts'] < 2
                or not 0 <= info['axis_tilt_radians'] <= np.radians(ec.success_axis_tolerance_degrees)
                or not 0 <= info['peak_palm_support_force'] <= ec.palm_support_force_tolerance):
            raise ValueError('Source did not execute a genuine full-turn no-palm success')
        rows = {key: np.asarray(value) for key, value in rows.items()}
        used = len(rows['actions'])
        report, physics, commands = audit_episode(env, RecordedPolicy(controls[:used]), 0, used)
        if not report['finger_supported_rotation_success']:
            raise ValueError('Independent force audit did not verify the full-turn demonstration')
        final_info = {key: info[key] for key in (
            'success', 'drop', 'angle', 'retained_angle', 'finger_contacts', 'axis_tilt_radians',
            'palm_support_violation', 'peak_palm_support_force', 'failure_reason')}
        return rows, initial, dict(used_commands=used, source_commands=len(controls),
            unused_tail_commands=len(controls)-used, max_replay_errors=errors,
            action_conversion=action_conversion,
            comparison_tolerances={'observations': 1e-5, 'qpos_qvel': 1e-8},
            final_info=final_info, audit=report), physics, commands
    finally:
        env.close()


def prepare(sources, config, output):
    sources = [Path(source).resolve() for source in sources]
    output, config = Path(output), Path(config)
    if not sources:
        raise ValueError('At least one source is required')
    if output.exists() or output.with_suffix(output.suffix+'.metadata.json').exists():
        raise FileExistsError(output)
    cfg, ec = read_environment(config)
    interface = policy_interface(cfg.get('bootstrap', {}))
    if interface != {'kind': 'absolute_joint_target'}:
        raise ValueError('Verified absolute demonstrations require an absolute config without bootstrap.control')
    digests = [sha256(source) for source in sources]
    if len(set(digests)) != len(digests):
        raise ValueError('Duplicate source archives cannot supply distinct demonstration episodes')
    audit_root = output.parent/(output.stem+'_audit')
    audit_root.mkdir(parents=True, exist_ok=False)
    numerical_sources = ['scripts/prepare_bootstrap_demonstrations.py', 'scripts/train_bootstrap.py',
        'scripts/audit_policy.py', 'scripts/audit_recorded_controls.py', 'mpc_dex/env.py',
        'mpc_dex/bootstrap_env.py', 'mpc_dex/contracts.py', 'mpc_dex/replay.py', 'uv.lock']
    source_hashes = {name: sha256(ROOT/name) for name in numerical_sources}
    snapshot = audit_root/'source'; snapshot.mkdir()
    for name in numerical_sources:
        shutil.copyfile(ROOT/name, snapshot/Path(name).name)
    shutil.copyfile(config, snapshot/'config.json')
    episodes, all_rows = [], []
    for index, (source, digest) in enumerate(zip(sources, digests)):
        rows, initial, record, physics, commands = _execute(source, ec)
        directory = audit_root/f'episode_{index:03d}'; directory.mkdir()
        shutil.copyfile(source, directory/'input.npz')
        if sha256(source) != digest or sha256(directory/'input.npz') != digest:
            raise ValueError('Source archive changed during replay')
        np.savez_compressed(directory/'actual_transitions.npz', **rows,
                            done=rows['terminated'] | rows['truncated'],
                            **{'initial_'+key: value for key, value in initial.items()})
        np.savez_compressed(directory/'physics.npz', samples=physics, columns=np.asarray(PHYSICS_COLUMNS))
        _json(directory/'commands.json', commands)
        record.update(source=str(source), source_sha256=digest,
                      audit_directory=str(directory.resolve()), episode=index)
        _json(directory/'report.json', record)
        episodes.append(record); all_rows.append(rows)
    if any(sha256(ROOT/name) != digest for name, digest in source_hashes.items()):
        raise ValueError('Numerical sources changed during demonstration preparation')
    replay = ReplayBuffer(capacity=sum(len(rows['actions']) for rows in all_rows))
    for rows in all_rows:
        replay.add_episode(rows)  # Derives next actions only from actually executed successors.
    replay.save(output)
    metadata = dict(kind='verified-bootstrap-demonstrations', demonstrator='MPC diagnostic (not SAC)',
        demonstration_successes=len(episodes), successful_full_turns=len(episodes),
        rotation_validation=ROTATION_VALIDATION, episodes=episodes, transitions=len(replay),
        replay_sha256=sha256(output), environment_contract=environment_contract(ec),
        policy_interface=interface, environment=asdict(ec), config=str(config.resolve()),
        config_sha256=sha256(config), numerical_source_sha256=source_hashes,
        versions={'mujoco': mujoco.__version__, 'numpy': np.__version__},
        primitive_eligible=False, main_mpc_dataset_eligible=False, production_eligible=False,
        reward='actual absolute-target environment reward; recomputed in contact simulation',
        boundary='first actual full-turn success per source; terminal next action invalid',
        config_provenance=cfg.get('demonstration_preparation_provenance'))
    _json(output.with_suffix(output.suffix+'.metadata.json'), metadata)
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', nargs='+', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.config, args.output), indent=2))
