"""Bounded physical 40 mm prefix-and-hold branches; no state teleportation."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
from mpc_dex.cli import read_config
from mpc_dex.contracts import environment_contract, sha256
from mpc_dex.env import AllegroRotationEnv, ROTATION_VALIDATION
from scripts.audit_recorded_controls import audit


OUTPUT = Path(__file__).resolve().parent
SOURCE = OUTPUT.parent / 'transitions.npz'
CONFIG = ROOT / 'configs/diagnostic_rotation_40mm_continuous.json'
_, continuous = read_config(CONFIG)
strict = replace(continuous, terminate_on_success=True)
strict_config = OUTPUT / 'strict_config.json'
strict_config.write_text(json.dumps({'environment': asdict(strict)}, indent=2)+'\n')
with np.load(SOURCE, allow_pickle=False) as archive:
    original = {key: archive[key].copy() for key in archive.files}
summary = {'purpose': 'diagnostic physical 40 mm bootstrap behavior, not SAC or production primitive',
           'source': str(SOURCE), 'source_sha256': sha256(SOURCE),
           'config_sha256': sha256(CONFIG), 'rotation_validation': ROTATION_VALIDATION,
           'strict_environment_contract': environment_contract(strict),
           'source_hashes': {name: sha256(ROOT/name) for name in
                            ['mpc_dex/env.py', 'scripts/audit_policy.py',
                             'scripts/audit_recorded_controls.py', 'uv.lock']},
           'experiment_source_sha256': sha256(__file__),
           'branches': []}
for prefix in [18, 19, 20]:
    for fraction in [0., .25, .5, .75]:
        name = f'prefix{prefix}_blend{round(fraction*100):02d}'
        directory = OUTPUT / name
        directory.mkdir()
        env = AllegroRotationEnv(strict)
        obs, info = env.reset(seed=42)
        initial = {key: getattr(env.data, key).copy() for key in ['qpos', 'qvel', 'ctrl']}
        initial['observation'] = obs.copy()
        trace = {key: [] for key in ['obs', 'actions', 'rewards', 'next_obs',
                                    'done', 'terminated', 'truncated', 'qpos', 'qvel']}
        def execute(action):
            global obs, info
            before = obs.copy()
            obs, reward, terminated, truncated, info = env.step(action)
            values = dict(obs=before, actions=np.asarray(action).copy(), rewards=reward,
                          next_obs=obs.copy(), done=terminated or truncated,
                          terminated=terminated, truncated=truncated,
                          qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy())
            for key, value in values.items():
                trace[key].append(value)
            return terminated or truncated
        try:
            for index, action in enumerate(original['actions'][:prefix]):
                if execute(action):
                    raise RuntimeError('Original prefix unexpectedly terminated in strict replay')
                np.testing.assert_array_equal(env.data.qpos, original['qpos'][index])
                np.testing.assert_array_equal(env.data.qvel, original['qvel'][index])
            current = env.data.ctrl.copy()
            next_target = env.to_joint_targets(original['actions'][prefix])
            action = env.to_action(current + fraction*(next_target-current))
            actual_target = env.to_joint_targets(action)
            for _ in range(3):
                if execute(action):
                    break
            record = dict(name=name, prefix_commands=prefix, blend_fraction=fraction,
                          added_commands=len(trace['actions'])-prefix,
                          success=bool(info['success']), drop=bool(info['drop']),
                          palm_support_violation=bool(info['palm_support_violation']),
                          peak_palm_support_force=float(info['peak_palm_support_force']),
                          failure_reason=info['failure_reason'], angle=float(info['angle']),
                          retained_angle=float(info['retained_angle']),
                          finger_contacts=int(info['finger_contacts']),
                          axis_tilt_radians=float(info['axis_tilt_radians']),
                          held_joint_target=actual_target.tolist(),
                          production_eligible=False)
            np.savez_compressed(directory/'transitions.npz',
                **{key: np.asarray(value) for key,value in trace.items()},
                **{'initial_'+key: value for key,value in initial.items()},
                joint_positions=np.vstack((initial['qpos'][env.q_ids],
                                           np.asarray(trace['qpos'])[:, env.q_ids])))
            (directory/'record.json').write_text(json.dumps(record, indent=2)+'\n')
        finally:
            env.close()
        if record['success']:
            independent = audit(directory/'transitions.npz', strict_config,
                                directory/'independent_audit')
            aftermath = audit(directory/'transitions.npz', CONFIG,
                              directory/'posthold_2s_audit', hold_steps=20)
            record['independent_audit'] = {key: independent[key] for key in [
                'finger_supported_rotation_success', 'independent_rotation_success',
                'final_force_qualified_angle', 'independent_drop',
                'independent_palm_support_violation', 'peak_support_normal_forces']}
            record['posthold_2s'] = {key: aftermath[key] for key in [
                'finger_supported_rotation_success', 'independent_rotation_success',
                'final_force_qualified_angle', 'independent_drop',
                'independent_palm_support_violation', 'peak_support_normal_forces',
                'steps', 'simulated_seconds', 'final_finger_contacts',
                'final_force_bearing_fingers', 'final_axis_tilt_radians']}
        summary['branches'].append(record)
        (OUTPUT/'report.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(json.dumps(record), flush=True)
summary['source_hash_mismatches'] = [name for name,value in summary['source_hashes'].items()
                                     if sha256(ROOT/name) != value]
summary['completed_branches'] = len(summary['branches'])
summary['successes'] = sum(row['success'] for row in summary['branches'])
summary['independently_force_verified_successes'] = sum(
    row.get('independent_audit', {}).get('finger_supported_rotation_success', False)
    for row in summary['branches'])
(OUTPUT/'report.json').write_text(json.dumps(summary, indent=2)+'\n')
