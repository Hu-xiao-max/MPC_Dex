"""One post-hoc maximum-boundary prefix and unchanged-target hold branch."""
from pathlib import Path
from dataclasses import replace
import json
import sys

sys.dont_write_bytecode = True
D = Path(__file__).resolve().parent
sys.path.insert(0, str(D / 'source'))
import numpy as np
from mpc_dex.env import EnvConfig
from mpc_dex.contracts import sha256, environment_contract
from scripts.audit_policy import PHYSICS_COLUMNS, write_json
from scripts.audit_recorded_controls import RecordedPolicy
from audit_helpers import run_audit


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: value.copy() for key, value in data.items()}


def main():
    assert not (D / 'report.json').exists()
    selection = json.loads((D / 'selection.json').read_text())
    source = load(D / 'source_transitions.npz')
    physics = load(D / 'source_physics.npz')
    np.testing.assert_array_equal(physics['columns'], PHYSICS_COLUMNS)
    count = selection['boundary']
    assert count == 26
    actions = np.concatenate((source['actions'][:count], np.repeat(source['actions'][count-1:count], 20, axis=0)))
    assert actions.shape == (46, 16) and actions.dtype == np.float32
    np.savez_compressed(D / 'prescribed_controls.npz', actions=actions)
    ec = EnvConfig(**json.loads((D / 'config.json').read_text())['environment'])
    assert ec.terminate_on_success
    assert environment_contract(ec) == json.loads((D / 'source_report.json').read_text())['environment_contract']
    inputs = {str(path.relative_to(D)): sha256(path) for path in D.rglob('*') if path.is_file() and '__pycache__' not in path.parts}
    write_json(D / 'inputs_manifest.json', inputs)
    prefix_verified = False

    def verify_prefix(env, policy, result):
        nonlocal prefix_verified
        row = len(policy.actions) - 1
        if row >= count:
            return
        obs, reward, terminated, truncated, _ = result
        actual = dict(obs=policy.obs[-1], actions=policy.actions[-1], next_obs=obs,
            rewards=reward, terminated=terminated, truncated=truncated,
            qpos=env.data.qpos, qvel=env.data.qvel, targets=env.data.ctrl)
        for key, value in actual.items():
            np.testing.assert_array_equal(value, source[key][row], err_msg=f'prefix row {row + 1}: {key}')
        observer = env.finger_contacts.__self__
        np.testing.assert_array_equal(np.asarray(observer.rows), physics['samples'][:(row+1)*env.substeps])
        if row + 1 == count:
            prefix_verified = True
            write_json(D / 'prefix_verification.json', dict(controls=count, physics_rows=count*env.substeps,
                transition_fields=list(actual), all_bitexact=True, force_qualified_angle=observer.force_qualified_angle,
                time_seconds=float(env.data.time), verified_before_first_additional_hold=True))
            print('PREFIX_VERIFIED ' + json.dumps(dict(controls=count, physics_rows=count*env.substeps,
                  force_qualified_angle=observer.force_qualified_angle)), flush=True)

    result, executed, _ = run_audit(RecordedPolicy(actions), ec, D / 'strict_audit', len(actions), verify=verify_prefix)
    assert prefix_verified
    np.testing.assert_array_equal(executed, actions[:len(executed)])
    trace = load(D / 'strict_audit/transitions.npz')
    for key in ('initial_qpos', 'initial_qvel', 'initial_ctrl', 'initial_joint_positions'):
        np.testing.assert_array_equal(trace[key], source[key])
    for key in ('obs', 'actions', 'rewards', 'next_obs', 'terminated', 'truncated', 'qpos', 'qvel', 'targets'):
        np.testing.assert_array_equal(trace[key][:count], source[key][:count])
    np.savez_compressed(D / 'executed_control_origins.npz',
        origins=np.asarray(['recorded_existing_prefix'] * count + ['fixed_selected_boundary_target'] * (len(executed)-count)),
        source_rows_one_based=np.r_[np.arange(1, count+1), np.repeat(count, len(executed)-count)],
        actual_new_mpc_correction_mask=np.zeros(len(executed), dtype=bool))
    result['policy_kind'] = 'post_hoc_maximum_boundary_prefix_then_exact_fixed_target'
    write_json(D / 'strict_audit/report.json', result)
    independent = hold = None
    if result['finger_supported_rotation_success']:
        independent, replayed, _ = run_audit(RecordedPolicy(executed), ec, D / 'independent_success_replay', len(executed))
        np.testing.assert_array_equal(replayed, executed)
        repeated = load(D / 'independent_success_replay/transitions.npz')
        for key in trace:
            np.testing.assert_array_equal(repeated[key], trace[key])
        np.testing.assert_array_equal(load(D / 'independent_success_replay/physics.npz')['samples'],
                                      load(D / 'strict_audit/physics.npz')['samples'])
        independent['all_transitions_and_physics_bitexact'] = True
        assert independent['finger_supported_rotation_success']
        write_json(D / 'independent_success_replay/report.json', independent)
        held = np.concatenate((executed, np.repeat(executed[-1:], 20, axis=0)))
        hold, _, _ = run_audit(RecordedPolicy(held), replace(ec, terminate_on_success=False), D / 'post_success_hold_2s', len(held))
        hp = load(D / 'post_success_hold_2s/physics.npz')['samples']
        original = load(D / 'strict_audit/physics.npz')['samples']
        np.testing.assert_array_equal(hp[:len(original)], original)
        after = hp[len(original):]
        col = PHYSICS_COLUMNS.index
        hold.update(requested_hold_seconds=2., completed_hold_seconds=len(after)*ec.timestep,
            completed_two_seconds_without_failure=bool(len(after)==1000 and not hold['independent_drop'] and not hold['independent_palm_support_violation']),
            minimum_hold_loaded_fingers=int(after[:,col('force_bearing_fingers')].min()) if len(after) else None,
            maximum_hold_axis_tilt_radians=float(after[:,col('axis_tilt_radians')].max()) if len(after) else None,
            minimum_hold_force_qualified_angle=float(after[:,col('force_qualified_angle')].min()) if len(after) else None,
            force_retained_every_hold_physics_step=bool(len(after)==1000 and np.all(after[:,col('force_retained')]>0)))
        write_json(D / 'post_success_hold_2s/report.json', hold)
    assert all(sha256(D / name)==digest for name, digest in inputs.items())
    report = dict(purpose='One explicitly post-hoc maximum recorded control-boundary hold branch',
        production_eligible=False, policy_only_success=False, training_updates=0, planning_calls=0,
        selection=selection, prefix_bitexact_verified_before_hold=True,
        strict_audit=result, independent_success_replay=independent, post_success_hold_2s=hold,
        executed_hold_controls=len(executed)-count, environment_contract=environment_contract(ec), input_hashes=inputs)
    write_json(D / 'report.json', report)
    write_json(D / 'manifest.json', {str(p.relative_to(D)):sha256(p) for p in D.rglob('*')
        if p.is_file() and '__pycache__' not in p.parts and p.name not in ('manifest.json','run.log')})
    print(json.dumps({k:result[k] for k in ('steps','finger_supported_rotation_success','environment_failure_reason',
                                          'max_force_qualified_angle','final_force_qualified_angle','independent_drop')},indent=2), flush=True)


if __name__ == '__main__':
    main()
