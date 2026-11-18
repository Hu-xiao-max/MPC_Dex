"""Bounded artifact-only MPC recovery from the recorded prefix-6 handoff state.

Execution requires --execute plus the frozen execution_manifest.json. The six
recorded controls are physically replayed and verified bit-for-bit, then the
archived corrected native MPC recipe supplies at most twenty corrections. No
policy inference, learning, hold branch or parameter sweep occurs.
"""
from __future__ import annotations

from pathlib import Path
from collections import Counter
from dataclasses import asdict
import argparse
import hashlib
import json
import os
import platform
import sys
import time

D = Path(__file__).resolve().parent
for required in ('mpc_dex/env.py', 'mpc_dex/mpc.py', 'mpc_dex/rollouts.py', 'mpc_dex/native_rollouts.py',
                 'scripts/audit_policy.py', 'scripts/probe_mpc.py'):
    if not (D / 'source' / required).is_file():
        raise RuntimeError('Missing archived numerical source; prepare first: ' + required)
sys.path.insert(0, str(D / 'source'))
sys.path.insert(0, str(D))

import mujoco
import numpy as np

from mpc_dex.env import AllegroRotationEnv, EnvConfig
from mpc_dex.contracts import environment_contract, sha256
from mpc_dex.mpc import CEMConfig, NoFeasiblePlan
from mpc_dex.primitives import PrimitiveBasis
from mpc_dex.rollouts import RolloutCostConfig
from mpc_dex.native_rollouts import NativeMujocoRolloutEvaluator
from scripts.probe_mpc import action_search_bounds
from scripts.audit_policy import PhysicsObserver, PHYSICS_COLUMNS, audit_episode, write_json
from scripts.audit_recorded_controls import RecordedPolicy
from recovery_recipe import ProjectedReferenceCEM, integration_state

for name in ('mpc_dex.env', 'mpc_dex.mpc', 'mpc_dex.rollouts', 'mpc_dex.native_rollouts',
             'mpc_dex.primitives', 'scripts.audit_policy', 'scripts.probe_mpc'):
    if not Path(sys.modules[name].__file__).resolve().is_relative_to((D / 'source').resolve()):
        raise RuntimeError('Runtime imported source outside the frozen artifact: ' + name)

FIELDS = ('obs', 'actions', 'rewards', 'next_obs', 'terminated', 'truncated',
          'qpos', 'qvel', 'targets', 'integration_state')
PREFIX_CHECK_FIELDS = FIELDS
INITIAL_FIELDS = ('obs', 'qpos', 'qvel', 'ctrl', 'integration_state')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def verify_manifest():
    manifest = json.loads((D / 'execution_manifest.json').read_text())
    for name, digest in manifest.items():
        if not (D / name).is_file() or sha(D / name) != digest:
            raise RuntimeError('Frozen execution source/input hash changed: ' + name)
    return manifest


def select_prefix(source, controls):
    """Keep the first `controls` recorded transitions; initial_* fields stay whole."""
    total = len(source['actions'])
    if not 1 <= controls < total:
        raise ValueError('Prefix must be shorter than the recorded trajectory')
    if source['terminated'][:controls].any() or source['truncated'][:controls].any():
        raise ValueError('Recorded prefix terminates before handoff')
    return {key: (value if key.startswith('initial_') else value[:controls]) for key, value in source.items()}


def initial_state(env):
    return dict(obs=None, qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(), ctrl=env.data.ctrl.copy(),
                joint_positions=env.data.qpos[env.q_ids].copy(), integration_state=integration_state(env),
                time=float(env.data.time))


def strict_force_success(report):
    return bool(report['environment_success'] and report['finger_supported_rotation_success']
                and report['terminated'] and not report['truncated'] and not report['environment_drop']
                and not report['independent_drop'] and not report['independent_palm_support_violation']
                and report['no_external_support'])


def correction_pairs(trace, prefix_controls):
    """Rows after the physical prefix; prefix rows are evidence, never labels."""
    steps = len(trace['actions'])
    if steps <= prefix_controls:
        return None
    pairs = {key: trace[key][prefix_controls:] for key in FIELDS}
    pairs['source_control_1based'] = np.arange(prefix_controls + 1, steps + 1)
    pairs['control_times'] = trace['control_times'][prefix_controls + 1:]
    return pairs


def save_transitions(path, initial, rows, q_ids):
    arrays = {key: np.asarray(rows[key]) for key in FIELDS}
    joints = np.vstack((initial['joint_positions'], arrays['qpos'][:, q_ids]))
    np.savez_compressed(path, **arrays,
                        **{'initial_' + key: value for key, value in initial.items() if key != 'obs'},
                        initial_obs=initial['obs'],
                        joint_positions=joints, control_times=np.r_[0., rows['control_times']])


def independent_replay(ec, seed, actions, output, original, original_physics, prefix_controls):
    """Fresh cold-reset physical replay of every executed control with a separate force audit."""
    output.mkdir()
    env = AllegroRotationEnv(ec)
    initial, rows = {}, {key: [] for key in (*FIELDS, 'control_times')}
    reset, step = env.reset, env.step
    recorded = RecordedPolicy(actions)

    def recording_reset(*args, **kwargs):
        result = reset(*args, **kwargs)
        initial.update(initial_state(env))
        initial['obs'] = result[0].copy()
        return result

    class RecordingPolicy:
        def act(self, obs, deterministic=True):
            action = recorded.act(obs, deterministic=deterministic)
            rows['obs'].append(obs.copy())
            rows['actions'].append(action.copy())
            return action

    def recording_step(action):
        result = step(action)
        ob, reward, terminated, truncated, _ = result
        for key, value in dict(next_obs=ob.copy(), rewards=reward, terminated=terminated, truncated=truncated,
                               qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(), targets=env.data.ctrl.copy(),
                               integration_state=integration_state(env)).items():
            rows[key].append(value)
        rows['control_times'].append(float(env.data.time) - initial['time'])
        return result

    env.reset, env.step = recording_reset, recording_step
    try:
        report, physics, commands = audit_episode(env, RecordingPolicy(), seed, len(actions), True)
        save_transitions(output / 'transitions.npz', initial, rows, env.q_ids)
        np.savez_compressed(output / 'physics.npz', samples=physics, columns=np.asarray(PHYSICS_COLUMNS))
        write_json(output / 'commands.json', commands)
        replay = read_npz(output / 'transitions.npz')
        comparisons = {key: bool(np.array_equal(replay[key], original[key])) for key in original}
        comparisons['physics'] = bool(np.array_equal(physics, original_physics))
        if not all(comparisons.values()):
            raise RuntimeError(f'Independent execution mismatch: {comparisons}')
        report.update(policy_kind='recorded_prefix6_then_corrected_native_mpc', production_eligible=False,
                      policy_only_success=False, prefix_controls=prefix_controls,
                      mpc_controls=len(actions) - prefix_controls, replay_bitexact=comparisons,
                      model_equality_constraints=int(env.model.neq))
        report['strict_force_success'] = strict_force_success(report)
        write_json(output / 'report.json', report)
        return report
    finally:
        env.close()


def run_branch(ec, protocol):
    output = D / 'recovery'
    output.mkdir()
    prefix_controls = protocol['prefix_controls']
    source_path = D / 'inputs/prefix_stage/prefix_transitions.npz'
    full = read_npz(source_path)
    assert len(full['actions']) == 9 and full['terminated'][-1] and full['actions'].dtype == np.float32
    prefix = select_prefix(full, prefix_controls)
    prefix_physics = read_npz(D / 'inputs/prefix_stage/prefix_physics.npz')
    assert np.array_equal(prefix_physics['columns'], PHYSICS_COLUMNS)
    env = AllegroRotationEnv(ec)
    obs, info = env.reset(seed=protocol['reset_seed'])
    initial = initial_state(env)
    initial['obs'] = obs.copy()
    for key in INITIAL_FIELDS:
        assert np.array_equal(initial[key], full['initial_' + key]), key
    low, high, search = action_search_bounds(env, protocol['action_bound'])
    prior_path = D / 'inputs/previous_recovery/prior_transitions.npz'
    prior_data = read_npz(prior_path)
    frames = np.vstack((initial['joint_positions'], prior_data['qpos'][:, env.q_ids]))
    prior = PrimitiveBasis.from_trajectories(frames, provenance=dict(
        source=str(prior_path), sha256=sha256(prior_path),
        purpose='Identical diagnostic measured 40 mm PCA prior; not a SAC primitive', successful_policy=False))
    expected = json.loads((D / 'inputs/previous_recovery/reference_planner_report.json').read_text())
    for key in ('mean', 'basis', 'coefficient_min', 'coefficient_max', 'explained_variance_ratio'):
        assert np.array_equal(getattr(prior, key), expected['diagnostic_basis'][key]), key
    prior.save(output / 'diagnostic_prior.npz')
    observer = PhysicsObserver(env)
    rows = {key: [] for key in (*FIELDS, 'control_times')}
    commands, evaluator, planner = [], None, None
    began = time.monotonic()
    report = dict(seed=protocol['reset_seed'], planner_seed=protocol['planner_seed'], status='prefix_replay',
                  source_prefix_sha256=sha256(source_path), source_prefix_controls_available=9,
                  environment_contract=environment_contract(ec), prefix_controls=prefix_controls,
                  max_mpc_controls=protocol['max_mpc_controls'], action_search=search,
                  production_eligible=False, main_demonstrations=0, policy_only_success=False,
                  policy_inference=False, learning_updates=0, commands=commands)

    def execute(action, origin, extra=None):
        nonlocal obs, info
        before, previous = obs.copy(), env.data.ctrl.copy()
        obs, reward, terminated, truncated, info = env.step(action)
        for key, value in dict(obs=before, actions=action.copy(), rewards=reward, next_obs=obs.copy(),
                               terminated=terminated, truncated=truncated, qpos=env.data.qpos.copy(),
                               qvel=env.data.qvel.copy(), targets=env.data.ctrl.copy(),
                               integration_state=integration_state(env)).items():
            rows[key].append(value)
        rows['control_times'].append(float(env.data.time) - initial['time'])
        commands.append(dict(control=len(commands) + 1, origin=origin, reward=float(reward),
                             raw_angle=observer.raw_angle, retained_angle=observer.qualified_angle,
                             force_qualified_angle=observer.force_qualified_angle,
                             finger_contacts=int(info['finger_contacts']),
                             axis_tilt_radians=float(info['axis_tilt_radians']), drop=bool(info['drop']),
                             environment_success=bool(info['success']),
                             palm_support_violation=bool(info['palm_support_violation']),
                             failure_reason=info['failure_reason'],
                             target_delta_max=float(abs(env.data.ctrl - previous).max()),
                             action_abs_max=float(abs(action).max()), **(extra or {})))
        assert len(observer.rows) == len(commands) * env.substeps
        return terminated or truncated

    try:
        for index in range(prefix_controls):
            assert np.array_equal(obs, prefix['obs'][index])
            assert not execute(prefix['actions'][index], 'recorded_source_prefix')
            for key in PREFIX_CHECK_FIELDS:
                assert np.array_equal(rows[key][-1], prefix[key][index]), (index, key)
        assert np.array_equal(np.asarray(observer.rows), prefix_physics['samples'][:prefix_controls * env.substeps])
        branch = integration_state(env)
        np.savez_compressed(output / 'actual_branch_state.npz', qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(),
                            ctrl=env.data.ctrl.copy(), observation=obs.copy(), integration_state=branch,
                            elapsed_seconds=float(env.data.time) - initial['time'], search_low=low, search_high=high,
                            raw_angle=observer.raw_angle, retained_angle=observer.qualified_angle,
                            force_qualified_angle=observer.force_qualified_angle,
                            recorded_next_control=full['actions'][prefix_controls].copy())
        report.update(prefix_all_fields_and_physics_bitexact=True, handoff_seconds=float(env.data.time) - initial['time'],
                      branch_force_qualified_angle=observer.force_qualified_angle,
                      branch_finger_contacts=int(info['finger_contacts']), branch_axis_tilt_radians=float(info['axis_tilt_radians']),
                      true_live_execution_start=env.data.ctrl.tolist(),
                      live_start_out_of_search_dimensions=np.flatnonzero((env.data.ctrl < low) | (env.data.ctrl > high)).tolist())
        planner = ProjectedReferenceCEM(prior, low, high, CEMConfig(**protocol['cem']), protocol['planner_seed'])
        planner.execution_start = env.data.ctrl.copy()
        planner.reset(env.data.ctrl.copy())
        assert np.array_equal(branch, integration_state(env))
        cost = RolloutCostConfig(**protocol['cost'])
        evaluator = NativeMujocoRolloutEvaluator(env, timestep_seconds=protocol['cem']['timestep_seconds'], config=cost,
                                                 require_native=True, cache_directory=D / 'native_cache')
        report.update(status='mpc_recovery', cem=asdict(planner.config), cost=asdict(cost),
                      native_build=evaluator.native_provenance, no_slew_limit=True)
        write_json(output / 'report.json', report)
        for index in range(1, protocol['max_mpc_controls'] + 1):
            before = integration_state(env)
            planner.execution_start = env.data.ctrl.copy()
            reference_begin = len(planner.reference_records)
            planning_started = time.monotonic()
            try:
                target, diagnostics = planner.plan(evaluator)
            except NoFeasiblePlan as error:
                assert np.array_equal(before, integration_state(env))
                report.update(status='no_feasible_plan', error=str(error),
                              last_rejection_counts=dict(Counter(evaluator.last_diagnostics.get('rejection_reasons', []))),
                              last_failed_evaluation=evaluator.last_diagnostics)
                break
            assert np.array_equal(before, integration_state(env))
            selected_cost = float(evaluator(planner.last_sequence[None])[0])
            assert np.isclose(selected_cost, diagnostics['best_cost'], rtol=1e-10, atol=1e-10)
            prediction = evaluator.last_diagnostics.copy()
            assert prediction['joint_target_execution'] == 'float32_normalized_action'
            assert np.array_equal(before, integration_state(env))
            action = env.to_action(target)
            executed = env.to_joint_targets(action)
            assert np.all(executed >= low) and np.all(executed <= high)
            assert float(abs(action).max()) <= search['effective_bound']
            extra = dict(recovery_control=index, planning_seconds=time.monotonic() - planning_started,
                         planner=diagnostics, prediction=prediction,
                         reference_candidates=planner.reference_records[reference_begin:],
                         true_live_execution_start=planner.execution_start.tolist(), selected_target=target.tolist(),
                         executed_target=executed.tolist(), target_mapping_max_radians=float(abs(target - executed).max()),
                         proposals_and_execution_within_bounds=True, planning_left_live_integration_bitexact=True)
            if prefix_controls + index - 1 < len(full['actions']):
                recorded = full['actions'][prefix_controls + index - 1]
                extra.update(recorded_control_same_index_max_abs_difference=float(abs(action - recorded).max()),
                             recorded_control_same_index_l2_difference=float(np.linalg.norm(action - recorded)))
            boundary = execute(action, 'actual_corrected_native_mpc', extra)
            write_json(output / 'report.json', report)
            print(json.dumps({key: value for key, value in commands[-1].items()
                              if key not in ('planner', 'prediction', 'reference_candidates', 'selected_target',
                                             'executed_target', 'true_live_execution_start')}), flush=True)
            if boundary:
                report['status'] = 'environment_boundary'
                break
        else:
            report['status'] = 'mpc_budget_exhausted'
    except BaseException as error:
        report.update(status='execution_fault', error=repr(error))
        raise
    finally:
        report.update(wall_seconds=time.monotonic() - began, executed_controls=len(commands),
                      executed_mpc_controls=max(0, len(commands) - prefix_controls),
                      reference_records=planner.reference_records if planner is not None else [],
                      final_info={key: info[key] for key in ('angle', 'retained_angle', 'success', 'drop',
                                  'palm_support_violation', 'failure_reason', 'finger_contacts', 'axis_tilt_radians')})
        write_json(output / 'report.json', report)
        save_transitions(output / 'transitions.npz', initial, rows, env.q_ids)
        np.savez_compressed(output / 'physics.npz', samples=np.asarray(observer.rows), columns=np.asarray(PHYSICS_COLUMNS))
        np.savez_compressed(output / 'control_origins.npz', origin=np.asarray([row['origin'] for row in commands]),
                            expert_correction_mask=np.arange(len(commands)) >= prefix_controls)
        observer.close()
        if evaluator is not None:
            evaluator.close()
        env.close()
    trace = read_npz(output / 'transitions.npz')
    physics = read_npz(output / 'physics.npz')['samples']
    audit = independent_replay(ec, protocol['reset_seed'], trace['actions'], output / 'independent_audit',
                               trace, physics, prefix_controls)
    report.update(independent_audit=audit, strict_force_success=audit['strict_force_success'])
    if info['success'] and not audit['strict_force_success']:
        report['qualification_failure'] = 'Environment termination did not satisfy independent force qualification; no continuation executed'
    write_json(output / 'report.json', report)
    return report, trace, physics, audit, full


def package(report, trace, physics, audit, full, protocol):
    prefix_controls = protocol['prefix_controls']
    steps = len(trace['actions'])
    pairs = correction_pairs(trace, prefix_controls)
    projected = [item for item in report['reference_records'] if item['projected']]
    columns = list(np.load(D / 'recovery/physics.npz')['columns'])
    col = {name: physics[:, index] for index, name in enumerate(columns)}
    valid = ((col['raw_angle'] >= 2 * np.pi) & (col['force_qualified_angle'] >= 2 * np.pi)
             & (col['force_retained'] > 0) & (col['palm_support_violation'] == 0))
    first = np.flatnonzero(valid)
    final = dict(status=report['status'], strict_force_success=audit['strict_force_success'],
                 prefix_controls=prefix_controls, mpc_corrections=steps - prefix_controls, total_controls=steps,
                 physics_samples=len(physics), handoff_seconds=report['handoff_seconds'],
                 handoff_force_qualified_angle=report['branch_force_qualified_angle'],
                 final_seconds=float(trace['control_times'][-1]), final_raw_angle=audit['final_raw_angle'],
                 final_retained_angle=audit['final_qualified_angle'],
                 final_force_qualified_angle=audit['final_force_qualified_angle'],
                 maximum_force_qualified_angle=audit['max_force_qualified_angle'],
                 first_force_fullturn_physics_seconds=float(col['time'][first[0]]) if len(first) else None,
                 final_loaded_fingers=audit['final_force_bearing_fingers'], final_axis_tilt_radians=audit['final_axis_tilt_radians'],
                 independent_drop=audit['independent_drop'], palm_support=audit['independent_palm_support_violation'],
                 no_external_support=audit['no_external_support'], environment_failure_reason=audit['environment_failure_reason'],
                 prefix_all_fields_and_physics_bitexact=report['prefix_all_fields_and_physics_bitexact'],
                 full_execution_independently_replayed_bitexact=all(audit['replay_bitexact'].values()),
                 projected_reference_batches=len(projected),
                 maximum_reference_projection_radians=max((item['projection_max_radians'] for item in projected), default=0.),
                 total_planning_seconds=sum(item.get('planning_seconds', 0.) for item in report['commands']),
                 wall_seconds=report['wall_seconds'],
                 recorded_continuation_comparison=[{k: v for k, v in item.items() if k.startswith('recorded_control_same_index')}
                                                   | dict(control=item['control']) for item in report['commands']
                                                   if 'recorded_control_same_index_max_abs_difference' in item],
                 known_nominal_continuation=protocol['known_nominal_continuation'],
                 no_post_success_continuation_or_hold=True, no_alternate_prefix_or_parameter_sweep=True,
                 policy_inference=False, production_eligible=False, main_demonstrations=0, SAC_successes=0,
                 learning_updates=0, trajectory_sha256=sha(D / 'recovery/transitions.npz'),
                 physics_sha256=sha(D / 'recovery/physics.npz'),
                 independent_audit_sha256=sha(D / 'recovery/independent_audit/report.json'))
    if pairs is not None:
        assert np.array_equal(pairs['actions'], np.clip(pairs['actions'], -.98, .98))
        assert np.array_equal(read_npz(D / 'recovery/control_origins.npz')['expert_correction_mask'],
                              np.arange(steps) >= prefix_controls)
        np.savez_compressed(D / 'actual_mpc_correction_pairs.npz', **pairs)
        serialized = read_npz(D / 'actual_mpc_correction_pairs.npz')
        assert all(np.array_equal(serialized[key], pairs[key]) for key in pairs)
        metadata = dict(kind='diagnostic-prefix6-handoff-mpc-correction-supervised-pairs',
                        rows=steps - prefix_controls, prefix_rows_excluded=prefix_controls,
                        strict_force_success_of_source_episode=audit['strict_force_success'],
                        data_sha256=sha(D / 'actual_mpc_correction_pairs.npz'),
                        source_trajectory='recovery/transitions.npz', source_trajectory_sha256=final['trajectory_sha256'],
                        observation_dimension=60, policy_interface={'kind': 'absolute_joint_target'},
                        no_next_actions_fabricated=True, not_a_cold_reset_replay_dataset=True,
                        source_prefix_sha256=report['source_prefix_sha256'],
                        production_eligible=False, main_demonstrations=0, SAC_successes=0, learning_updates=0,
                        row_provenance=[dict(row=index - prefix_controls, source_transition_row_0based=index,
                                             source_control_1based=index + 1, origin='actually_executed_corrected_native_mpc')
                                        for index in range(prefix_controls, steps)])
        write_json(D / 'actual_mpc_correction_pairs.metadata.json', metadata)
        final.update(correction_pairs_sha256=metadata['data_sha256'],
                     correction_pairs_metadata_sha256=sha(D / 'actual_mpc_correction_pairs.metadata.json'))
    else:
        final['correction_pairs'] = 'none: no MPC control was executed'
    write_json(D / 'final_report.json', final)
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit('Prepared runner only. Execution requires separate --execute after review.')
    if (D / 'runtime.json').exists():
        raise SystemExit('This stage already started; no overwrite or implicit restart')
    verify_manifest()
    protocol = json.loads((D / 'protocol.json').read_text())
    if protocol['status'] != 'frozen-ready-for-separate-explicit-execution':
        raise RuntimeError('Protocol has not been frozen')
    ec = EnvConfig(**json.loads((D / 'config.json').read_text())['environment'])
    assert ec.task == 'rotation' and ec.object_diameter == .04 and ec.horizon_steps == 100 and ec.terminate_on_success
    write_json(D / 'runtime.json', dict(python=sys.version, platform=platform.platform(), numpy=np.__version__,
                                        mujoco=mujoco.__version__, pid=os.getpid(), argv=sys.argv,
                                        policy_inference_or_training=False,
                                        numerical_package_root=str(Path(sys.modules['mpc_dex'].__file__).resolve().parent),
                                        frozen_execution_manifest_sha256=sha(D / 'execution_manifest.json')))
    summary = dict(status='running', protocol=protocol, production_eligible=False, learning_updates=0, main_demonstrations=0)
    write_json(D / 'summary.json', summary)
    try:
        report, trace, physics, audit, full = run_branch(ec, protocol)
        final = package(report, trace, physics, audit, full, protocol)
        verify_manifest()
        summary.update(status='complete', outcome=dict(
            branch_status=report['status'], mpc_controls=report['executed_mpc_controls'],
            strict_force_success=audit['strict_force_success'],
            handoff_force_qualified_angle=report['branch_force_qualified_angle'],
            final_force_qualified_angle=audit['final_force_qualified_angle'],
            maximum_force_qualified_angle=audit['max_force_qualified_angle'],
            drop=audit['independent_drop'], palm=audit['independent_palm_support_violation'],
            no_external_support=audit['no_external_support'], report_path='recovery/report.json'),
            strict_force_successes=int(audit['strict_force_success']), final_report_sha256=sha(D / 'final_report.json'))
        print(json.dumps(dict(event='branch_complete', **summary['outcome'])), flush=True)
    except BaseException as error:
        summary.update(status='fault', error=repr(error))
        raise
    finally:
        write_json(D / 'summary.json', summary)
        write_json(D / 'manifest.json', {str(path.relative_to(D)): sha(path) for path in sorted(D.rglob('*'))
                   if path.is_file() and '__pycache__' not in path.parts and path.name not in ('manifest.json', 'run.log')})


if __name__ == '__main__':
    main()
