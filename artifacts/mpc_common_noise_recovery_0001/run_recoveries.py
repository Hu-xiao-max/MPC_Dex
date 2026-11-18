"""Five fixed noisy prefixes followed by one prescribed MPC recovery each.

Artifact-local diagnostic only. No learning, source edits, prefix selection,
post-success holds, parameter sweeps, or production demonstration exports.
"""
from pathlib import Path
from dataclasses import asdict
from collections import Counter
import json
import os
import platform
import sys
import time

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D / 'source'))
import mujoco
import numpy as np
import torch
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

FIELDS = ('obs', 'actions', 'rewards', 'next_obs', 'terminated', 'truncated',
          'qpos', 'qvel', 'targets', 'integration_state')
SEEDS = tuple(range(17021, 17026))


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def initial_state(env):
    return dict(qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(),
                ctrl=env.data.ctrl.copy(), joint_positions=env.data.qpos[env.q_ids].copy(),
                integration_state=integration_state(env), time=float(env.data.time))


def save_transitions(path, initial, rows, q_ids):
    arrays = {key: np.asarray(rows[key]) for key in FIELDS}
    joints = np.vstack((initial['joint_positions'], arrays['qpos'][:, q_ids]))
    np.savez_compressed(path, **arrays,
        **{'initial_' + key: value for key, value in initial.items()},
        joint_positions=joints, control_times=np.r_[0., rows['control_times']])


def verify_inputs():
    inputs = json.loads((D / 'inputs_manifest.json').read_text())
    for name, digest in inputs.items():
        if sha256(D / name) != digest:
            raise RuntimeError(f'Frozen input changed: {name}')


def independent_audit(ec, seed, actions, output, original, original_physics):
    """Fresh cold-reset physical replay, with separately recomputed force audit."""
    output.mkdir()
    env = AllegroRotationEnv(ec)
    initial, rows = {}, {key: [] for key in (*FIELDS, 'control_times')}
    reset, step = env.reset, env.step
    recorded = RecordedPolicy(actions)

    def recording_reset(*args, **kwargs):
        result = reset(*args, **kwargs)
        initial.update(initial_state(env))
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
        for key, value in dict(next_obs=ob.copy(), rewards=reward, terminated=terminated,
            truncated=truncated, qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(),
            targets=env.data.ctrl.copy(), integration_state=integration_state(env)).items():
            rows[key].append(value)
        rows['control_times'].append(float(env.data.time) - initial['time'])
        return result

    env.reset, env.step = recording_reset, recording_step
    try:
        report, physics, commands = audit_episode(env, RecordingPolicy(), seed, len(actions), True)
        save_transitions(output / 'transitions.npz', initial, rows, env.q_ids)
        np.savez_compressed(output / 'physics.npz', samples=physics, columns=np.asarray(PHYSICS_COLUMNS))
        write_json(output / 'commands.json', commands)
        comparisons = {}
        replay = read_npz(output / 'transitions.npz')
        for key in original:
            comparisons[key] = bool(np.array_equal(replay[key], original[key]))
        comparisons['physics'] = bool(np.array_equal(physics, original_physics))
        if not all(comparisons.values()):
            raise RuntimeError(f'Independent execution mismatch: {comparisons}')
        report.update(policy_kind='actual_fixed_mean_noisy_prefix_then_corrected_mpc',
            production_eligible=False, policy_only_success=False, prefix_controls=9,
            mpc_controls=len(actions) - 9, replay_bitexact=comparisons,
            strict_force_success=bool(report['environment_success']
                and report['finger_supported_rotation_success']
                and report['terminated'] and not report['truncated']
                and not report['environment_drop'] and not report['independent_drop']
                and not report['independent_palm_support_violation'] and report['no_external_support']))
        write_json(output / 'report.json', report)
        return report
    finally:
        env.close()


def run_branch(seed, ec, protocol):
    output = D / f'seed_{seed}'
    output.mkdir()
    source = D / 'inputs/noisy_prefixes' / f'seed_{seed}'
    prefix = read_npz(source / 'transitions.npz')
    prefix_physics = read_npz(source / 'physics.npz')
    assert len(prefix['actions']) == 9 and not prefix['terminated'].any() and not prefix['truncated'].any()
    assert np.array_equal(prefix_physics['columns'], PHYSICS_COLUMNS)
    env = AllegroRotationEnv(ec)
    obs, info = env.reset(seed=seed)
    initial = initial_state(env)
    for key in ('qpos', 'qvel', 'ctrl', 'joint_positions', 'integration_state'):
        assert np.array_equal(initial[key], prefix['initial_' + key]), key
    low, high, search = action_search_bounds(env, .98)
    prior_data = read_npz(D / 'inputs/previous_recovery/prior_transitions.npz')
    frames = np.vstack((initial['joint_positions'], prior_data['qpos'][:, env.q_ids]))
    prior = PrimitiveBasis.from_trajectories(frames, provenance=dict(
        source=str(D / 'inputs/previous_recovery/prior_transitions.npz'),
        sha256=sha256(D / 'inputs/previous_recovery/prior_transitions.npz'),
        purpose='Identical diagnostic measured 40 mm PCA prior; not a SAC primitive', successful_policy=False))
    expected = json.loads((D / 'inputs/previous_recovery/reference_planner_report.json').read_text())
    for key in ('mean', 'basis', 'coefficient_min', 'coefficient_max', 'explained_variance_ratio'):
        assert np.array_equal(getattr(prior, key), expected['diagnostic_basis'][key]), key
    prior.save(output / 'diagnostic_prior.npz')
    observer = PhysicsObserver(env)
    rows = {key: [] for key in (*FIELDS, 'control_times')}
    commands, evaluator, planner = [], None, None
    began = time.monotonic()
    report = dict(seed=seed, planner_seed=42, status='prefix_replay',
        source_prefix_sha256=sha256(source / 'transitions.npz'), source_draws_sha256=sha256(source / 'matched_noise.npz'),
        environment_contract=environment_contract(ec), prefix_controls=9, max_mpc_controls=20,
        action_search=search, production_eligible=False, main_demonstrations=0,
        policy_only_success=False, learning_updates=0, commands=commands)

    def execute(action, origin, extra=None):
        nonlocal obs, info
        before, previous = obs.copy(), env.data.ctrl.copy()
        obs, reward, terminated, truncated, info = env.step(action)
        for key, value in dict(obs=before, actions=action.copy(), rewards=reward, next_obs=obs.copy(),
            terminated=terminated, truncated=truncated, qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(),
            targets=env.data.ctrl.copy(), integration_state=integration_state(env)).items():
            rows[key].append(value)
        rows['control_times'].append(float(env.data.time) - initial['time'])
        commands.append(dict(control=len(commands) + 1, origin=origin,
            reward=float(reward), raw_angle=observer.raw_angle, retained_angle=observer.qualified_angle,
            force_qualified_angle=observer.force_qualified_angle, finger_contacts=int(info['finger_contacts']),
            axis_tilt_radians=float(info['axis_tilt_radians']), drop=bool(info['drop']),
            environment_success=bool(info['success']), palm_support_violation=bool(info['palm_support_violation']),
            failure_reason=info['failure_reason'], target_delta_max=float(abs(env.data.ctrl - previous).max()),
            action_abs_max=float(abs(action).max()), **(extra or {})))
        assert len(observer.rows) == len(commands) * env.substeps
        return terminated or truncated

    try:
        for index in range(9):
            assert not execute(prefix['actions'][index], 'recorded_fixed_mean_noisy_prefix')
            for key in FIELDS:
                assert np.array_equal(rows[key][-1], prefix[key][index]), (seed, index, key)
        assert np.array_equal(np.asarray(observer.rows), prefix_physics['samples'])
        branch = integration_state(env)
        np.savez_compressed(output / 'actual_branch_state.npz', qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(),
            ctrl=env.data.ctrl.copy(), observation=obs.copy(), integration_state=branch,
            search_low=low, search_high=high, raw_angle=observer.raw_angle,
            retained_angle=observer.qualified_angle, force_qualified_angle=observer.force_qualified_angle)
        report.update(prefix_transition_physics_bitexact=True, branch_time_seconds=float(env.data.time),
            branch_force_qualified_angle=observer.force_qualified_angle,
            true_live_execution_start=env.data.ctrl.tolist(),
            live_start_out_of_search_dimensions=np.flatnonzero((env.data.ctrl < low) | (env.data.ctrl > high)).tolist())
        planner = ProjectedReferenceCEM(prior, low, high, CEMConfig(**protocol['cem']), 42)
        planner.execution_start = env.data.ctrl.copy()
        planner.reset(env.data.ctrl.copy())
        assert np.array_equal(branch, integration_state(env))
        cost = RolloutCostConfig(**protocol['cost'])
        evaluator = NativeMujocoRolloutEvaluator(env, timestep_seconds=.1, config=cost,
            require_native=True, cache_directory=D / 'native_cache')
        report.update(status='mpc_recovery', cem=asdict(planner.config), cost=asdict(cost),
            native_build=evaluator.native_provenance, no_slew_limit=True)
        write_json(output / 'report.json', report)
        for index in range(1, 21):
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
            boundary = execute(action, 'actual_corrected_native_mpc', dict(recovery_control=index,
                planning_seconds=time.monotonic() - planning_started, planner=diagnostics, prediction=prediction,
                reference_candidates=planner.reference_records[reference_begin:],
                true_live_execution_start=planner.execution_start.tolist(), selected_target=target.tolist(),
                executed_target=executed.tolist(), target_mapping_max_radians=float(abs(target - executed).max()),
                proposals_and_execution_within_bounds=True, planning_left_live_integration_bitexact=True))
            write_json(output / 'report.json', report)
            print(json.dumps(dict(seed=seed, **{key: value for key, value in commands[-1].items()
                if key not in ('planner', 'prediction', 'reference_candidates', 'selected_target',
                               'executed_target', 'true_live_execution_start')})), flush=True)
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
            executed_mpc_controls=max(0, len(commands) - 9),
            reference_records=planner.reference_records if planner is not None else [],
            final_info={key: info[key] for key in ('angle', 'retained_angle', 'success', 'drop',
                'palm_support_violation', 'failure_reason', 'finger_contacts', 'axis_tilt_radians')})
        write_json(output / 'report.json', report)
        save_transitions(output / 'transitions.npz', initial, rows, env.q_ids)
        np.savez_compressed(output / 'physics.npz', samples=np.asarray(observer.rows), columns=np.asarray(PHYSICS_COLUMNS))
        np.savez_compressed(output / 'control_origins.npz', origin=np.asarray([row['origin'] for row in commands]),
            expert_correction_mask=np.arange(len(commands)) >= 9)
        observer.close()
        if evaluator is not None:
            evaluator.close()
        env.close()
    original = read_npz(output / 'transitions.npz')
    physics = read_npz(output / 'physics.npz')['samples']
    audit = independent_audit(ec, seed, original['actions'], output / 'independent_audit', original, physics)
    report.update(independent_audit=audit, strict_force_success=audit['strict_force_success'])
    if info['success'] and not audit['strict_force_success']:
        report['qualification_failure'] = 'Environment termination did not satisfy independent force qualification; no continuation executed'
    write_json(output / 'report.json', report)
    verify_inputs()
    print(json.dumps(dict(event='branch_complete', seed=seed, status=report['status'],
        mpc_controls=report['executed_mpc_controls'], strict_force_success=report['strict_force_success'],
        final_force_qualified_angle=audit['final_force_qualified_angle'],
        maximum_force_qualified_angle=audit['max_force_qualified_angle'])), flush=True)
    return report


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    verify_inputs()
    ec = EnvConfig(**json.loads((D / 'config.json').read_text())['environment'])
    protocol = json.loads((D / 'protocol.json').read_text())
    assert ec.terminate_on_success and ec.object_diameter == .04 and ec.horizon_steps == 100
    write_json(D / 'runtime.json', dict(python=sys.version, platform=platform.platform(),
        numpy=np.__version__, mujoco=mujoco.__version__, torch=str(torch.__version__),
        torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
        gpu_inference_or_training=False, pid=os.getpid(), argv=sys.argv,
        numerical_package_root=str(Path(sys.modules['mpc_dex'].__file__).resolve().parent)))
    results = []
    summary = dict(status='running', protocol=protocol, outcomes=results, production_eligible=False,
        learning_updates=0, main_demonstrations=0)
    write_json(D / 'summary.json', summary)
    try:
        for seed in SEEDS:
            report = run_branch(seed, ec, protocol)
            results.append(dict(seed=seed, status=report['status'],
                mpc_controls=report['executed_mpc_controls'], strict_force_success=report['strict_force_success'],
                branch_force_qualified_angle=report['branch_force_qualified_angle'],
                final_force_qualified_angle=report['independent_audit']['final_force_qualified_angle'],
                maximum_force_qualified_angle=report['independent_audit']['max_force_qualified_angle'],
                drop=report['independent_audit']['independent_drop'],
                palm=report['independent_audit']['independent_palm_support_violation'],
                no_external_support=report['independent_audit']['no_external_support'],
                report_path=f'seed_{seed}/report.json'))
            write_json(D / 'summary.json', summary)
        summary.update(status='complete', completed_branches=len(results),
            strict_force_successes=sum(row['strict_force_success'] for row in results))
    except BaseException as error:
        summary.update(status='fault', error=repr(error), completed_branches=len(results))
        raise
    finally:
        write_json(D / 'summary.json', summary)
        write_json(D / 'manifest.json', {str(path.relative_to(D)): sha256(path) for path in sorted(D.rglob('*'))
            if path.is_file() and '__pycache__' not in path.parts and path.name not in ('manifest.json', 'run.log')})


if __name__ == '__main__':
    main()
