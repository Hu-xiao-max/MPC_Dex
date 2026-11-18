"""Collect verified full-turn demonstrations from cold reset with the corrected native MPC and the diagnostic prior.

Usage: collect.py <seed_start> <seed_end_exclusive>   (one process per seed range; 8 native workers each)
Each attempt: cold reset (physical seed 42), CEM seed = attempt seed, at most 40 controls, stop at the first
environment boundary. A strict success is re-executed from a fresh cold reset (independent replay with a
separately recomputed force audit) and must match bit-for-bit; only then is it stored under
episodes/seed_<n>/ with obs/actions/rewards/next_obs/flags/qpos/qvel/targets/integration states and the
2 ms physics samples. Failures are recorded in attempts.jsonl. The prior is the measured 40 mm diagnostic PCA
prior (not a SAC-derived primitive), a disclosed deviation from the paper's primitive; nothing here is
eligible for the main-pipeline gates.
"""
import hashlib, importlib.util, json, sys, time
from pathlib import Path
R = Path('/home/autel001/code/mpc_dex/artifacts/mpc_prefix6_handoff_recovery_0001')
sys.path.insert(0, str(R / 'source')); sys.path.insert(0, str(R))
import numpy as np
from mpc_dex.env import AllegroRotationEnv, EnvConfig
from mpc_dex.mpc import CEMConfig, NoFeasiblePlan
from mpc_dex.primitives import PrimitiveBasis
from mpc_dex.rollouts import RolloutCostConfig
from mpc_dex.native_rollouts import NativeMujocoRolloutEvaluator
from mpc_dex.contracts import sha256
from scripts.probe_mpc import action_search_bounds
from scripts.audit_policy import PhysicsObserver, PHYSICS_COLUMNS, audit_episode, write_json
from scripts.audit_recorded_controls import RecordedPolicy
from recovery_recipe import ProjectedReferenceCEM, integration_state
OUT = Path(__file__).resolve().parent
protocol = json.loads((R / 'protocol.json').read_text())
ec = EnvConfig(**json.loads((R / 'config.json').read_text())['environment'])
FIELDS = ('obs', 'actions', 'rewards', 'next_obs', 'terminated', 'truncated', 'qpos', 'qvel', 'targets', 'integration_state')
MAX_CONTROLS = 40


def run_attempt(seed):
    env = AllegroRotationEnv(ec); obs, info = env.reset(seed=42)
    initial = dict(obs=obs.copy(), qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(), ctrl=env.data.ctrl.copy(), integration_state=integration_state(env))
    q_ids = np.asarray(env.q_ids).copy()
    low, high, search = action_search_bounds(env, protocol['action_bound'])
    with np.load(R / 'inputs/previous_recovery/prior_transitions.npz') as a:
        frames = np.vstack((env.data.qpos[env.q_ids].copy(), a['qpos'][:, env.q_ids]))
    prior = PrimitiveBasis.from_trajectories(frames, provenance=dict(source=str(R / 'inputs/previous_recovery/prior_transitions.npz'), sha256=sha256(R / 'inputs/previous_recovery/prior_transitions.npz'), purpose='measured 40 mm diagnostic PCA prior; not a SAC primitive', successful_policy=False))
    observer = PhysicsObserver(env)
    planner = ProjectedReferenceCEM(prior, low, high, CEMConfig(**protocol['cem']), seed)
    planner.execution_start = env.data.ctrl.copy(); planner.reset(env.data.ctrl.copy())
    evaluator = NativeMujocoRolloutEvaluator(env, timestep_seconds=protocol['cem']['timestep_seconds'], config=RolloutCostConfig(**protocol['cost']), require_native=True, cache_directory=R / 'native_cache')
    rows = {k: [] for k in FIELDS}; started = time.monotonic(); status = 'budget'; planning = []
    try:
        for index in range(MAX_CONTROLS):
            before = integration_state(env); planner.execution_start = env.data.ctrl.copy(); t0 = time.monotonic()
            try:
                target, diagnostics = planner.plan(evaluator)
            except NoFeasiblePlan:
                status = 'no_feasible_plan'; break
            assert np.array_equal(before, integration_state(env))
            planning.append(time.monotonic() - t0)
            action = env.to_action(target); assert float(abs(action).max()) <= search['effective_bound']
            prev = obs.copy(); obs, reward, term, trunc, info = env.step(action)
            for k, v in dict(obs=prev, actions=action.copy(), rewards=reward, next_obs=obs.copy(), terminated=term, truncated=trunc, qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(), targets=env.data.ctrl.copy(), integration_state=integration_state(env)).items():
                rows[k].append(v)
            if term or trunc:
                status = 'boundary'; break
        physics = np.asarray(observer.rows, np.float64)
        col = {k: physics[:, i] for i, k in enumerate(PHYSICS_COLUMNS)}
        external = ('palm_force_norm', 'table_force_norm', 'other_force_norm', 'table_normal_force', 'other_normal_force', 'object_applied_wrench_norm', 'object_applied_generalized_force_norm')
        no_external = not observer.palm_support_violation and all(np.max(col[k], initial=0) <= 1e-6 for k in external)
        strict = bool(status == 'boundary' and info.get('success') and not info.get('drop') and col['force_bearing_fingers'][-1] >= 2 and observer.raw_angle >= 2 * np.pi and observer.qualified_angle >= 2 * np.pi and observer.force_qualified_angle >= 2 * np.pi and col['force_retained'][-1] and not observer.independent_drop and no_external and env.model.neq == 0)
        report = dict(seed=seed, status=status, controls=len(rows['actions']), strict_force_success=strict, environment_success=bool(info.get('success')), failure_reason=info.get('failure_reason'),
                      final_force_qualified_angle=observer.force_qualified_angle, palm=bool(observer.palm_support_violation), drop=bool(observer.independent_drop), no_external_support=no_external,
                      planning_seconds_total=float(sum(planning)), wall_seconds=time.monotonic() - started, raw_return=float(np.sum(rows['rewards'])) if rows['rewards'] else 0.)
    finally:
        observer.close(); evaluator.close(); env.close()
    report['hand_joint_qpos_indices'] = q_ids.tolist()
    return report, {k: np.asarray(v) for k, v in rows.items()}, initial, physics


def independent_replay(actions, original, initial, original_physics):
    env = AllegroRotationEnv(ec); rows = {k: [] for k in FIELDS}; init = {}
    reset, step = env.reset, env.step; recorded = RecordedPolicy(actions)
    def recording_reset(*a, **k):
        result = reset(*a, **k); init.update(obs=result[0].copy(), qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(), ctrl=env.data.ctrl.copy(), integration_state=integration_state(env)); return result
    class Pol:
        def act(self, obs, deterministic=True):
            a = recorded.act(obs, deterministic=deterministic); rows['obs'].append(obs.copy()); rows['actions'].append(a.copy()); return a
    def recording_step(a):
        result = step(a); ob, reward, term, trunc, _ = result
        for k, v in dict(next_obs=ob.copy(), rewards=reward, terminated=term, truncated=trunc, qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(), targets=env.data.ctrl.copy(), integration_state=integration_state(env)).items():
            rows[k].append(v)
        return result
    env.reset, env.step = recording_reset, recording_step
    try:
        report, physics, commands = audit_episode(env, Pol(), 42, len(actions), True)
        same = all(np.array_equal(np.asarray(rows[k]), original[k]) for k in FIELDS) and all(np.array_equal(init[k], initial[k]) for k in initial) and np.array_equal(physics, original_physics)
        report['bitexact_replay'] = bool(same)
        report['strict_force_success'] = bool(report['environment_success'] and report['finger_supported_rotation_success'] and report['terminated'] and not report['truncated'] and not report['environment_drop'] and not report['independent_drop'] and not report['independent_palm_support_violation'] and report['no_external_support'])
        return report
    finally:
        env.close()


def main():
    start, end = int(sys.argv[1]), int(sys.argv[2])
    (OUT / 'episodes').mkdir(exist_ok=True)
    with (OUT / f'attempts_{start}_{end}.jsonl').open('a') as log:
        for seed in range(start, end):
            report, rows, initial, physics = run_attempt(seed)
            if report['strict_force_success']:
                audit = independent_replay(rows['actions'], rows, initial, physics)
                report['independent_audit'] = dict(bitexact_replay=audit['bitexact_replay'], strict_force_success=audit['strict_force_success'], final_force_qualified_angle=audit['final_force_qualified_angle'], final_force_bearing_fingers=audit['final_force_bearing_fingers'])
                if audit['bitexact_replay'] and audit['strict_force_success']:
                    d = OUT / 'episodes' / f'seed_{seed}'; d.mkdir(exist_ok=True)
                    np.savez_compressed(d / 'transitions.npz', **rows, **{'initial_' + k: v for k, v in initial.items()}, joint_positions=np.vstack((initial['qpos'][np.asarray(report['hand_joint_qpos_indices'])][None, :], rows['qpos'][:, np.asarray(report['hand_joint_qpos_indices'])])))
                    np.savez_compressed(d / 'physics.npz', samples=physics, columns=np.asarray(PHYSICS_COLUMNS))
                    write_json(d / 'report.json', dict(report, transitions_sha256=sha256(d / 'transitions.npz'), physics_sha256=sha256(d / 'physics.npz'), stored=True))
                    report['stored'] = True
            log.write(json.dumps(report) + '\n'); log.flush()
            print(json.dumps({k: report[k] for k in ('seed', 'status', 'controls', 'strict_force_success', 'wall_seconds')} | {'stored': report.get('stored', False)}), flush=True)


if __name__ == '__main__':
    main()
