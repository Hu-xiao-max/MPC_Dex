"""Verified cold-reset MPC demonstrations at a non-archived control period (route b).

Usage: collect.py <period_seconds> <seed_start> <seed_end_exclusive> [--max-controls 40] [--out DIR]
Same contract as mpc_cold_demonstrations_0001/collect.py (cold reset seed 42, CEM seed = attempt seed, strict force-qualified
success, independent re-execution from a fresh reset with a separately recomputed force audit, stored only on a bit-exact
match, demonstration layout with physics samples), except that the environment's control period is <period_seconds> and the
planner step / horizon scale with it (step = period, horizon = 5 periods, 6 knots). The stored report carries the period,
the physics substeps per control and the environment config used. Nothing here is primitive-eligible.
"""
import argparse, json, sys, time
from pathlib import Path
A = Path('/home/autel001/code/mpc_dex/artifacts'); R = A / 'mpc_prefix6_handoff_recovery_0001'; HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(R / 'source')); sys.path.insert(0, str(R)); sys.path.insert(0, str(HERE))
import numpy as np
import probe
from probe import make_env, strict_outcome, protocol, FIELDS, base_env
from mpc_dex.mpc import CEMConfig, NoFeasiblePlan
from mpc_dex.primitives import PrimitiveBasis
from mpc_dex.rollouts import RolloutCostConfig
from mpc_dex.native_rollouts import NativeMujocoRolloutEvaluator
from mpc_dex.contracts import sha256
from scripts.probe_mpc import action_search_bounds
from scripts.audit_policy import PhysicsObserver, PHYSICS_COLUMNS, audit_episode, write_json
from scripts.audit_recorded_controls import RecordedPolicy
from recovery_recipe import ProjectedReferenceCEM, integration_state


FLOOR_SCALE = 1.


def run_attempt(period, seed, max_controls):
    env = make_env(period); obs, info = env.reset(seed=42)
    initial = dict(obs=obs.copy(), qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(), ctrl=env.data.ctrl.copy(), integration_state=integration_state(env)); q_ids = np.asarray(env.q_ids).copy()
    low, high, search = action_search_bounds(env, protocol['action_bound'])
    with np.load(R / 'inputs/previous_recovery/prior_transitions.npz') as a:
        frames = np.vstack((env.data.qpos[env.q_ids].copy(), a['qpos'][:, env.q_ids]))
    prior = PrimitiveBasis.from_trajectories(frames, provenance=dict(source=str(R / 'inputs/previous_recovery/prior_transitions.npz'), sha256=sha256(R / 'inputs/previous_recovery/prior_transitions.npz'), purpose='measured 40 mm diagnostic PCA prior; not a SAC primitive', successful_policy=False))
    cem = CEMConfig(**dict(protocol['cem'], control_seconds=period, timestep_seconds=period, horizon_seconds=5 * period, knots=6))
    observer = PhysicsObserver(env); planner = ProjectedReferenceCEM(prior, low, high, cem, seed); planner.execution_start = env.data.ctrl.copy(); planner.reset(env.data.ctrl.copy())
    planner.exploration_floor = planner.exploration_floor * FLOOR_SCALE; planner.refinement_floor = planner.refinement_floor * FLOOR_SCALE  # proposal spread scaled with the period (about 0.1 / period)
    evaluator = NativeMujocoRolloutEvaluator(env, timestep_seconds=period, config=RolloutCostConfig(**protocol['cost']), require_native=True, cache_directory=HERE / 'native_cache')
    rows = {k: [] for k in FIELDS}; started = time.monotonic(); status = 'budget'; planning = []; term = trunc = False
    try:
        for _ in range(max_controls):
            before = integration_state(env); planner.execution_start = env.data.ctrl.copy(); t0 = time.monotonic()
            try: target, diag = planner.plan(evaluator)
            except NoFeasiblePlan: status = 'no_feasible_plan'; break
            assert np.array_equal(before, integration_state(env)); planning.append(time.monotonic() - t0)
            action = env.to_action(target); assert float(abs(action).max()) <= search['effective_bound']
            prev = obs.copy(); obs, reward, term, trunc, info = env.step(action)
            for k, v in dict(obs=prev, actions=action.copy(), rewards=reward, next_obs=obs.copy(), terminated=term, truncated=trunc, qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(), targets=env.data.ctrl.copy(), integration_state=integration_state(env)).items(): rows[k].append(v)
            if term or trunc: status = 'terminated' if term else 'truncated_by_horizon'; break
        ok, physics = strict_outcome(env, observer, info, term, trunc)
        report = dict(period=period, seed=seed, status=status, controls=len(rows['actions']), strict_force_success=ok, environment_success=bool(info.get('success')), failure_reason=info.get('failure_reason'), final_force_qualified_angle=observer.force_qualified_angle,
                      substeps=env.substeps, environment_config=dict(env.config.__dict__) if hasattr(env.config, '__dict__') else None, env_overrides=dict(probe.ENV_OVERRIDES), object_shape=env.config.object_shape, object_diameter=env.config.object_diameter, planning_seconds_total=float(sum(planning)), wall_seconds=time.monotonic() - started, hand_joint_qpos_indices=q_ids.tolist(), cem=dict(cem.__dict__), floor_scale=FLOOR_SCALE, horizon_controls=env.config.horizon_steps)
    finally:
        observer.close(); evaluator.close(); env.close()
    return report, {k: np.asarray(v) for k, v in rows.items()}, initial, physics


def independent_replay(period, actions, original, initial, original_physics):
    env = make_env(period); rows = {k: [] for k in FIELDS}; init = {}; reset, step = env.reset, env.step; recorded = RecordedPolicy(actions)
    def recording_reset(*a, **k):
        result = reset(*a, **k); init.update(obs=result[0].copy(), qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(), ctrl=env.data.ctrl.copy(), integration_state=integration_state(env)); return result
    class Pol:
        def act(self, obs, deterministic=True):
            a = recorded.act(obs, deterministic=deterministic); rows['obs'].append(obs.copy()); rows['actions'].append(a.copy()); return a
    def recording_step(a):
        result = step(a); ob, reward, term, trunc, _ = result
        for k, v in dict(next_obs=ob.copy(), rewards=reward, terminated=term, truncated=trunc, qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(), targets=env.data.ctrl.copy(), integration_state=integration_state(env)).items(): rows[k].append(v)
        return result
    env.reset, env.step = recording_reset, recording_step
    try:
        report, physics, commands = audit_episode(env, Pol(), 42, len(actions), True)
        same = all(np.array_equal(np.asarray(rows[k]), original[k]) for k in FIELDS) and all(np.array_equal(init[k], initial[k]) for k in initial) and np.array_equal(physics, original_physics)
        report['bitexact_replay'] = bool(same)
        report['strict_force_success'] = bool(report['environment_success'] and report['finger_supported_rotation_success'] and report['terminated'] and not report['truncated'] and not report['environment_drop'] and not report['independent_drop'] and not report['independent_palm_support_violation'] and report['no_external_support'])
        return report
    finally: env.close()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('period', type=float); ap.add_argument('seed_start', type=int); ap.add_argument('seed_end', type=int); ap.add_argument('--max-controls', type=int, default=40); ap.add_argument('--out', default=None); ap.add_argument('--floor-scale', type=float, default=None, help='proposal spread scale (default 0.1 / period)'); ap.add_argument('--shape', default=None, choices=[None, 'hexagonal', 'circular']); ap.add_argument('--diameter', type=float, default=None)
    a = ap.parse_args(); P = a.period; global FLOOR_SCALE; FLOOR_SCALE = a.floor_scale if a.floor_scale is not None else round(0.1 / P, 4)
    if a.shape: probe.ENV_OVERRIDES['object_shape'] = a.shape
    if a.diameter: probe.ENV_OVERRIDES['object_diameter'] = float(a.diameter)
    OUT = Path(a.out).resolve() if a.out else HERE / f'demos_P{int(round(P * 1000)):04d}ms'; (OUT / 'episodes').mkdir(parents=True, exist_ok=True)
    with (OUT / f'attempts_{a.seed_start}_{a.seed_end}.jsonl').open('a') as log:
        for seed in range(a.seed_start, a.seed_end):
            if (OUT / 'episodes' / f'seed_{seed}' / 'report.json').exists(): continue
            report, rows, initial, physics = run_attempt(P, seed, a.max_controls)
            if report['strict_force_success']:
                audit = independent_replay(P, rows['actions'], rows, initial, physics)
                report['independent_audit'] = dict(bitexact_replay=audit['bitexact_replay'], strict_force_success=audit['strict_force_success'], final_force_qualified_angle=audit['final_force_qualified_angle'], final_force_bearing_fingers=audit.get('final_force_bearing_fingers'))
                if audit['bitexact_replay'] and audit['strict_force_success']:
                    d = OUT / 'episodes' / f'seed_{seed}'; d.mkdir(exist_ok=True)
                    np.savez_compressed(d / 'transitions.npz', **rows, **{'initial_' + k: v for k, v in initial.items()}, joint_positions=np.vstack((initial['qpos'][np.asarray(report['hand_joint_qpos_indices'])][None, :], rows['qpos'][:, np.asarray(report['hand_joint_qpos_indices'])])))
                    np.savez_compressed(d / 'physics.npz', samples=physics, columns=np.asarray(PHYSICS_COLUMNS))
                    write_json(d / 'report.json', dict(report, transitions_sha256=sha256(d / 'transitions.npz'), physics_sha256=sha256(d / 'physics.npz'), stored=True)); report['stored'] = True
            log.write(json.dumps(report, default=str) + '\n'); log.flush()
            print(json.dumps({k: report[k] for k in ('period', 'seed', 'status', 'controls', 'strict_force_success', 'wall_seconds')} | {'stored': report.get('stored', False)}), flush=True)


if __name__ == '__main__':
    main()
