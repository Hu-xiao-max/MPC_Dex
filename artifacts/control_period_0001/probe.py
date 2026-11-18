"""Control-period feasibility probe (route b): with everything else archived, change only the control period P of the
environment, run cold-reset MPC demonstrations with the archived corrected recipe (planner step = P, horizon = 5 P, 6 knots,
CEM 350/10/3, diagnostic prior, .98 bound), and measure the quantity that decided route (a): the open-loop tolerance of a
successful demonstration (perturb control j by L2 r, replay the remaining recorded controls, hold the last one up to 20
extra controls). Nothing here is primitive-eligible; a changed control period is a disclosed deviation from the archived task.

Usage: probe.py <period_seconds> <seed_from> <seed_to> [--max-successes 3] [--radii 0.003,0.01,0.03,0.1] [--dirs 3]
"""
import argparse, json, math, sys, time
from pathlib import Path
A = Path('/home/autel001/code/mpc_dex/artifacts'); R = A / 'mpc_prefix6_handoff_recovery_0001'; OUT = Path(__file__).resolve().parent; HERE_CACHE = OUT / 'native_cache'
sys.path.insert(0, str(R / 'source')); sys.path.insert(0, str(R))
import numpy as np
from mpc_dex.env import AllegroRotationEnv, EnvConfig
from mpc_dex.mpc import CEMConfig, NoFeasiblePlan
from mpc_dex.primitives import PrimitiveBasis
from mpc_dex.rollouts import RolloutCostConfig
from mpc_dex.native_rollouts import NativeMujocoRolloutEvaluator
from mpc_dex.contracts import sha256
from scripts.probe_mpc import action_search_bounds
from scripts.audit_policy import PhysicsObserver, PHYSICS_COLUMNS
from recovery_recipe import ProjectedReferenceCEM, integration_state
protocol = json.loads((R / 'protocol.json').read_text()); base_env = json.loads((R / 'config.json').read_text())['environment']
FLOOR_SCALE = 1.
FIELDS = ('obs', 'actions', 'rewards', 'next_obs', 'terminated', 'truncated', 'qpos', 'qvel', 'targets', 'integration_state')


ENV_OVERRIDES = {}  # route (b) diagnostics: e.g. object_shape / object_diameter


def make_env(period):
    cfg = dict(base_env); cfg['control_seconds'] = float(period); cfg.update(ENV_OVERRIDES)
    cfg['horizon_steps'] = int(base_env['horizon_steps'])  # the archived horizon is 100 controls (10 s at 0.1 s); kept as a count of control decisions, since the planner's per-control motion is period-independent
    return AllegroRotationEnv(EnvConfig(**cfg))


def strict_outcome(env, observer, info, term, trunc):
    physics = np.asarray(observer.rows, np.float64)
    if physics.ndim != 2 or len(physics) == 0: return False, physics.reshape(0, len(PHYSICS_COLUMNS))
    col = {k: physics[:, i] for i, k in enumerate(PHYSICS_COLUMNS)}
    ext = ('palm_force_norm', 'table_force_norm', 'other_force_norm', 'table_normal_force', 'other_normal_force', 'object_applied_wrench_norm', 'object_applied_generalized_force_norm')
    no_external = not observer.palm_support_violation and all(np.max(col[k], initial=0) <= 1e-6 for k in ext)
    ok = bool(term and not trunc and info.get('success') and not info.get('drop') and col['force_bearing_fingers'][-1] >= 2 and observer.raw_angle >= 2 * np.pi and observer.qualified_angle >= 2 * np.pi and observer.force_qualified_angle >= 2 * np.pi and col['force_retained'][-1] and not observer.independent_drop and no_external and env.model.neq == 0)
    return ok, physics


def mpc_attempt(period, seed, max_controls):
    env = make_env(period); obs, info = env.reset(seed=42)
    low, high, search = action_search_bounds(env, protocol['action_bound'])
    with np.load(R / 'inputs/previous_recovery/prior_transitions.npz') as a:
        frames = np.vstack((env.data.qpos[env.q_ids].copy(), a['qpos'][:, env.q_ids]))
    prior = PrimitiveBasis.from_trajectories(frames, provenance=dict(source=str(R / 'inputs/previous_recovery/prior_transitions.npz'), sha256=sha256(R / 'inputs/previous_recovery/prior_transitions.npz'), purpose='measured 40 mm diagnostic PCA prior; not a SAC primitive', successful_policy=False))
    cem = CEMConfig(**dict(protocol['cem'], control_seconds=period, timestep_seconds=period, horizon_seconds=5 * period, knots=6))
    observer = PhysicsObserver(env); planner = ProjectedReferenceCEM(prior, low, high, cem, seed); planner.execution_start = env.data.ctrl.copy(); planner.reset(env.data.ctrl.copy())
    planner.exploration_floor = planner.exploration_floor * FLOOR_SCALE; planner.refinement_floor = planner.refinement_floor * FLOOR_SCALE  # proposal spread per planner step, scaled with the period
    evaluator = NativeMujocoRolloutEvaluator(env, timestep_seconds=period, config=RolloutCostConfig(**protocol['cost']), require_native=True, cache_directory=OUT / 'native_cache')
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
        rep = dict(period=period, seed=seed, status=status, controls=len(rows['actions']), strict_force_success=ok, failure_reason=info.get('failure_reason'), final_force_qualified_angle=observer.force_qualified_angle, substeps=env.substeps, horizon_controls=env.config.horizon_steps, env_overrides=dict(ENV_OVERRIDES), planning_seconds_total=float(sum(planning)), wall_seconds=time.monotonic() - started, floor_scale=FLOOR_SCALE)
    finally:
        observer.close(); evaluator.close(); env.close()
    return rep, {k: np.asarray(v) for k, v in rows.items()}


def replay_tolerance(period, actions, j, radius, direction, hold_seconds=2.):
    """Perturb control j along a ray fixed by (period, j, direction) (the same ray for every radius, so tolerance is monotone in r;
    r = 0 is the exact replay), replay the remaining recorded controls, then hold the last one for hold_seconds (in controls: round(hold/period))."""
    rng = np.random.default_rng(np.random.SeedSequence([90000, int(round(period * 1000)), j, direction])); u = rng.standard_normal(16); u /= np.linalg.norm(u)
    env = make_env(period); obs, info = env.reset(seed=42); observer = PhysicsObserver(env); n = len(actions); term = trunc = False; hold = max(1, int(round(hold_seconds / period)))
    try:
        i = 0; extra = 0
        while not (term or trunc):
            if i == j: a = np.clip(actions[j] + radius * u, -1, 1).astype(np.float32)
            elif i < n: a = actions[i]
            elif extra < hold: a = actions[-1]; extra += 1
            else: break
            obs, reward, term, trunc, info = env.step(a); i += 1
        ok, _ = strict_outcome(env, observer, info, term, trunc); return ok
    finally:
        observer.close(); env.close()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('period', type=float); ap.add_argument('seed_from', type=int); ap.add_argument('seed_to', type=int); ap.add_argument('--max-successes', type=int, default=3); ap.add_argument('--radii', default='0,0.0003,0.001,0.003,0.01,0.03,0.1'); ap.add_argument('--dirs', type=int, default=3); ap.add_argument('--max-controls', type=int, default=None); ap.add_argument('--floor-scale', type=float, default=1.); ap.add_argument('--shape', default=None, choices=[None, 'hexagonal', 'circular']); ap.add_argument('--diameter', type=float, default=None); ap.add_argument('--out', default=None)
    a = ap.parse_args(); global FLOOR_SCALE, OUT; FLOOR_SCALE = a.floor_scale
    if a.shape: ENV_OVERRIDES['object_shape'] = a.shape
    if a.diameter: ENV_OVERRIDES['object_diameter'] = float(a.diameter)
    if a.out: OUT = Path(a.out).resolve(); OUT.mkdir(parents=True, exist_ok=True); (OUT / 'native_cache').exists() or __import__('shutil').copytree(HERE_CACHE, OUT / 'native_cache')
    P = a.period; radii = [float(x) for x in a.radii.split(',')]; max_controls = a.max_controls if a.max_controls is not None else 40  # the archived collector's budget, a count of control decisions
    (OUT / 'episodes').mkdir(exist_ok=True); tag = f'P{int(round(P * 1000)):04d}ms' + (f'_fs{a.floor_scale:g}' if a.floor_scale != 1 else '') + (f"_{a.shape}" if a.shape else '') + (f'_d{int(round(a.diameter * 1000))}mm' if a.diameter else ''); results = []
    log = (OUT / f'attempts_{tag}.jsonl').open('a')
    for seed in range(a.seed_from, a.seed_to + 1):
        if sum(r['strict_force_success'] for r in results) >= a.max_successes: break
        rep, rows = mpc_attempt(P, seed, max_controls); print(json.dumps({k: rep[k] for k in ('period', 'seed', 'status', 'controls', 'strict_force_success', 'failure_reason', 'final_force_qualified_angle', 'wall_seconds')}), flush=True)
        if rep['strict_force_success']:
            np.savez_compressed(OUT / 'episodes' / f'{tag}_seed{seed}.npz', **rows)
            tol = {}
            for j in range(rep['controls']):
                tol[j] = {str(r_): sum(replay_tolerance(P, rows['actions'], j, r_, d) for d in range(a.dirs)) for r_ in radii}
            rep['open_loop_tolerance'] = tol; rep['tolerance_dirs'] = a.dirs; rep['tolerance_hold_seconds'] = 2.
            print(f'  open-loop tolerance (successes of {a.dirs}) per control j and radius: ' + ' | '.join(f"j{j}: " + ' '.join(f"{r_}:{tol[j][str(r_)]}" for r_ in radii) for j in tol), flush=True)
        results.append(rep); log.write(json.dumps(rep) + '\n'); log.flush()
    log.close(); (OUT / f'probe_{tag}_seeds{a.seed_from}_{a.seed_to}.json').write_text(json.dumps(results, indent=1) + '\n')


if __name__ == '__main__':
    main()
