"""Existence test for an imprecise closed-loop controller: the archived native MPC (planner sees the true state, replans every 0.1 s) on the 40 mm cylinder from the cold reset, but every executed control is the planned one plus Gaussian noise of std sigma per dimension (clipped to the search bound). Success is the strict force-qualified criterion. Usage: mpc_action_noise.py <sigma> <seed_from> <seed_to> [max_controls]  -> appends to attempts_sigma<sigma>.jsonl"""
import json, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; CP = HERE.parent / 'control_period_0001'; R = HERE.parent / 'mpc_prefix6_handoff_recovery_0001'
sys.path.insert(0, str(CP)); sys.path.insert(0, str(R)); sys.path.insert(0, '/home/autel001/code/mpc_dex')
import numpy as np, probe
probe.ENV_OVERRIDES.update(object_shape='circular')
from probe import make_env, strict_outcome, protocol, FIELDS
from mpc_dex.mpc import CEMConfig, NoFeasiblePlan
from mpc_dex.primitives import PrimitiveBasis
from mpc_dex.rollouts import RolloutCostConfig
from mpc_dex.native_rollouts import NativeMujocoRolloutEvaluator
from mpc_dex.contracts import sha256
from scripts.probe_mpc import action_search_bounds
from scripts.audit_policy import PhysicsObserver
from recovery_recipe import ProjectedReferenceCEM, integration_state
sigma, s0, s1 = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]); max_controls = int(sys.argv[4]) if len(sys.argv) > 4 else 40; period = 0.1
def attempt(seed):
    env = make_env(period); obs, info = env.reset(seed=42); noise = np.random.default_rng(900000 + seed)
    low, high, search = action_search_bounds(env, protocol['action_bound'])
    with np.load(R / 'inputs/previous_recovery/prior_transitions.npz') as a: frames = np.vstack((env.data.qpos[env.q_ids].copy(), a['qpos'][:, env.q_ids]))
    prior = PrimitiveBasis.from_trajectories(frames, provenance=dict(source=str(R / 'inputs/previous_recovery/prior_transitions.npz'), sha256=sha256(R / 'inputs/previous_recovery/prior_transitions.npz'), purpose='measured 40 mm diagnostic PCA prior; not a SAC primitive', successful_policy=False))
    cem = CEMConfig(**dict(protocol['cem'], control_seconds=period, timestep_seconds=period, horizon_seconds=5 * period, knots=6))
    observer = PhysicsObserver(env); planner = ProjectedReferenceCEM(prior, low, high, cem, seed); planner.execution_start = env.data.ctrl.copy(); planner.reset(env.data.ctrl.copy())
    evaluator = NativeMujocoRolloutEvaluator(env, timestep_seconds=period, config=RolloutCostConfig(**protocol['cost']), require_native=True, cache_directory=HERE / 'native_cache')
    started = time.monotonic(); status = 'budget'; planning = []; term = trunc = False; n = 0; dev = []
    try:
        for _ in range(max_controls):
            planner.execution_start = env.data.ctrl.copy(); t0 = time.monotonic()
            try: target, diag = planner.plan(evaluator)
            except NoFeasiblePlan: status = 'no_feasible_plan'; break
            planning.append(time.monotonic() - t0)
            planned = env.to_action(target); executed = np.clip(planned + noise.normal(0., sigma, planned.shape), -search['effective_bound'], search['effective_bound']).astype(planned.dtype)
            dev.append(float(np.linalg.norm(executed - planned)))
            obs, reward, term, trunc, info = env.step(executed); n += 1
            if term or trunc: status = 'terminated' if term else 'truncated_by_horizon'; break
        ok, physics = strict_outcome(env, observer, info, term, trunc)
        return dict(sigma=sigma, seed=seed, status=status, controls=n, strict_force_success=bool(ok), environment_success=bool(info.get('success')), failure_reason=info.get('failure_reason'), final_force_qualified_angle=float(observer.force_qualified_angle), executed_minus_planned_l2_mean=float(np.mean(dev)) if dev else 0., planning_seconds_total=float(sum(planning)), wall_seconds=time.monotonic() - started, object_shape=env.config.object_shape, period=period)
    finally:
        observer.close(); evaluator.close(); env.close()
for seed in range(s0, s1 + 1):
    rep = attempt(seed); print(json.dumps(rep), flush=True)
    with open(HERE / f'attempts_sigma{sigma:g}.jsonl', 'a') as f: f.write(json.dumps(rep) + '\n')
