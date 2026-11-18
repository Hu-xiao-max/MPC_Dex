"""Per-control trace of one MPC attempt at a given period: planner cost / valid candidates, action step, angles, contacts.
Usage: diag_trace.py <period> <seed> <max_controls>"""
import json, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np
from probe import make_env, strict_outcome, protocol, R
from mpc_dex.mpc import CEMConfig, NoFeasiblePlan
from mpc_dex.primitives import PrimitiveBasis
from mpc_dex.rollouts import RolloutCostConfig
from mpc_dex.native_rollouts import NativeMujocoRolloutEvaluator
from mpc_dex.contracts import sha256
from scripts.probe_mpc import action_search_bounds
from scripts.audit_policy import PhysicsObserver, PHYSICS_COLUMNS
from recovery_recipe import ProjectedReferenceCEM, integration_state
P, seed, maxc = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]); FS = float(sys.argv[4]) if len(sys.argv) > 4 else 1.
env = make_env(P); obs, info = env.reset(seed=42); low, high, search = action_search_bounds(env, protocol['action_bound'])
with np.load(R / 'inputs/previous_recovery/prior_transitions.npz') as a: frames = np.vstack((env.data.qpos[env.q_ids].copy(), a['qpos'][:, env.q_ids]))
prior = PrimitiveBasis.from_trajectories(frames, provenance=dict(source='x', sha256='x', purpose='diag', successful_policy=False))
cem = CEMConfig(**dict(protocol['cem'], control_seconds=P, timestep_seconds=P, horizon_seconds=5 * P, knots=6)); observer = PhysicsObserver(env)
planner = ProjectedReferenceCEM(prior, low, high, cem, seed); planner.execution_start = env.data.ctrl.copy(); planner.reset(env.data.ctrl.copy()); planner.exploration_floor = planner.exploration_floor * FS; planner.refinement_floor = planner.refinement_floor * FS
ev = NativeMujocoRolloutEvaluator(env, timestep_seconds=P, config=RolloutCostConfig(**protocol['cost']), require_native=True, cache_directory=R / 'native_cache')
print(f'P={P} floor_scale={FS} substeps={env.substeps} floors: exploration residual {planner.exploration_floor[-16:].mean():.2f} refinement {planner.refinement_floor[-16:].mean():.2f} coeff spread {planner.exploration_floor[:prior.components].round(2).tolist()}')
print('ctrl | best cost | valid/1050 | |a_t - a_{t-1}| | raw angle | force-qual angle | fingers | obj pos z | plan s')
prev_a = env.to_action(env.data.ctrl.copy()); t0 = time.monotonic()
try:
    for t in range(maxc):
        planner.execution_start = env.data.ctrl.copy(); s0 = time.monotonic()
        try: target, diag = planner.plan(ev)
        except NoFeasiblePlan: print('no feasible plan'); break
        a = env.to_action(target); obs, r_, term, trunc, info = env.step(a)
        print(f'{t:4d} | {diag["best_cost"]:9.3f} | {diag["valid_candidates"]:4d} | {np.linalg.norm(a - prev_a):.3f} | {observer.raw_angle:6.3f} | {observer.force_qualified_angle:6.3f} | {env.finger_contacts()} | {env.object_position()[2]:.4f} | {time.monotonic() - s0:.1f}', flush=True); prev_a = a
        if term or trunc: print('boundary', info.get('failure_reason'), info.get('success')); break
    ok, _ = strict_outcome(env, observer, info, term, trunc); print('strict success', ok, 'wall', round(time.monotonic() - t0, 1))
finally: observer.close(); ev.close(); env.close()
