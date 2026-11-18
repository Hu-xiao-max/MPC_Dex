"""Feasibility probe: corrected native MPC with the diagnostic prior from cold reset (prefix 0), several CEM seeds.
Reuses the archived recovery stage's frozen source and recipe read-only; nothing here is a demonstration export."""
import importlib.util, json, sys, time
from pathlib import Path
R = Path('/home/autel001/code/mpc_dex/artifacts/mpc_prefix6_handoff_recovery_0001')
sys.path.insert(0, str(R / 'source')); sys.path.insert(0, str(R))
import numpy as np
from mpc_dex.env import AllegroRotationEnv, EnvConfig
from mpc_dex.mpc import CEMConfig, NoFeasiblePlan
from mpc_dex.primitives import PrimitiveBasis
from mpc_dex.rollouts import RolloutCostConfig
from mpc_dex.native_rollouts import NativeMujocoRolloutEvaluator
from scripts.probe_mpc import action_search_bounds
from scripts.audit_policy import PhysicsObserver
from recovery_recipe import ProjectedReferenceCEM, integration_state
OUT = Path(__file__).resolve().parent
protocol = json.loads((R / 'protocol.json').read_text())
ec = EnvConfig(**json.loads((R / 'config.json').read_text())['environment'])
seeds = [int(s) for s in sys.argv[1:]] or [42, 43, 44]
results = []
for seed in seeds:
    env = AllegroRotationEnv(ec); obs, info = env.reset(seed=42)
    low, high, search = action_search_bounds(env, protocol['action_bound'])
    with np.load(R / 'inputs/previous_recovery/prior_transitions.npz') as a:
        frames = np.vstack((env.data.qpos[env.q_ids].copy(), a['qpos'][:, env.q_ids]))
    prior = PrimitiveBasis.from_trajectories(frames, provenance=dict(source='diagnostic prior (feasibility probe)', successful_policy=False))
    observer = PhysicsObserver(env)
    planner = ProjectedReferenceCEM(prior, low, high, CEMConfig(**protocol['cem']), seed)
    planner.execution_start = env.data.ctrl.copy(); planner.reset(env.data.ctrl.copy())
    evaluator = NativeMujocoRolloutEvaluator(env, timestep_seconds=protocol['cem']['timestep_seconds'], config=RolloutCostConfig(**protocol['cost']), require_native=True, cache_directory=R / 'native_cache')
    started = time.monotonic(); controls = 0; status = 'budget'; actions = []
    try:
        for index in range(1, 41):
            planner.execution_start = env.data.ctrl.copy()
            try:
                target, diagnostics = planner.plan(evaluator)
            except NoFeasiblePlan:
                status = 'no_feasible_plan'; break
            action = env.to_action(target); actions.append(action.copy())
            obs, reward, term, trunc, info = env.step(action); controls += 1
            if term or trunc:
                status = 'success' if info['success'] and not info['drop'] else f"terminal:{info['failure_reason']}"; break
        rep = dict(seed=seed, status=status, controls=controls, seconds=round(time.monotonic() - started, 1),
                   force_qualified_angle=observer.force_qualified_angle, palm=bool(observer.palm_support_violation), drop=bool(observer.independent_drop))
    finally:
        observer.close(); evaluator.close(); env.close()
    np.savez_compressed(OUT / f'seed_{seed}_actions.npz', actions=np.asarray(actions, np.float32))
    results.append(rep); print(json.dumps(rep), flush=True)
(OUT / f'results_{"_".join(map(str, seeds))}.json').write_text(json.dumps(results, indent=1) + '\n')
