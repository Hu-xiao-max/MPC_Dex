"""One DART rollout: the persistent warm-started planner (planner_seed) from the cold reset, executed controls = label + Gaussian noise
(std sigma per dimension, noise_seed), every step stored (obs, label, executed, qpos, qvel, ctrl targets, strict outcome at the end).
Usage: dart_rollout.py <planner_seed> <sigma> <noise_seed> [max_controls=40] -> rollouts/p<planner_seed>_sigma<sigma>_n<noise_seed>.npz (+ .json)"""
import json, sys, time
import numpy as np
from common import *
ps, sigma, ns = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]); max_controls = int(sys.argv[4]) if len(sys.argv) > 4 else 40; HERE = Path(__file__).resolve().parent; (HERE / 'rollouts').mkdir(exist_ok=True)
env = make_env(PERIOD); obs, info = env.reset(seed=42); noise = np.random.default_rng(920000 + 1000 * ns + ps)
planner, evaluator, search, cem = planner_for(env, ps, HERE / 'native_cache'); observer = PhysicsObserver(env)
rows = dict(obs=[], labels=[], actions=[], next_obs=[], rewards=[], terminated=[], truncated=[], qpos=[], qvel=[], targets=[]); term = trunc = False; status = 'budget'; started = time.monotonic(); planning = []
try:
    for t in range(max_controls):
        t0 = time.monotonic(); label, diag = plan_once(planner, evaluator, env); planning.append(time.monotonic() - t0)
        if label is None: status = 'no_feasible_plan'; break
        executed = np.clip(label + noise.normal(0., sigma, label.shape), -search['effective_bound'], search['effective_bound']).astype(np.float32)
        prev = obs.copy(); obs, reward, term, trunc, info = env.step(executed)
        for k, v in dict(obs=prev, labels=label.astype(np.float32), actions=executed, next_obs=obs.copy(), rewards=reward, terminated=term, truncated=trunc, qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(), targets=env.data.ctrl.copy()).items(): rows[k].append(v)
        if term or trunc: status = 'terminated' if term else 'truncated_by_horizon'; break
    ok, physics = strict_outcome(env, observer, info, term, trunc)
    rep = dict(planner_seed=ps, sigma=sigma, noise_seed=ns, status=status, controls=len(rows['labels']), strict_force_success=bool(ok), environment_success=bool(info.get('success')), failure_reason=info.get('failure_reason'), final_force_qualified_angle=float(observer.force_qualified_angle), planning_seconds_total=float(sum(planning)), wall_seconds=time.monotonic() - started, executed_minus_label_l2_mean=float(np.mean([np.linalg.norm(a - l) for a, l in zip(rows['actions'], rows['labels'])])) if rows['labels'] else 0.)
finally:
    observer.close(); evaluator.close(); env.close()
name = f'p{ps}_sigma{sigma:g}_n{ns}'; np.savez_compressed(HERE / 'rollouts' / f'{name}.npz', **{k: np.asarray(v) for k, v in rows.items()}, physics=np.asarray(physics, np.float64)); json.dump(rep, open(HERE / 'rollouts' / f'{name}.json', 'w'), indent=1); print(json.dumps(rep), flush=True)
