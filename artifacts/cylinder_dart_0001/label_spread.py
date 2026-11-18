"""Label spread of the free re-planning planner along its own closed-loop cylinder trajectory from the cold reset.
Usage: label_spread.py <planner_seed> <sigma> [fresh_planners=3]
At every step: the persistent (warm-started) planner's control is the label that is executed (plus Gaussian noise of std sigma per
dimension, clipped); before executing, `fresh_planners` planners with other CEM seeds and no warm start are run once at the same
state. Records, per step, the L2 distance of each fresh label to the persistent one and between fresh labels. -> label_spread_s<seed>_sigma<sigma>.json"""
import json, sys, time
import numpy as np
from common import *
seed, sigma = int(sys.argv[1]), float(sys.argv[2]); K = int(sys.argv[3]) if len(sys.argv) > 3 else 3; HERE = Path(__file__).resolve().parent
env = make_env(PERIOD); obs, info = env.reset(seed=42); noise = np.random.default_rng(910000 + seed)
planner, evaluator, search, cem = planner_for(env, seed, HERE / 'native_cache'); observer = PhysicsObserver(env)
steps = []; term = trunc = False; started = time.monotonic()
try:
    for t in range(40):
        label, diag = plan_once(planner, evaluator, env)
        if label is None: steps.append(dict(t=t, status='no_feasible_plan')); break
        fresh = []
        for i in range(K):
            fp, fe, _, _ = planner_for(env, 7000 + 10 * t + i, HERE / 'native_cache')
            fl, _ = plan_once(fp, fe, env); fe.close(); fresh.append(None if fl is None else fl)
        valid = [f for f in fresh if f is not None]
        rec = dict(t=t, fresh_feasible=len(valid), fresh_to_persistent_l2=[float(np.linalg.norm(f - label)) for f in valid], fresh_pairwise_l2=[float(np.linalg.norm(valid[i] - valid[j])) for i in range(len(valid)) for j in range(i + 1, len(valid))], label_norm=float(np.linalg.norm(label)))
        executed = np.clip(label + noise.normal(0., sigma, label.shape), -search['effective_bound'], search['effective_bound']).astype(label.dtype)
        obs, reward, term, trunc, info = env.step(executed); rec['executed_minus_label_l2'] = float(np.linalg.norm(executed - label)); steps.append(rec)
        print(json.dumps(rec), flush=True)
        if term or trunc: break
    ok, physics = strict_outcome(env, observer, info, term, trunc)
    out = dict(seed=seed, sigma=sigma, fresh_planners=K, strict_force_success=bool(ok), controls=len([s for s in steps if 'label_norm' in s]), final_force_qualified_angle=float(observer.force_qualified_angle), failure_reason=info.get('failure_reason'), steps=steps, wall_seconds=time.monotonic() - started)
finally:
    observer.close(); evaluator.close(); env.close()
out['horizon_controls'] = HORIZON_CONTROLS; tag = '' if HORIZON_CONTROLS == 5 else f'_h{HORIZON_CONTROLS}'; json.dump(out, open(HERE / f'label_spread_s{seed}_sigma{sigma:g}{tag}.json', 'w'), indent=1); print('done', out['strict_force_success'], out['controls'], round(out['wall_seconds']))
