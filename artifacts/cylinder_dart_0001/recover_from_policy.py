"""Can the re-planning planner recover from the states the frozen policy reaches from the cold reset?
Usage: recover_from_policy.py <variant> <checkpoint> <b,b,...> [max_controls=40]
The policy (deterministic, stage inference contract) is rolled out from the cold reset in the collector environment; its controls are
recorded. For each b: fresh cold reset, the first b policy controls are executed, then the persistent planner takes over until the
boundary or max_controls. Records the strict outcome, the planner's first label at the policy-visited state and its L2 distance to
the policy's own control there. -> recover_<variant idx>_<checkpoint>.json"""
import json, sys, time
import numpy as np
from common import *
variant, ckpt = sys.argv[1], sys.argv[2]; bs = [int(x) for x in sys.argv[3].split(',')]; max_controls = int(sys.argv[4]) if len(sys.argv) > 4 else 40; HERE = Path(__file__).resolve().parent
agent, act, r = load_policy(variant, ckpt)
env = make_env(PERIOD); obs, info = env.reset(seed=42); pol = []; term = trunc = False
while not (term or trunc) and len(pol) < 40:
    a = np.asarray(act(obs, True), np.float32); pol.append(a); obs, reward, term, trunc, info = env.step(a)
print('policy cold rollout:', len(pol), 'controls, failure', info.get('failure_reason'), flush=True); env.close()
out = dict(variant=variant, checkpoint=ckpt, policy_controls=len(pol), policy_failure=info.get('failure_reason'), branches=[])
for b in bs:
    if b > len(pol): break
    env = make_env(PERIOD); obs, info = env.reset(seed=42); observer = PhysicsObserver(env); term = trunc = False; started = time.monotonic()
    for a in pol[:b]: obs, reward, term, trunc, info = env.step(a)
    planner, evaluator, search, cem = planner_for(env, 42 + b, HERE / 'native_cache'); n = 0; first = None; status = 'budget'
    try:
        if term or trunc: status = 'terminated_before_handoff'
        else:
            for _ in range(max_controls):
                label, diag = plan_once(planner, evaluator, env)
                if label is None: status = 'no_feasible_plan'; break
                if first is None: first = label
                obs, reward, term, trunc, info = env.step(label); n += 1
                if term or trunc: status = 'terminated' if term else 'truncated_by_horizon'; break
        ok, physics = strict_outcome(env, observer, info, term, trunc)
        rec = dict(b=b, status=status, mpc_controls=n, strict_force_success=bool(ok), failure_reason=info.get('failure_reason'), final_force_qualified_angle=float(observer.force_qualified_angle), policy_control_at_state_vs_mpc_label_l2=(None if first is None else float(np.linalg.norm(pol[b] - first))) if b < len(pol) else None, wall_seconds=time.monotonic() - started)
    finally:
        observer.close(); evaluator.close(); env.close()
    out['branches'].append(rec); print(json.dumps(rec), flush=True)
json.dump(out, open(HERE / f'recover_{variant[18:22]}_{ckpt}.json', 'w'), indent=1)
