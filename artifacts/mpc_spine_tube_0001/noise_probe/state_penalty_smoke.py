import sys; sys.argv = ['x']; sys.path.insert(0, '/home/autel001/code/mpc_dex/artifacts/mpc_spine_tube_0001')
import collect_tube as C, numpy as np, json
src, phys = C.L.load_demo('seed_193'); SA, SO = src['actions'], src['obs']
for w in (100., 1000.):
    for salt in (1000005, 1000006):
        rep, rows, init, physics, origins = C.tube_branch(src, phys, 6, 0.003, 0, 25, mode=dict(name='tracking', floor_scale=0.005, horizon=1.5, track_weight=10000., state_weight=w), seed_salt=salt)
        acts, obs = rows['actions'], rows['obs']; j = 6
        da = [np.linalg.norm(acts[t] - SA[t]) for t in range(j + 1, min(len(acts), len(SA)))]; do = [np.linalg.norm(obs[t] - SO[t]) for t in range(j + 1, min(len(obs), len(SO)))]
        print(json.dumps(dict(state_weight=w, salt=salt, status=rep['status'], success=rep['strict_force_success'], controls=rep['controls'], wall=round(rep['wall_seconds']), action_dev=[round(float(x), 3) for x in da], obs_dev=[round(float(x), 2) for x in do])), flush=True)
