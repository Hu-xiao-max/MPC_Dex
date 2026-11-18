"""Is the tube's feedback law locally linear and does it work closed-loop? Per spine time index t, fit delta_a = K_t delta_s by ridge
least squares over every branch row at index t (delta = branch row - spine row), holding out one perturbation direction, then run
the time-indexed law a_t = clip(spine_a_t + K_t (s - spine_s_t)) closed-loop: spine prefixes, perturbed-control basins
(the law continues after an r-perturbed control at j), and corrected-state handoffs of held-out branches.
Usage: linear_law.py <collection> <holdout_direction> [ridge list]"""
import importlib.util, json, sys, time
from pathlib import Path
A = Path('/home/autel001/code/mpc_dex/artifacts'); STAGE = A / 'bootstrap_variant_0087_ms_spine193_tube_from84'; HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('ll_runner', STAGE / 'run.py'); r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
np = r.np; protocol = json.loads((STAGE / 'protocol.json').read_text()); ec = r.EnvConfig(**json.loads((STAGE / 'config.json').read_text())['environment'])
coll, hold = sys.argv[1], int(sys.argv[2]); ridges = [float(x) for x in (sys.argv[3].split(',') if len(sys.argv) > 3 else ['1e-4', '1e-3', '1e-2', '1e-1', '1'])]
E = A / 'mpc_spine_tube_0001' / coll / 'episodes'
spine = np.load(A / 'mpc_cold_demonstrations_0001/episodes/seed_193/transitions.npz'); SA, SO = spine['actions'].astype(np.float64), spine['obs'].astype(np.float64); n = len(SA)
with np.load(A / 'mpc_cold_demonstrations_0001/episodes/seed_193/physics.npz') as z: SP = z['samples']
branches = []
for d in sorted(E.iterdir()):
    if not (d / 'report.json').exists(): continue
    rep = json.loads((d / 'report.json').read_text())
    if not rep.get('mpc_controls', 0): continue
    with np.load(d / 'transitions.npz') as z: src = {k: z[k].copy() for k in z.files}
    with np.load(d / 'physics.npz') as z: phys = z['samples'].copy()
    branches.append(dict(name=d.name, j=rep['perturbed_control_index'], dirn=rep['perturbation_direction'], r=rep['perturbation_radius'], src=src, phys=phys))
print(f'{len(branches)} branches; holdout direction {hold}: {sum(b["dirn"] == hold for b in branches)}')
# per-t datasets over correction rows (t > j), restricted to t < n (spine has n rows)
data = {t: dict(train=([], []), test=([], [])) for t in range(1, n)}
for b in branches:
    acts, obs = b['src']['actions'].astype(np.float64), b['src']['obs'].astype(np.float64)
    for t in range(b['j'] + 1, min(len(acts), n)):
        ds, da = obs[t] - SO[t], acts[t] - SA[t]; key = 'test' if b['dirn'] == hold else 'train'; data[t][key][0].append(ds); data[t][key][1].append(da)
def fit(S, Aa, lam):
    S = np.asarray(S); Aa = np.asarray(Aa)
    return Aa.T @ S @ np.linalg.inv(S.T @ S + lam * np.eye(S.shape[1]))  # (16, 60)
print('t | n_train n_test | |delta_a| median (signal) | ridge -> held-out residual median (relative to signal) ; in-sample residual')
best = {}
for t in range(1, n):
    S, Aa = data[t]['train']; St, At = data[t]['test']
    if len(S) < 4 or len(St) == 0: print(f'{t:2d} | {len(S):3d} {len(St):3d} | insufficient'); continue
    sig = np.median(np.linalg.norm(At, axis=1)); line = []
    for lam in ridges:
        K = fit(S, Aa, lam); res = np.median(np.linalg.norm(np.asarray(At) - np.asarray(St) @ K.T, axis=1)); ins = np.median(np.linalg.norm(np.asarray(Aa) - np.asarray(S) @ K.T, axis=1)); line.append((lam, res, ins))
    lam_best = min(line, key=lambda x: x[1]); best[t] = lam_best[0]
    print(f'{t:2d} | {len(S):3d} {len(St):3d} | {sig:.4f} | ' + ' '.join(f'{lam:g}->{res:.4f}({res / sig:.2f});{ins:.4f}' for lam, res, ins in line), flush=True)
# closed loop with K fitted on all directions at the best ridge per t (fallback: nearest t with a fit)
K = {}
for t in range(1, n):
    S = data[t]['train'][0] + data[t]['test'][0]; Aa = data[t]['train'][1] + data[t]['test'][1]
    if len(S) >= 4: K[t] = fit(S, Aa, best.get(t, 1e-2))
def law(obs, t):
    tt = min(t, n - 1); kk = K[tt] if tt in K else (K[max(k for k in K if k <= tt)] if any(k <= tt for k in K) else None)
    a = SA[tt] + (kk @ (obs.astype(np.float64) - SO[tt]) if kk is not None else 0)
    return np.clip(a, -1, 1).astype(np.float32)
def rec(a): return dict(action=np.asarray(a, np.float32), log_prob=np.nan, raw_mean=np.full(16, np.nan, np.float32), latent_std=np.full(16, np.nan, np.float32), density_is_stochastic_sample=False, learner_bc_updates=-1, learner_pretrain_updates=-1, learner_online_updates=-1)
def run(prefix, src, phys, first=None):
    r.seed_all(42); env = r.environment_for_policy(ec, protocol['policy_interface']); ep = None
    try:
        ep = r.Episode(env, prefix, src, phys, protocol['physical_reset_seed'], 'linear_law', 42, protocol['raw_reward_divisor']); t = prefix
        if first is not None: ep.advance(rec(first), policy=False); t += 1
        while not (ep.term or ep.trunc) and t < 60: ep.advance(rec(law(ep.obs, t)), policy=False); t += 1
        rep = ep.report(); return bool(rep['finger_supported_rotation_success']), rep['environment_failure_reason'], round(rep['final_force_qualified_angle'], 2)
    finally:
        if ep is not None: ep.close()
        env.close()
spine_src = {k: spine[k] for k in spine.files}
print('closed loop, spine prefixes k=0..13 (deterministic law):', [run(k, spine_src, SP)[0] for k in range(n)], flush=True)
print('basins: perturb control j by r (3 fresh directions), then the law continues')
rng = np.random.default_rng(123)
for j in range(n - 1):
    cells = []
    for rad in (0.003, 0.01, 0.03):
        ok = 0
        for _ in range(3):
            u = rng.standard_normal(16); u /= np.linalg.norm(u); ok += run(j, spine_src, SP, first=np.clip(SA[j] + rad * u, -1, 1))[0]
        cells.append(f'r{rad}:{ok}/3')
    print(f'  j={j}: ' + ' '.join(cells), flush=True)
held = [b for b in branches if b['dirn'] == hold]; res = []
for b in held:
    origin = b['src']['control_origin']; s0 = int(np.flatnonzero(origin == 'perturbed')[-1]) + 1; res.append((b['j'], run(s0, b['src'], b['phys'])[0]))
print('held-out branches, handoff at the first corrected state:', sum(o for _, o in res), '/', len(res), '| j<=9:', sum(o for j, o in res if j <= 9), '/', sum(1 for j, _ in res if j <= 9), '| by j', {j: f"{sum(o for jj, o in res if jj == j)}/{sum(1 for jj, _ in res if jj == j)}" for j in sorted({j for j, _ in res})})
json.dump(dict(best_ridge=best, K={str(t): k.tolist() for t, k in K.items()}), open(HERE / f'linear_law_{coll}_hold{hold}.json', 'w'))
