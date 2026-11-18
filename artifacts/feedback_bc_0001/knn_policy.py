"""Non-parametric feedback policy: k-nearest-neighbour regression (distance-weighted) over the spine rows and the training
branches' correction rows, evaluated closed-loop like train.py. Usage: knn_policy.py <manifest result.json> <k> [--scale]"""
import importlib.util, json, sys, time
from pathlib import Path
A = Path('/home/autel001/code/mpc_dex/artifacts'); STAGE = A / 'bootstrap_variant_0087_ms_spine193_tube_from84'; F = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('knn_runner', STAGE / 'run.py'); r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
np = r.np; protocol = json.loads((STAGE / 'protocol.json').read_text()); ec = r.EnvConfig(**json.loads((STAGE / 'config.json').read_text())['environment'])
man = json.loads(Path(sys.argv[1]).read_text())['manifest']; K = int(sys.argv[2]); scale = '--scale' in sys.argv
spine_d = A / 'mpc_cold_demonstrations_0001/episodes/seed_193'; sp = np.load(spine_d / 'transitions.npz'); SA, SO = sp['actions'], sp['obs']
with np.load(spine_d / 'physics.npz') as z: SP = z['samples']
def load(x):
    d = A / 'mpc_spine_tube_0001' / x['collection'] / 'episodes' / x['name']
    with np.load(d / 'transitions.npz') as z: src = {k: z[k].copy() for k in z.files}
    with np.load(d / 'physics.npz') as z: phys = z['samples'].copy()
    return src, phys
def corr(src):
    o = src['control_origin']; return int(np.flatnonzero((o == 'perturbed') | (o == 'policy'))[-1]) + 1
O, Aa = [SO], [SA]
for x in man['train']:
    src, _ = load(x); s0 = corr(src); O.append(src['obs'][s0:]); Aa.append(src['actions'][s0:])
O = np.concatenate(O).astype(np.float64); Aa = np.concatenate(Aa).astype(np.float64)
sd = O.std(0) + 1e-6 if scale else np.ones(60); On = O / sd
def predict(obs):
    d = np.linalg.norm(On - obs.astype(np.float64) / sd, axis=1); idx = np.argsort(d)[:K]; w = 1. / (d[idx] + 1e-3); return (Aa[idx] * w[:, None]).sum(0) / w.sum()
hobs, hact = [], []
for x in man['heldout']:
    src, _ = load(x); s0 = corr(src); hobs.append(src['obs'][s0:]); hact.append(src['actions'][s0:])
hobs = np.concatenate(hobs); hact = np.concatenate(hact); err = np.linalg.norm(np.stack([predict(o) for o in hobs]) - hact, axis=1)
print(f'kNN k={K} scale={scale}: training rows {len(O)}; held-out regression L2 median {np.median(err):.4f} p90 {np.quantile(err, .9):.4f}', flush=True)
def rec(a): return dict(action=np.clip(np.asarray(a, np.float32), -1, 1), log_prob=np.nan, raw_mean=np.full(16, np.nan, np.float32), latent_std=np.full(16, np.nan, np.float32), density_is_stochastic_sample=False, learner_bc_updates=-1, learner_pretrain_updates=-1, learner_online_updates=-1)
def run(prefix, src, phys):
    r.seed_all(42); env = r.environment_for_policy(ec, protocol['policy_interface']); ep = None
    try:
        ep = r.Episode(env, prefix, src, phys, protocol['physical_reset_seed'], 'knn', 42, protocol['raw_reward_divisor']); t = 0
        while not (ep.term or ep.trunc) and t < 60: ep.advance(rec(predict(ep.obs)), policy=False); t += 1
        rep = ep.report(); return bool(rep['finger_supported_rotation_success']), rep['policy_suffix_controls'], rep['environment_failure_reason']
    finally:
        if ep is not None: ep.close()
        env.close()
spine_src = {k: sp[k] for k in sp.files}
print('cold:', run(0, spine_src, SP), '| spine prefixes k=1..13:', [run(k, spine_src, SP)[0] for k in range(1, len(SA))], flush=True)
for label, items in (('training', man['train']), ('held-out', man['heldout'])):
    res = []
    for x in items[:30]:
        src, phys = load(x); s0 = corr(src); rep = json.loads((A / 'mpc_spine_tube_0001' / x['collection'] / 'episodes' / x['name'] / 'report.json').read_text()); res.append((rep['perturbed_control_index'], run(s0, src, phys)[0]))
    early = [o for j, o in res if j <= 9]; print(f'{label} branches: handoffs {sum(o for _, o in res)}/{len(res)}; j<=9: {sum(early)}/{len(early)}; by j', {j: f"{sum(o for jj, o in res if jj == j)}/{sum(1 for jj, _ in res if jj == j)}" for j in sorted({j for j, _ in res})}, flush=True)
# basin test: perturb spine control j by r in 3 fresh directions, then the kNN policy continues
rng = np.random.default_rng(123); print('basins (kNN continues after an r-perturbed control j):', flush=True)
for j in range(len(SA) - 1):
    cells = []
    for rad in (0.003, 0.01, 0.03):
        ok = 0
        for _ in range(3):
            u = rng.standard_normal(16); u /= np.linalg.norm(u); first = np.clip(SA[j] + rad * u, -1, 1).astype(np.float32)
            r.seed_all(42); env = r.environment_for_policy(ec, protocol['policy_interface']); ep = None
            try:
                ep = r.Episode(env, j, spine_src, SP, protocol['physical_reset_seed'], 'knn_basin', 42, protocol['raw_reward_divisor']); ep.advance(rec(first), policy=False); t = 0
                while not (ep.term or ep.trunc) and t < 60: ep.advance(rec(predict(ep.obs)), policy=False); t += 1
                ok += int(ep.report()['finger_supported_rotation_success'])
            finally:
                if ep is not None: ep.close()
                env.close()
        cells.append(f'r{rad}:{ok}/3')
    print(f'  j={j}: ' + ' '.join(cells), flush=True)
