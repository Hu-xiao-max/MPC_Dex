"""Is the feedback law learnable by supervised regression alone? BC (mean only) from the 0084 actor on the spine rows plus every
tube correction row, then closed-loop tests: cold reset, spine prefixes, and the tube's first corrected states.
Usage: bc_tube_probe.py <steps> <lr> <tag>"""
import importlib.util, json, sys, time
from pathlib import Path
A = Path('/home/autel001/code/mpc_dex/artifacts'); D = A / 'bootstrap_variant_0087_ms_spine193_tube_from84'
spec = importlib.util.spec_from_file_location('bcprobe', D / 'run.py'); r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
np, torch = r.np, r.torch
torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32 = False
steps, lr, tag = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
protocol = json.loads((D / 'protocol.json').read_text()); ec = r.EnvConfig(**json.loads((D / 'config.json').read_text())['environment'])
sources = r.load_sources(protocol); demos = r.multisource_demonstrations(protocol, sources)
spine = [d for d in demos if d['name'] == 'seed_193'][0]; tube = [d for d in demos if d['name'].startswith('track_')]
obs = np.concatenate([spine['arrays']['obs']] + [d['arrays']['obs'] for d in tube]); act = np.concatenate([spine['arrays']['actions']] + [d['arrays']['actions'] for d in tube])
print('BC rows', len(obs), '(spine 14 + tube', len(obs) - 14, ')', flush=True)
agent = r.SACAgent.load(A / 'bootstrap_variant_0084_ms_spine193_sarsa_from82/checkpoints/online_4000/agent.pt', 'cuda')
for g in agent.bc_optimizer.param_groups: g['lr'] = lr
rng = np.random.default_rng(0); t0 = time.monotonic()
for i in range(steps):
    idx = rng.integers(len(obs), size=256); out = agent.bc_step(dict(obs=obs[idx], actions=act[idx]))
    if i % 1000 == 0 or i == steps - 1: print(f'step {i} bc_loss {out["bc_loss"]:.5f}', flush=True)
with torch.no_grad():
    m, _ = agent.actor(torch.as_tensor(obs, device='cuda')); fit = (m.tanh().cpu().numpy() - act); print('fit L2 per row: median', np.median(np.linalg.norm(fit, axis=1)).round(4), 'max', np.linalg.norm(fit, axis=1).max().round(3), '| spine rows', np.linalg.norm(fit[:14], axis=1).round(3).tolist())
def run(prefix, src, phys, det, seed):
    r.seed_all(seed); env = r.environment_for_policy(ec, protocol['policy_interface']); ep = None
    try:
        ep = r.Episode(env, prefix, src, phys, protocol['physical_reset_seed'], 'bc_probe', seed, protocol['raw_reward_divisor'])
        while not (ep.term or ep.trunc): ep.advance(r.command(agent, ep.obs, det))
        rep = ep.report(); return bool(rep['finger_supported_rotation_success']), rep['policy_suffix_controls'], round(rep['final_force_qualified_angle'], 2), rep['environment_failure_reason']
    finally:
        if ep is not None: ep.close()
        env.close()
sp = sources['seed_193']
print('cold deterministic:', run(0, sp['source'], sp['physics'], True, 42), '| cold stochastic:', [run(0, sp['source'], sp['physics'], False, 500 + i)[0] for i in range(5)])
print('spine prefixes k=1..13 deterministic:', [run(k, sp['source'], sp['physics'], True, 42)[0] for k in range(1, 14)])
print('spine prefixes k=1..13 stochastic (3 each):', [sum(run(k, sp['source'], sp['physics'], False, 600 + 10 * k + i)[0] for i in range(3)) for k in range(1, 14)])
names = sorted(n for n in sources if n.startswith('track_')); rng2 = np.random.default_rng(1); pick = [names[i] for i in rng2.choice(len(names), size=24, replace=False)]
res = []
for n in pick:
    s = sources[n]; first = int(np.flatnonzero(s['origin'] == 'policy')[-1]) + 1; ok, ctrl, ang, fail = run(first, s['source'], s['physics'], True, 42); res.append((n.split('_', 2)[2], ok, ctrl, s['length'] - first))
print('corrected-state handoffs (deterministic): successes', sum(x[1] for x in res), '/', len(res)); print('  ', [(x[0], 'P' if x[1] else 'F', f'{x[2]}/{x[3]}') for x in res])
agent.save(Path(__file__).resolve().parent / f'bc_tube_{tag}.pt'); print('wall', round(time.monotonic() - t0, 1))
