"""Why does the BC-fitted actor (L2 <= 1e-5 at the recorded states) fail from the cold reset? Per-step action and observation deviation along the deterministic rollout; an obs-frozen rollout (actor fed the recorded observations, i.e. open-loop through the actor); actor sensitivity to observation perturbations."""
import importlib.util, json, numpy as np, torch
from pathlib import Path
A = Path('/home/autel001/code/mpc_dex/artifacts'); D = A / 'bootstrap_variant_0099_cyl_spine172_from96'; OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('r99d', D / 'run.py'); r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
torch.set_num_threads(1); torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32 = False
p = json.loads((D / 'protocol.json').read_text()); ec = r.EnvConfig(**json.loads((D / 'config.json').read_text())['environment'])
sources = r.load_sources(p); src, phys, n = sources['seed_172']['source'], sources['seed_172']['physics'], sources['seed_172']['length']
agent = r.SACAgent.load(A / 'bootstrap_variant_0096_cyl_ms_s1to9_long_from94/checkpoints/online_4000/agent.pt', 'cuda'); agent.actor.load_state_dict(torch.load(OUT / 'actor_bcfit_172.pt'))
np.set_printoptions(precision=6, suppress=True, linewidth=220)
def act(o):
    with torch.no_grad(): return agent.actor(torch.as_tensor(np.asarray(o, np.float32)[None], device='cuda'))[0].tanh()[0].cpu().numpy()
def rec(action): return dict(action=np.asarray(action, np.float32), log_prob=np.nan, raw_mean=np.full(16, np.nan, np.float32), latent_std=np.full(16, np.nan, np.float32), density_is_stochastic_sample=False, learner_bc_updates=-1, learner_pretrain_updates=-1, learner_online_updates=-1)
def run(mode):
    r.seed_all(42); env = r.environment_for_policy(ec, p['policy_interface']); ep = None; log = []
    try:
        ep = r.Episode(env, 0, src, phys, p['physical_reset_seed'], f'diag_{mode}', 0, p['raw_reward_divisor']); t = 0
        while not (ep.term or ep.trunc):
            o = np.asarray(ep.obs, np.float32); d_obs = float(np.linalg.norm(o - src['obs'][t])) if t < n else float('nan')
            if mode == 'closed': a = act(o)
            elif mode == 'obs_frozen': a = act(src['obs'][min(t, n - 1)])
            elif mode == 'replay': a = src['actions'][min(t, n - 1)]
            d_act = float(np.linalg.norm(a - src['actions'][t])) if t < n else float('nan')
            log.append((t, d_obs, d_act)); ep.advance(rec(a), policy=False); t += 1
        rep = ep.report(); return dict(success=bool(rep['finger_supported_rotation_success']), controls=t, failure=rep['environment_failure_reason'], angle=round(rep['final_force_qualified_angle'], 3), log=log)
    finally:
        if ep is not None: ep.close()
        env.close()
res = {}
for mode in ('replay', 'obs_frozen', 'closed'):
    res[mode] = run(mode); x = res[mode]
    print(f"{mode:11s} success {x['success']} controls {x['controls']} failure {x['failure']} angle {x['angle']}"); print('   t, |obs - rec obs|, |action - rec action|:', [(t, f'{a:.2e}', f'{b:.2e}') for t, a, b in x['log']])
# actor sensitivity: random observation perturbations of radius rho at the recorded states
rng = np.random.default_rng(0)
for rho in (1e-5, 1e-4, 1e-3, 1e-2):
    g = []
    for t in range(n):
        for _ in range(20):
            d = rng.standard_normal(src['obs'].shape[1]).astype(np.float32); d *= rho / np.linalg.norm(d)
            g.append(np.linalg.norm(act(src['obs'][t] + d) - act(src['obs'][t])) / rho)
    print(f'actor gain |da|/|dobs| at radius {rho:.0e}: median {np.median(g):.2f} max {np.max(g):.2f}')
json.dump(res, open(OUT / 'diag_172.json', 'w'), indent=1)
