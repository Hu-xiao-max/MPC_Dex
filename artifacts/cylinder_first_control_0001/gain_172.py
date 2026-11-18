"""Read-only recomputation of the console-only numbers of sections 1-2: the start actor's per-row residual, the fitted actor's per-row residual and BC MSE, its latent std at the rows, and its gain |da|/|dobs| under random observation perturbations. Writes gain_172.json."""
import importlib.util, json, numpy as np, torch
from pathlib import Path
A = Path('/home/autel001/code/mpc_dex/artifacts'); D = A / 'bootstrap_variant_0099_cyl_spine172_from96'; OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('r99g', D / 'run.py'); r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
torch.set_num_threads(1); torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32 = False
p = json.loads((D / 'protocol.json').read_text()); src = r.load_sources(p)['seed_172']['source']; obs = torch.as_tensor(src['obs'], device='cuda'); acts = torch.as_tensor(src['actions'], device='cuda')
agent = r.SACAgent.load(A / 'bootstrap_variant_0096_cyl_ms_s1to9_long_from94/checkpoints/online_4000/agent.pt', 'cuda')
def residual(): 
    with torch.no_grad(): return [round(float(x), 6) for x in (agent.actor(obs)[0].tanh() - acts).norm(dim=1).cpu().numpy()]
out = {'start_residual_0096_online_4000': residual()}
agent.actor.load_state_dict(torch.load(OUT / 'actor_bcfit_172.pt')); out['fitted_residual'] = residual()
with torch.no_grad():
    mean, log_std = agent.actor(obs)[:2]; eps = 0.02; out['fitted_bc_mse_atanh'] = float(((mean - torch.atanh(acts.clamp(-1 + eps, 1 - eps))) ** 2).mean()); out['fitted_latent_std_range'] = [float(log_std.exp().min()), float(log_std.exp().max())]
def act(o):
    with torch.no_grad(): return agent.actor(torch.as_tensor(np.asarray(o, np.float32)[None], device='cuda'))[0].tanh()[0].cpu().numpy()
rng = np.random.default_rng(0); out['gain'] = {}
for rho in (1e-5, 1e-4, 1e-3, 1e-2):
    g = []
    for t in range(len(src['obs'])):
        for _ in range(20):
            d = rng.standard_normal(src['obs'].shape[1]).astype(np.float32); d *= rho / np.linalg.norm(d); g.append(float(np.linalg.norm(act(src['obs'][t] + d) - act(src['obs'][t])) / rho))
    out['gain'][f'{rho:.0e}'] = dict(median=round(float(np.median(g)), 3), max=round(float(np.max(g)), 3))
json.dump(out, open(OUT / 'gain_172.json', 'w'), indent=1); print(json.dumps(out))
