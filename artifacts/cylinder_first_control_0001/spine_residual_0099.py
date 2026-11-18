"""Residual of a stage's actor at the recorded states of one demonstration: L2 between the deterministic actor action and the recorded control at each state, per checkpoint. Usage: spine_residual_0099.py [variant] [demo] -> spine_residual_<variant idx>_<demo>.json"""
import importlib.util, json, sys, numpy as np, torch
from pathlib import Path
A = Path('/home/autel001/code/mpc_dex/artifacts'); OUT = Path(__file__).resolve().parent
variant = sys.argv[1] if len(sys.argv) > 1 else 'bootstrap_variant_0099_cyl_spine172_from96'; demo = sys.argv[2] if len(sys.argv) > 2 else 'seed_172'; D = A / variant
spec = importlib.util.spec_from_file_location('sr_runner', D / 'run.py'); r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
torch.set_num_threads(1); torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32 = False
p = json.loads((D / 'protocol.json').read_text()); src = r.load_sources(p)[demo]['source']; obs = torch.as_tensor(src['obs'], device='cuda'); acts = src['actions']
out = {}
for ck in sorted((D / 'checkpoints').iterdir(), key=lambda q: q.name):
    if not (ck / 'agent.pt').exists(): continue
    agent = r.SACAgent.load(ck / 'agent.pt', 'cuda')
    with torch.no_grad(): a = agent.actor(obs)[0].tanh().cpu().numpy()
    out[ck.name] = [round(float(x), 5) for x in np.linalg.norm(a - acts, axis=1)]; print(ck.name, out[ck.name], flush=True)
json.dump(out, open(OUT / f'spine_residual_{variant[18:22]}_{demo}.json', 'w'), indent=1)
