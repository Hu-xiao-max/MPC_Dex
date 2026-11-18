"""Scratch probe: fine-tune the 0096 online_4000 actor by BC alone on seed_172's eight rows; report per-row L2 to the recorded controls and roll out from the cold reset deterministically (and with the actor's own std) in the stage environment. Nothing is written into any stage directory."""
import importlib.util, json, sys, time, numpy as np, torch
from pathlib import Path
A = Path('/home/autel001/code/mpc_dex/artifacts'); D = A / 'bootstrap_variant_0099_cyl_spine172_from96'; OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('r99c', D / 'run.py'); r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
torch.set_num_threads(1); torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32 = False
p = json.loads((D / 'protocol.json').read_text()); ec = r.EnvConfig(**json.loads((D / 'config.json').read_text())['environment'])
sources = r.load_sources(p); src, phys, n = sources['seed_172']['source'], sources['seed_172']['physics'], sources['seed_172']['length']
start = sys.argv[1] if len(sys.argv) > 1 else str(A / 'bootstrap_variant_0096_cyl_ms_s1to9_long_from94/checkpoints/online_4000/agent.pt')
lr = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-4; steps = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
agent = r.SACAgent.load(start, 'cuda'); obs = torch.as_tensor(src['obs'], device='cuda'); acts = torch.as_tensor(src['actions'], device='cuda')
eps = 0.02; target = torch.atanh(acts.clamp(-1 + eps, 1 - eps))
opt = torch.optim.Adam(agent.actor.parameters(), lr=lr)
def l2():
    with torch.no_grad(): return (agent.actor(obs)[0].tanh() - acts).norm(dim=1).cpu().numpy()
print('start L2 per row:', l2().round(4), flush=True); t0 = time.monotonic()
for it in range(1, steps + 1):
    mean = agent.actor(obs)[0]; loss = ((mean - target) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if it % 2000 == 0 or it == steps:
        d = l2(); print(f'step {it}: bc mse {loss.item():.2e}  L2 per row {d.round(5)}  max {d.max():.5f}  ({time.monotonic() - t0:.0f}s)', flush=True)
        if d.max() < 5e-5: break
d = l2()
def rollout(prefix, det, seed):
    r.seed_all(seed); env = r.environment_for_policy(ec, p['policy_interface']); ep = None
    try:
        ep = r.Episode(env, prefix, src, phys, p['physical_reset_seed'], f'probe172_p{prefix}', seed, p['raw_reward_divisor'])
        while not (ep.term or ep.trunc): ep.advance(r.command(agent, ep.obs, det))
        rep = ep.report(); return dict(success=bool(rep['finger_supported_rotation_success']), controls=rep['policy_suffix_controls'], failure=rep['environment_failure_reason'], angle=round(rep['final_force_qualified_angle'], 3))
    finally:
        if ep is not None: ep.close()
        env.close()
res = {'final_l2': d.tolist(), 'det': {k: rollout(k, True, 42) for k in range(0, 4)}, 'stoch_cold': [rollout(0, False, 50000 + i) for i in range(1, 11)]}
print('deterministic rollouts by prefix k:', res['det']); print('stochastic cold (actor std):', sum(x['success'] for x in res['stoch_cold']), '/ 10', [x['controls'] for x in res['stoch_cold']])
torch.save(agent.actor.state_dict(), OUT / 'actor_bcfit_172.pt'); json.dump(res, open(OUT / 'bc_fit_172.json', 'w'), indent=1)
