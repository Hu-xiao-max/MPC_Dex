"""Read-only basin probe at the prefix-5 handoff: recorded control 6 + perturbation, then the frozen 0039 policy."""
import importlib.util, json, sys
from pathlib import Path
A = Path('/home/autel001/code/mpc_dex/artifacts'); V = A / 'bootstrap_variant_0039_bc_first_l100_std01'
spec = importlib.util.spec_from_file_location('r39', V / 'run.py'); r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
np, torch = r.np, r.torch
torch.set_num_threads(1); torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32 = False
agent = r.SACAgent.load(V / 'checkpoints/online_2000/agent.pt', 'cuda')
proto = json.loads((V / 'protocol.json').read_text()); ec = r.EnvConfig(**json.loads((V / 'config.json').read_text())['environment'])
with np.load(V / 'prefix_transitions.npz') as z: source = {k: z[k].copy() for k in z.files}
with np.load(V / 'prefix_physics.npz') as z: physics = z['samples'].copy()
def trial(prefix, first_action, seed, deterministic):
    r.seed_all(seed); env = r.environment_for_policy(ec, proto['policy_interface'])
    try:
        ep = r.Episode(env, prefix, source, physics, 42, 'probe', seed, 200., noisy_last=first_action)
        while not (ep.term or ep.trunc):
            ep.advance(r.command(agent, ep.obs, deterministic))
        rep = ep.report(); ep.close(); return rep['finger_supported_rotation_success'], rep['policy_suffix_controls'], rep['environment_failure_reason']
    finally: env.close()
rng = np.random.default_rng(5)
out = {}
for handoff, label in ((6, 'control6_at_s5'), (7, 'control7_at_s6')):
    rec = source['actions'][handoff - 1].astype(np.float64)
    for radius in (0., .01, .02, .05, .1):
        res = []
        for k in range(8 if radius > 0 else 1):
            d = rng.standard_normal(16); d = d / np.linalg.norm(d) * radius
            a = np.clip(rec + d, -1, 1).astype(np.float32)
            res.append(trial(handoff, a, 42, True))
        out[f'{label} r={radius}'] = f"{sum(x[0] for x in res)}/{len(res)} det; fails at {[x[1] for x in res if not x[0]]}"
    # stochastic policy after the exact recorded control
    st = [trial(handoff, source['actions'][handoff - 1].copy(), s, False) for s in range(31001, 31011)]
    out[f'{label} exact + stochastic policy'] = f"{sum(x[0] for x in st)}/10"
# policy itself from prefix 5 (mean control 6 = policy's own)
det5 = [trial(5, None, 42, True)]; st5 = [trial(5, None, s, False) for s in range(31001, 31011)]
out['policy from prefix 5: det, stoch'] = f"{det5[0][0]} ({det5[0][1]} ctrls, {det5[0][2]}); {sum(x[0] for x in st5)}/10"
with torch.no_grad():
    m = agent.actor(torch.as_tensor(source['obs'][5][None], device='cuda'))[0].tanh()[0].cpu().numpy()
out['policy mean at s5 vs recorded control 6: L2, maxabs'] = f"{np.linalg.norm(m - source['actions'][5]):.4f}, {np.abs(m - source['actions'][5]).max():.4f}"
print(json.dumps(out, indent=1))
