"""Read-only basin probe: recorded control h at prefix h-1 plus a random perturbation, then the frozen policy.

Usage: basin_probe.py <variant_dir> <checkpoint> <handoff> [<handoff> ...]
"""
import importlib.util, json, sys
from pathlib import Path
A = Path('/home/autel001/code/mpc_dex/artifacts'); V = A / sys.argv[1]; label = sys.argv[2]; handoffs = [int(x) for x in sys.argv[3:]]
spec = importlib.util.spec_from_file_location('probe_runner', V / 'run.py'); r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
np, torch = r.np, r.torch
torch.set_num_threads(1); torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32 = False
agent = r.SACAgent.load(V / 'checkpoints' / label / 'agent.pt', 'cuda')
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
rng = np.random.default_rng(5); out = {}
for h in handoffs:
    rec = source['actions'][h - 1].astype(np.float64)
    for radius in (0., .005, .01, .02, .05):
        res = []
        for k in range(8 if radius > 0 else 1):
            d = rng.standard_normal(16); d = d / np.linalg.norm(d) * radius
            res.append(trial(h, np.clip(rec + d, -1, 1).astype(np.float32), 42, True))
        out[f'control{h}_at_s{h - 1} r={radius}'] = f"{sum(x[0] for x in res)}/{len(res)} det; failing suffix lengths {[x[1] for x in res if not x[0]]}"
    st = [trial(h, source['actions'][h - 1].copy(), s, False) for s in range(31001, 31011)]
    out[f'control{h}_at_s{h - 1} exact + stochastic policy'] = f"{sum(x[0] for x in st)}/10"
    with torch.no_grad():
        m = agent.actor(torch.as_tensor(source['obs'][h - 1][None], device='cuda'))[0].tanh()[0].cpu().numpy()
    out[f'policy mean at s{h - 1} vs recorded control {h}: L2'] = f"{np.linalg.norm(m - source['actions'][h - 1]):.4f}"
    det = trial(h - 1, None, 42, True); stp = [trial(h - 1, None, s, False) for s in range(31001, 31011)]
    out[f'policy from prefix {h - 1}: det, stoch'] = f"{det[0]} ({det[1]} ctrls, {det[2]}); {sum(x[0] for x in stp)}/10"
print(json.dumps(dict(variant=sys.argv[1], checkpoint=label, **out), indent=1))
