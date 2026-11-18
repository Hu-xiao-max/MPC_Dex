"""Basins along a demonstration: perturb control j by a random direction of L2 radius r, then either replay the remaining
recorded controls open-loop or let a frozen policy continue deterministically. Usage: basin_spine.py <variant> <ckpt> <demo> <j_from> <j_to> <radii comma> <trials>"""
import importlib.util, json, sys
from pathlib import Path
A = Path('/home/autel001/code/mpc_dex/artifacts'); variant, label, demo = sys.argv[1], sys.argv[2], sys.argv[3]; j0, j1 = int(sys.argv[4]), int(sys.argv[5]); radii = [float(x) for x in sys.argv[6].split(',')]; K = int(sys.argv[7]); TAIL = int(sys.argv[8]); D = A / variant
spec = importlib.util.spec_from_file_location('basin_runner', D / 'run.py'); r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
np, torch = r.np, r.torch
torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32 = False
protocol = json.loads((D / 'protocol.json').read_text()); ec = r.EnvConfig(**json.loads((D / 'config.json').read_text())['environment'])
agent = r.SACAgent.load(D / 'checkpoints' / label / 'agent.pt', 'cuda'); sources = r.load_sources(protocol); src, phys, n = sources[demo]['source'], sources[demo]['physics'], sources[demo]['length']
def rec(action): return dict(action=np.asarray(action, np.float32), log_prob=np.nan, raw_mean=np.full(16, np.nan, np.float32), latent_std=np.full(16, np.nan, np.float32), density_is_stochastic_sample=False, learner_bc_updates=-1, learner_pretrain_updates=-1, learner_online_updates=-1)
out = {}
for j in range(j0, j1 + 1):
    row = {}
    for rad in radii:
        res = {'hold': 0, 'tail': 0, 'policy': 0}
        for t in range(K):
            rng = np.random.default_rng(70000 + 1000 * j + int(rad * 1e4) * 10 + t); d = rng.standard_normal(16); d *= rad / np.linalg.norm(d) if rad > 0 else 0.
            pert = np.clip(src['actions'][j] + d, -1, 1).astype(np.float32)
            for mode in ('hold', 'tail', 'policy'):
                # hold: replay the remaining recorded controls open-loop, then repeat the last one (<= 20 extra) until the boundary
                # tail: replay the recorded controls open-loop up to control TAIL-1, then the frozen policy finishes (it completes from k >= TAIL)
                # policy: the frozen policy continues immediately after the perturbed control
                r.seed_all(42); env = r.environment_for_policy(ec, protocol['policy_interface']); ep = None
                try:
                    ep = r.Episode(env, j, src, phys, protocol['physical_reset_seed'], f'basin_{demo}_{j}', 0, protocol['raw_reward_divisor'])
                    ep.advance(rec(pert), policy=False); i = j + 1; extra = 0
                    while not (ep.term or ep.trunc):
                        if mode == 'hold':
                            if i < n: ep.advance(rec(src['actions'][i]), policy=False); i += 1
                            elif extra < 20: ep.advance(rec(src['actions'][n - 1]), policy=False); extra += 1
                            else: break
                        elif mode == 'tail' and i < TAIL: ep.advance(rec(src['actions'][i]), policy=False); i += 1
                        else: ep.advance(r.command(agent, ep.obs, True))
                    res[mode] += int(ep.report()['finger_supported_rotation_success'])
                finally:
                    if ep is not None: ep.close()
                    env.close()
        row[str(rad)] = res
    out[j] = row; print(f'control {j}: ' + ' | '.join(f"r={rad}: hold {row[str(rad)]['hold']}/{K} tail {row[str(rad)]['tail']}/{K} policy {row[str(rad)]['policy']}/{K}" for rad in radii), flush=True)
json.dump(out, open(Path(__file__).resolve().parent / f'basin_{demo}_{variant[18:22]}_{label}.json', 'w'), indent=1)
