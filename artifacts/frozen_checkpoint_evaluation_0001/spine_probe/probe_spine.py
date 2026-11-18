"""Policy competence along candidate spine demonstrations: for each prefix p = 0..n-1, 1 deterministic + K stochastic continuations.
Usage: probe_spine.py <variant> <ckpt> <K> <demo...>"""
import importlib.util, json, sys, time
from pathlib import Path
A = Path('/home/autel001/code/mpc_dex/artifacts'); variant, label, K = sys.argv[1], sys.argv[2], int(sys.argv[3]); D = A / variant
spec = importlib.util.spec_from_file_location('spine_runner', D / 'run.py'); r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
np, torch = r.np, r.torch
torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32 = False
protocol = json.loads((D / 'protocol.json').read_text()); ec = r.EnvConfig(**json.loads((D / 'config.json').read_text())['environment'])
agent = r.SACAgent.load(D / 'checkpoints' / label / 'agent.pt', 'cuda'); sources = r.load_sources(protocol)
out = {}
for name in sys.argv[4:]:
    n = sources[name]['length']; rows = []
    for p in range(n):
        cell = []
        for seed, det in [(42, True)] + [(60000 + 100 * p + i, False) for i in range(1, K + 1)]:
            r.seed_all(seed); env = r.environment_for_policy(ec, protocol['policy_interface']); ep = None
            try:
                ep = r.Episode(env, p, sources[name]['source'], sources[name]['physics'], protocol['physical_reset_seed'], f'{name}_p{p}', seed, protocol['raw_reward_divisor'])
                while not (ep.term or ep.trunc): ep.advance(r.command(agent, ep.obs, det))
                rep = ep.report(); cell.append(dict(det=det, ok=bool(rep['finger_supported_rotation_success']), ctrl=rep['policy_suffix_controls'], angle=round(rep['final_force_qualified_angle'], 2), fail=rep['environment_failure_reason']))
            finally:
                if ep is not None: ep.close()
                env.close()
        rows.append(dict(prefix=p, suffix=n - p, det=cell[0]['ok'], stoch=sum(c['ok'] for c in cell[1:]), K=K, det_ctrl=cell[0]['ctrl'], det_angle=cell[0]['angle'], det_fail=cell[0]['fail']))
    out[name] = rows
    print(name, 'n', n, ' '.join(f"p{x['prefix']}:{'P' if x['det'] else 'F'}{x['stoch']}" for x in rows), flush=True)
json.dump(out, open(Path(__file__).resolve().parent / f'spine_probe_{variant[18:22]}_{label}.json', 'w'), indent=1)
