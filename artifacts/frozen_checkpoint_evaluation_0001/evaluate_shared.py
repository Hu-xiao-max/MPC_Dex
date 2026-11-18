"""Large-sample frozen evaluation of a checkpoint at the shared cold-start states: the cold reset (k = 0) and, for a
multi-source stage with shared_prefix_groups, the states after k shared controls (k >= 1), each with 1 deterministic +
<trials> fresh-seed stochastic trials. The whole remaining turn is policy-controlled.

Usage: evaluate_shared.py <variant_dir_name> <checkpoint_label> <trials> [k ...]   (default k: 0 plus the stage's start_prefix_set)
"""
import hashlib, importlib.util, json, sys, time
from pathlib import Path
A = Path('/home/autel001/code/mpc_dex/artifacts'); OUT = Path(__file__).resolve().parent / 'results_shared'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
variant, label, trials = sys.argv[1], sys.argv[2], int(sys.argv[3]); D = A / variant
spec = importlib.util.spec_from_file_location('shared_runner_' + variant, D / 'run.py'); r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
np, torch = r.np, r.torch
torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32 = False
ck = D / 'checkpoints' / label; manifest = json.loads((ck / 'manifest.json').read_text()); assert sha(ck / 'agent.pt') == manifest['agent.pt']
protocol = json.loads((D / 'protocol.json').read_text()); ec = r.EnvConfig(**json.loads((D / 'config.json').read_text())['environment'])
agent = r.SACAgent.load(ck / 'agent.pt', 'cuda'); sources = r.load_sources(protocol) if protocol.get('prefix_sources') else {}
groups = protocol.get('shared_prefix_groups') or {'0': [None]}
ks = [int(x) for x in sys.argv[4:]] or sorted({0, *(protocol.get('start_prefix_set') or [])})
before = {k: r.tensor_hashes(getattr(agent, k)) for k in ('actor', 'critics', 'target_critics')}
out = dict(variant=variant, checkpoint=label, agent_sha256=manifest['agent.pt'], trials_per_cell=trials, cells={}); started = time.monotonic()
# the single-source stages replay the recorded nominal trajectory; multi-source stages replay a group representative (the shared state is identical for every member)
rec_source, rec_physics = None, None
if not sources:
    with np.load(D / 'prefix_transitions.npz', allow_pickle=False) as z: rec_source = {k: z[k].copy() for k in z.files}
    with np.load(D / 'prefix_physics.npz', allow_pickle=False) as z: rec_physics = z['samples'].copy()
for k in ks:
    rep = groups[str(k)][0] if str(k) in groups else None
    if k > 0: assert rep is not None, f'no shared group for k={k}'
    src, phys = (sources[rep]['source'], sources[rep]['physics']) if rep else (rec_source, rec_physics)
    rows = []
    for seed, det in [(protocol['eval_deterministic_seed'], True)] + [(50000 + 1000 * k + i, False) for i in range(1, trials + 1)]:
        r.seed_all(seed); env = r.environment_for_policy(ec, protocol['policy_interface']); ep = None
        try:
            ep = r.Episode(env, k, src, phys, protocol['physical_reset_seed'], f'shared_k{k}', seed, protocol['raw_reward_divisor'])
            while not (ep.term or ep.trunc):
                ep.advance(r.command(agent, ep.obs, det))
            rep_ = ep.report(); rows.append(dict(seed=seed, deterministic=det, success=bool(rep_['finger_supported_rotation_success']), controls=rep_['policy_suffix_controls'], failure=rep_['environment_failure_reason'], final_force_qualified_angle=rep_['final_force_qualified_angle']))
        finally:
            if ep is not None: ep.close()
            env.close()
    st = [x for x in rows if not x['deterministic']]
    out['cells'][f'k{k}'] = dict(representative=rep, group_size=len(groups[str(k)]) if str(k) in groups else None, deterministic_success=rows[0]['success'], stochastic_successes=sum(x['success'] for x in st), stochastic_trials=len(st), rows=rows)
    print(json.dumps(dict(k=k, det=rows[0]['success'], stoch=f"{sum(x['success'] for x in st)}/{len(st)}", max_angle=round(max(x['final_force_qualified_angle'] for x in rows), 3))), flush=True)
assert before == {k: r.tensor_hashes(getattr(agent, k)) for k in before}
out['wall_seconds'] = time.monotonic() - started
(OUT / variant).mkdir(parents=True, exist_ok=True); (OUT / variant / f'{label}.json').write_text(json.dumps(out, indent=1) + '\n')
print(json.dumps(dict(variant=variant, checkpoint=label, **{c: f"{v['stochastic_successes']}/{v['stochastic_trials']}{' D' if v['deterministic_success'] else ''}" for c, v in out['cells'].items()}, wall=round(out['wall_seconds'], 1))), flush=True)
