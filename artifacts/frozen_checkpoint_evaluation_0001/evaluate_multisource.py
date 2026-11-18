"""Large-sample frozen evaluation of a multi-source stage checkpoint: per source x suffix, 30 fresh-seed stochastic trials + 1 deterministic.

Usage: evaluate_multisource.py <variant_dir_name> <checkpoint_label> <trials> [source ...]  (default: the stage's eval_source_names)
"""
import hashlib, importlib.util, json, sys, time
from pathlib import Path
A = Path('/home/autel001/code/mpc_dex/artifacts'); OUT = Path(__file__).resolve().parent / 'results_multisource'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
variant, label, trials = sys.argv[1], sys.argv[2], int(sys.argv[3]); D = A / variant
spec = importlib.util.spec_from_file_location('ms_runner_' + variant, D / 'run.py'); r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
np, torch = r.np, r.torch
torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32 = False
ck = D / 'checkpoints' / label; manifest = json.loads((ck / 'manifest.json').read_text()); assert sha(ck / 'agent.pt') == manifest['agent.pt']
protocol = json.loads((D / 'protocol.json').read_text()); ec = r.EnvConfig(**json.loads((D / 'config.json').read_text())['environment'])
agent = r.SACAgent.load(ck / 'agent.pt', 'cuda'); sources = r.load_sources(protocol)
names = sys.argv[4:] or protocol['eval_source_names']
COLL = A / (protocol['prefix_sources'][0].get('collection', 'mpc_cold_demonstrations_0001') if protocol.get('prefix_sources') else 'mpc_cold_demonstrations_0001') / 'episodes'  # unseen sources come from the stage's own primary collection
for name in names:
    if name not in sources:  # a demonstration the stage never loaded: read it from the verified collection
        rep_ = json.loads((COLL / name / 'report.json').read_text()); assert rep_['strict_force_success'] and rep_['stored']
        with np.load(COLL / name / 'transitions.npz', allow_pickle=False) as z: src = {k: z[k].copy() for k in z.files}
        with np.load(COLL / name / 'physics.npz', allow_pickle=False) as z: phys = z['samples'].copy()
        assert src['terminated'][-1] and sha(COLL / name / 'transitions.npz') == rep_['transitions_sha256']
        sources[name] = dict(source=src, physics=phys, length=int(len(src['actions'])), external=True)
before = {k: r.tensor_hashes(getattr(agent, k)) for k in ('actor', 'critics', 'target_critics')}
out = dict(variant=variant, checkpoint=label, agent_sha256=manifest['agent.pt'], trials_per_cell=trials, holdout_source_names=protocol.get('holdout_source_names', []), cells={})
started = time.monotonic()
for name in names:
    n = sources[name]['length']
    for s in protocol['suffix_set']:
        prefix = max(0, n - s); rows = []  # as the in-run evaluation (run.py: prefix = max(0, n - s)): a suffix at least as long as the source is the cold reset
        for k, (seed, det) in enumerate([(protocol['eval_deterministic_seed'], True)] + [(40000 + 1000 * s + i, False) for i in range(1, trials + 1)]):
            r.seed_all(seed); env = r.environment_for_policy(ec, protocol['policy_interface']); ep = None
            try:
                ep = r.Episode(env, prefix, sources[name]['source'], sources[name]['physics'], protocol['physical_reset_seed'], f'{name}_s{s}', seed, protocol['raw_reward_divisor'])
                while not (ep.term or ep.trunc):
                    ep.advance(r.command(agent, ep.obs, det))
                rep = ep.report(); rows.append(dict(seed=seed, deterministic=det, success=bool(rep['finger_supported_rotation_success']), controls=rep['policy_suffix_controls'], failure=rep['environment_failure_reason']))
            finally:
                if ep is not None: ep.close()
                env.close()
        out['cells'][f'{name}:s{s}'] = dict(prefix=prefix, source_controls=n, clamped_to_cold_reset=(s >= n), deterministic_success=rows[0]['success'], stochastic_successes=sum(x['success'] for x in rows[1:]), stochastic_trials=trials, trials=rows, holdout=(name in out['holdout_source_names']) or (name not in protocol['prefix_source_names']))
assert before == {k: r.tensor_hashes(getattr(agent, k)) for k in before}
out['wall_seconds'] = time.monotonic() - started
(OUT / variant).mkdir(parents=True, exist_ok=True); result_name = f'{label}.json' if not sys.argv[4:] else f"{label}__{'_'.join(names)}.json"  # an explicit source list never overwrites the default-list result
(OUT / variant / result_name).write_text(json.dumps(out, indent=1) + '\n')
summary = {}
for s in protocol['suffix_set']:
    cells = [v for k, v in out['cells'].items() if k.endswith(f':s{s}')]
    summary[f's{s}'] = f"{sum(c['stochastic_successes'] for c in cells)}/{sum(c['stochastic_trials'] for c in cells)} det {sum(c['deterministic_success'] for c in cells)}/{len(cells)}"
    hold = [c for c in cells if c['holdout']]
    if hold: summary[f's{s}_holdout'] = f"{sum(c['stochastic_successes'] for c in hold)}/{sum(c['stochastic_trials'] for c in hold)}"
print(json.dumps(dict(variant=variant, checkpoint=label, **summary, wall=round(out['wall_seconds'], 1))), flush=True)
