"""Freeze one eight-control-prefix support probe before execution."""
from pathlib import Path
import hashlib
import json
import shutil

D = Path(__file__).resolve().parent
S = D.parent / 'bootstrap_0005/value_calibration/initial_0001'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


assert not (D / 'inputs_manifest.json').exists()
shutil.copytree(S / 'source', D / 'source', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
bindings = {}
for name, original in {
    'checkpoint.pt': S / 'checkpoint.pt',
    'checkpoint.pt.metadata.json': S / 'checkpoint.pt.metadata.json',
    'source_config.json': S / 'config.json',
    'initial_grasp.npz': S / 'initial_grasp.npz',
    'source_report.json': S / 'report.json',
    'prefix_transitions.npz': S / 'deterministic_strict/transitions.npz',
    'prefix_physics.npz': S / 'deterministic_strict/physics.npz',
    'prefix_report.json': S / 'deterministic_strict/report.json',
}.items():
    shutil.copyfile(original, D / name)
    bindings[name] = {'source': str(original), 'sha256': sha(original)}
    assert sha(D / name) == sha(original)
config = json.loads((D / 'source_config.json').read_text())
config['environment']['initial_grasp'] = str(D / 'initial_grasp.npz')
(D / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
protocol = {
    'purpose': 'Test stochastic final-suffix support before selecting a reset curriculum',
    'prefix_controls': 8,
    'prefix_source': 'Actual successful BC-origin deterministic trace of bootstrap0005 initialization',
    'policy_checkpoint_online_updates': 0,
    'policy_checkpoint_bc_updates': 12000,
    'latent_std': .05,
    'experiments': [{'seed': 42, 'deterministic': True}] + [{'seed': s, 'deterministic': False} for s in range(17041, 17046)],
    'sampling': 'Float32 CUDA actor.sample on shape(1,60), actual returned log probability; RNG begins at first suffix decision',
    'physical_horizon_including_prefix': 100,
    'unchanged_physics_reward_success_criteria': True,
    'new_learning_updates': 0,
    'new_mpc_queries': 0,
    'other_prefixes_or_std_sweeps': 0,
    'post_success_continuation': False,
    'all_results_are_prefix_assisted': True,
    'eligible_primitive': False,
    'cold_start_policy_success_claim': False,
    'source_bindings': bindings,
}
(D / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
(D / 'inputs_manifest.json').write_text(json.dumps({str(p.relative_to(D)): sha(p) for p in sorted(D.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name != 'inputs_manifest.json'}, indent=2) + '\n')
print(json.dumps({'prepared': True, 'checkpoint_sha256': sha(D / 'checkpoint.pt'), 'source_files': len(list((D / 'source').rglob('*')))}))
