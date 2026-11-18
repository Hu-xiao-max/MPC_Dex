"""Freeze the fixed descending-prefix screen before any physics executes."""
from pathlib import Path
import hashlib
import json
import shutil

D = Path(__file__).resolve().parent
S = D.parent / 'bootstrap_suffix_support_0001'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


assert not (D / 'inputs_manifest.json').exists()
source_manifest = json.loads((S / 'manifest.json').read_text())
assert all(sha(S / path) == digest for path, digest in source_manifest.items())
shutil.copytree(S / 'source', D / 'source', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
bindings = {}
for name in ('checkpoint.pt', 'checkpoint.pt.metadata.json', 'source_config.json', 'initial_grasp.npz', 'source_report.json', 'prefix_transitions.npz', 'prefix_physics.npz', 'prefix_report.json'):
    shutil.copyfile(S / name, D / name)
    bindings[name] = {'source': str(S / name), 'sha256': sha(S / name)}
    assert sha(D / name) == bindings[name]['sha256']
archive = D / 'prefix8_reference'
archive.mkdir()
for name in ('prepare.py', 'run.py', 'protocol.json', 'summary.json', 'inputs_manifest.json', 'manifest.json'):
    shutil.copyfile(S / name, archive / name)
config = json.loads((D / 'source_config.json').read_text())
config['environment']['initial_grasp'] = str(D / 'initial_grasp.npz')
write(D / 'config.json', config)
write(D / 'protocol.json', {
    'purpose': 'Locate stochastic suffix support with one fixed descending-prefix screen; no learning',
    'prefix_order': [7, 6, 5, 4, 3, 2, 1],
    'stop_after_first_stage_with_fewer_than_stochastic_successes': 4,
    'max_stages': 7,
    'experiments_per_stage': [{'seed': 42, 'deterministic': True}] + [{'seed': seed, 'deterministic': False} for seed in range(17041, 17046)],
    'prefix_source': 'Actual successful BC-origin nine-control deterministic trace of bootstrap0005 initialization',
    'latent_std': 0.05,
    'policy_checkpoint_bc_updates': 12000,
    'policy_checkpoint_online_updates': 0,
    'physical_horizon_including_prefix': 100,
    'sampling': 'CUDA float32 actor.sample with shape (1,60); independent seed reset each episode; no draws during the physically executed prefix',
    'strict_success': 'Environment success AND independently force-qualified full turn AND no environment/independent drop, no palm or other external support, actual success termination',
    'prefix_verification': 'Every saved initial field, all recorded prefix transitions/full integration states and all 2ms physics samples bit-exact; deterministic full nine-control trace bit-exact at every stage',
    'unchanged_physics_reward_success_criteria': True,
    'new_learning_updates': 0,
    'new_mpc_queries': 0,
    'state_injection_at_handoff': False,
    'history_rebasing_at_handoff': False,
    'post_success_continuation': False,
    'all_results_are_prefix_assisted': True,
    'eligible_primitive': False,
    'cold_start_policy_success_claim': False,
    'cold_zero_prefix_repeated': False,
    'selection_gate_is_not_reliability_estimate': True,
    'new_held_out_seeds_required_for_later_training_evaluation': True,
    'source_bindings': bindings,
    'source_prefix8_manifest_sha256': sha(S / 'manifest.json'),
})
write(D / 'command.json', {'argv': ['.venv/bin/python', str(D / 'run.py')], 'cwd': str(D.parents[1]), 'environment': {'PYTHONHASHSEED': '0'}})
write(D / 'inputs_manifest.json', {str(p.relative_to(D)): sha(p) for p in sorted(D.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name != 'inputs_manifest.json'})
print(json.dumps({'prepared': True, 'protocol_sha256': sha(D / 'protocol.json'), 'input_manifest_sha256': sha(D / 'inputs_manifest.json'), 'checkpoint_sha256': sha(D / 'checkpoint.pt')}))
