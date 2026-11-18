"""Freeze the authorized trained-policy rollback screen before physics."""
from pathlib import Path
import hashlib
import json
import shutil

D = Path(__file__).resolve().parent
S = D.parent / 'bootstrap_suffix_sac_0001'
B = D.parent / 'bootstrap_suffix_rollback_0001'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

assert not (D / 'inputs_manifest.json').exists()
assert sha(S / 'manifest.json') == '258842818fa0087aa70566996fad768fa088c1e6e3102d533c1c40f3e48edfc0'
assert json.loads((S / 'summary.json').read_text())['status'] == 'complete'
source_manifest = json.loads((S / 'manifest.json').read_text())
bindings = {}
for name, relative in {
    'checkpoint.pt': 'checkpoints/online_2000/agent.pt',
    'checkpoint.pt.metadata.json': 'checkpoints/online_2000/agent.pt.metadata.json',
    'source_checkpoint_state.json': 'checkpoints/online_2000/state.json',
    'source_checkpoint_manifest.json': 'checkpoints/online_2000/manifest.json',
    'source_execution_manifest.json': 'execution_manifest.json',
    'source_summary.json': 'summary.json',
    'source_config.json': 'config.json',
    'source_protocol.json': 'protocol.json',
    'source_report.json': 'source_report.json',
    'initial_grasp.npz': 'initial_grasp.npz',
    'prefix_transitions.npz': 'prefix_transitions.npz',
    'prefix_physics.npz': 'prefix_physics.npz',
    'prefix_report.json': 'prefix_report.json',
}.items():
    if relative in source_manifest:
        assert sha(S / relative) == source_manifest[relative]
    else:
        # Root's final manifest excludes files named manifest.json. Bind the
        # checkpoint's own manifest by checking every entry against that root.
        assert relative == 'checkpoints/online_2000/manifest.json'
        for entry, digest in json.loads((S / relative).read_text()).items():
            full = 'checkpoints/online_2000/' + entry
            assert digest == source_manifest[full] == sha(S / full)
    shutil.copyfile(S / relative, D / name)
    bindings[name] = {'source': str(S / relative), 'sha256': sha(S / relative)}
assert sha(D / 'checkpoint.pt') == 'e4a73d890963a70afc9337db7d29d345c38e192218a7b81dbd51cd6589232892'
shutil.copyfile(S / 'manifest.json', D / 'source_manifest.json')
shutil.copytree(S / 'source', D / 'source', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
for path in (D / 'source').rglob('*'):
    if path.is_file():
        assert sha(path) == source_manifest[str(path.relative_to(D))]
base = D / 'runner_reference'
base.mkdir()
for name in ('run.py', 'verify_sampling.py', 'protocol.json', 'manifest.json'):
    shutil.copyfile(B / name, base / name)

# Only artifact-local harness changes: physics, agent, replay and optimizers are copied unchanged.
run = (base / 'run.py').read_text()
replacements = [
    ("list(range(7, 0, -1))", "list(range(6, 0, -1))"),
    ("(60, 16, 12000, 0, 0)", "(60, 16, 12000, 500, 2000)"),
    ("(12000, 0, 0)", "(12000, 500, 2000)"),
    ("ob, info = original_reset(*args, **kwargs)", "kwargs['seed'] = protocol['physical_reset_seed']\n                ob, info = original_reset(*args, **kwargs)"),
    ("            if deterministic:\n                for key in checked_fields:\n                    np.testing.assert_array_equal(arrays[key], source[key])\n                np.testing.assert_array_equal(physics, source_physics['samples'])\n                assert strict_success(result)\n                assert same_rng(initial_rng, rng())\n            assert np.allclose(arrays['latent_std'][prefix_controls:], .05, atol=1e-8, rtol=0)",
     "            if deterministic:\n                assert same_rng(initial_rng, rng())\n            suffix_std = arrays['latent_std'][prefix_controls:]\n            assert np.isfinite(suffix_std).all() and (suffix_std >= agent.config.min_std * .99999).all() and (suffix_std <= agent.config.max_std * 1.00001).all()"),
    ("soft_return = float(discounts @ suffix_rewards - alpha_before * (discounts[1:] @ suffix_logp[1:]))", "soft_return = float(discounts @ suffix_rewards - alpha_before * protocol['reward_divisor'] * (discounts[1:] @ suffix_logp[1:]))"),
    ("deterministic_full_trace_bitexact=deterministic", "deterministic_suffix_may_differ_from_original=True, physical_reset_seed=protocol['physical_reset_seed']"),
    ("learner_online_updates=0, strict_force_qualified_success=strict_success(result)", "learner_online_updates=agent.online_updates, learner_pretrain_updates=agent.pretrain_updates, source_is_sac_trained=True, assisted_sac_completion=strict_success(result), strict_force_qualified_success=strict_success(result)"),
    ("observed_suffix_soft_return=soft_return, terminal_complete=terminal_complete", "observed_suffix_soft_return_raw_equivalent=soft_return, observed_suffix_soft_return_normalized=soft_return / protocol['reward_divisor'], alpha_normalized=alpha_before, alpha_raw_equivalent=alpha_before * protocol['reward_divisor'], suffix_std_min=float(suffix_std.min()), suffix_std_max=float(suffix_std.max()), terminal_complete=terminal_complete"),
    ("deterministic_full_trace_bitexact_at_every_stage=True", "trained_deterministic_suffix_not_required_to_reproduce_original=True"),
    ("else 'all_seven_stages_completed'", "else 'all_six_stages_completed'"),
]
for old, new in replacements:
    assert old in run, old
    run = run.replace(old, new)
(D / 'run.py').write_text(run)
shutil.copyfile(base / 'verify_sampling.py', D / 'verify_sampling.py')
compile(run, str(D / 'run.py'), 'exec')
compile((D / 'verify_sampling.py').read_text(), str(D / 'verify_sampling.py'), 'exec')
config = json.loads((D / 'source_config.json').read_text())
config['environment']['initial_grasp'] = str(D / 'initial_grasp.npz')
write(D / 'config.json', config)
write(D / 'protocol.json', {
    'purpose': 'Prespecified earlier-prefix stochastic-support screen with final genuinely SAC-trained frozen policy; no learning',
    'prefix_order': [6, 5, 4, 3, 2, 1],
    'stop_after_first_stage_with_fewer_than_stochastic_successes': 4,
    'max_stages': 6,
    'experiments_per_stage': [{'seed':42,'deterministic':True}] + [{'seed':seed,'deterministic':False} for seed in range(17141,17146)],
    'physical_reset_seed': 42,
    'prefix_source': 'Original actual BC-origin nine-control successful initial0005 trace; prefix physically replayed without rebase',
    'learner_source_counters': {'bc_updates':12000,'pretrain_updates':500,'online_updates':2000},
    'learned_state_dependent_std_preserved': True,
    'std_reinitialization': False,
    'reward_divisor': 200.0,
    'alpha_reporting': 'Actual final checkpoint alpha for normalized soft returns; multiply by200 for raw-reward-equivalent entropy weight',
    'physical_horizon_including_prefix': 100,
    'sampling': 'CUDA float32 singleton batch(1,60), actual returned density; seed reset per episode, no sampling draws during physical prefix',
    'strict_success': 'Environment success AND independently force-qualified turn, no drop/palm/external support, actual success termination',
    'prefix_verification': 'Initial observation/qpos/qvel/targets/full integration and every prefix transition/2ms physical sample bit-exact',
    'deterministic_suffix_expected_equal_original': False,
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
    'source_final_manifest_sha256':sha(S / 'manifest.json'),
})
write(D / 'preparation_checks.json', {'source_manifest_bound':True,'all_copied_inputs_source_hash_verified':True,'final_checkpoint_hash_verified':True,'artifact_harness_compiles':True,'no_source_numeric_edit':True,'no_physics_before_freeze':True})
write(D / 'command.json', {'argv':['.venv/bin/python',str(D / 'run.py')], 'cwd':str(D.parents[1]),'environment':{'PYTHONHASHSEED':'0'}})
write(D / 'inputs_manifest.json', {str(path.relative_to(D)):sha(path) for path in sorted(D.rglob('*')) if path.is_file() and '__pycache__' not in path.parts and path != D / 'inputs_manifest.json'})
print(json.dumps({'protocol_sha256':sha(D / 'protocol.json'),'inputs_manifest_sha256':sha(D / 'inputs_manifest.json'),'checkpoint_sha256':sha(D / 'checkpoint.pt')}))
