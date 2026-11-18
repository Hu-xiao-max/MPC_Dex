"""Prepare a new curriculum stage from full inherited SAC state; never execute."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

D = Path(__file__).resolve().parent
S = D.parent / 'bootstrap_suffix_sac_0001'
SCREEN = D.parent / 'bootstrap_suffix_sac_rollback_0001'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--finalize-prefix', type=int)
    args = parser.parse_args()
    if (D / 'execution_manifest.json').exists() or (D / 'runtime.json').exists():
        raise SystemExit('This stage is already frozen or started')
    if not (D / 'input_bindings.json').exists():
        source_manifest = json.loads((S / 'manifest.json').read_text())
        assert sha(S / 'manifest.json') == '258842818fa0087aa70566996fad768fa088c1e6e3102d533c1c40f3e48edfc0'
        inherited_sources = {name: digest for name, digest in source_manifest.items() if name.startswith('source/')}
        assert inherited_sources and all(sha(S / name) == digest for name, digest in inherited_sources.items())
        shutil.copytree(S / 'source', D / 'source', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        copied_sources = {str(p.relative_to(D)) for p in (D / 'source').rglob('*') if p.is_file()}
        assert copied_sources == set(inherited_sources)
        assert all(sha(D / name) == digest for name, digest in inherited_sources.items())
        bindings = {}
        for name, relative in {
            'initial.pt': 'checkpoints/online_2000/agent.pt',
            'initial.pt.metadata.json': 'checkpoints/online_2000/agent.pt.metadata.json',
            'inherited_state.json': 'checkpoints/online_2000/state.json',
            'prior_replay_lineage_only.npz': 'checkpoints/online_2000/replay.npz',
            'source_config.json': 'config.json',
            'initial_grasp.npz': 'initial_grasp.npz',
            'source_summary.json': 'summary.json',
            'source_protocol.json': 'protocol.json',
            'prefix_transitions.npz': 'prefix_transitions.npz',
            'prefix_physics.npz': 'prefix_physics.npz',
            'prefix_report.json': 'prefix_report.json',
        }.items():
            original = S / relative
            assert sha(original) == source_manifest[relative]
            shutil.copyfile(original, D / name)
            bindings[name] = dict(source=str(original), sha256=sha(original), inherited_manifest_key=relative)
        shutil.copyfile(S / 'manifest.json', D / 'source_manifest.json')
        bindings['source_manifest.json'] = dict(source=str(S / 'manifest.json'), sha256=sha(S / 'manifest.json'))
        write(D / 'input_bindings.json', bindings)
        config = json.loads((D / 'source_config.json').read_text())
        config['environment']['initial_grasp'] = str(D / 'initial_grasp.npz')
        write(D / 'config.json', config)
        inherited = json.loads((D / 'inherited_state.json').read_text())
        assert inherited['learner_counters'] == dict(bc_updates=12000, pretrain_updates=500, online_updates=2000)
        assert inherited['unfinished_episode'] and inherited['replay_rows'] == 2808
        write(D / 'protocol.json', dict(
            schema_version=1, kind='diagnostic-prefix-assisted-suffix-sac-continuation',
            status='prepared-awaiting-selected-stage-and-root-review',
            prefix_controls=None, fixed_prefix_stage=True, automatic_stage_advancement=False,
            train_seed=17091, physical_reset_seed=42,
            warmup_complete_episodes=256, minimum_warmup_force_successes_to_train=1,
            warmup_gate='Require >=1 actual strict force-qualified success in these 256 episodes, irrespective of five-screen outcomes; otherwise finish data-only before learning.',
            critic_pretrain_updates=500, online_policy_interactions=2000, checkpoints=[500, 1000, 2000],
            inherited_counters=inherited['learner_counters'],
            expected_final_counters=dict(bc_updates=12000, pretrain_updates=1000, online_updates=4000),
            inherited_optimizer_steps=dict(critic=2500, actor=2000, alpha=2000),
            expected_final_optimizer_steps=dict(critic=5000, actor=4000, alpha=4000),
            batch_size=256, utd=1, replay_capacity=100000,
            raw_reward_divisor=200.0, inherited_learned_alpha=inherited['normalized_alpha'],
            pretrain_alpha=.0001, actor_warmup_updates=5000,
            eval_deterministic_seed=42, eval_phase_stochastic_seeds=list(range(17201, 17206)),
            eval_final_cold_stochastic_seeds=list(range(17211, 17216)),
            retention_prefix_controls=7, retention_stochastic_seeds=list(range(17221, 17226)),
            retention_evaluation_points=['pretraining_baseline', 'online_2000'],
            evaluation_points=['pretraining_baseline', 'online_500', 'online_1000', 'online_2000'],
            physical_horizon_including_prefix=100, policy_interface={'kind': 'absolute_joint_target'},
            learner_import='Full exact network, target, optimizer, counter, learned alpha and learned std import; no resets',
            replay='Fresh empty stage-specific replay; only actual new-stage policy suffix transitions; old replay copied as lineage only',
            prior_open_replay_semantics='Preserve source open row unchanged; never import, finish, discard in source, or connect it to a new reset',
            replay_tradeoff='Stage-specific frozen-policy successors suit SoftSARSA and avoid dominance by prior-stage rows; retention evaluation measures possible forgetting.',
            pretrain='Unchanged SACAgent.pretrain_critic_step; actual next actions; sampled raw rewards / 200',
            online='Unchanged SACAgent.update; sampled raw rewards / 200; one update per actual policy step; cumulative LR ramp',
            sampling='Singleton batch (1,60) CUDA float32 actor.sample; direct sampled log probability; new global seed17091',
            normalization_caveat='Reward/entropy units remain those of SAC0001; inherited learned alpha is not reset. This changes stage, replay distribution and collection seed, not a pure isolated causal comparison.',
            complete_episode_definition='Environment terminated or horizon truncated; any truncated return tail remains unknown',
            warmup_alpha_for_calibration=inherited['normalized_alpha'],
            pretrain_calibration_alpha_caveat='Report inherited learned online alpha separately from unchanged .0001 SoftSARSA objective.',
            no_prefix_rows_in_replay=True, no_trajectory_export=True,
            eligible_for_primitive=False, main_mpc_dataset_eligible=False,
            new_bc_updates=0, head_refinement=False, extra_epochs=False,
            post_success_continuation=False, resume_implementation=False,
            checkpoint_semantics='New curriculum stage with fresh reset/replay/RNG; full learner continuation, not exact environment resume',
            future_checkpoint_use='A later trained 60D checkpoint requires a new frozen cold-policy collection; assisted episodes are never primitive evidence.',
            source_bindings=bindings))
    bindings = json.loads((D / 'input_bindings.json').read_text())
    assert all(sha(D / name) == row['sha256'] for name, row in bindings.items())
    assert sha(D / 'source_manifest.json') == '258842818fa0087aa70566996fad768fa088c1e6e3102d533c1c40f3e48edfc0'
    inherited_sources = {name: digest for name, digest in json.loads((D / 'source_manifest.json').read_text()).items()
                         if name.startswith('source/')}
    copied_sources = {str(p.relative_to(D)) for p in (D / 'source').rglob('*')
                      if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    assert copied_sources == set(inherited_sources)
    assert all(sha(D / name) == digest for name, digest in inherited_sources.items())
    if args.finalize_prefix is not None:
        if not 1 <= args.finalize_prefix <= 6:
            raise SystemExit('This stage must be selected from the fixed descending prefix6..1 screen')
        if not (D / 'stage_binding.json').exists():
            raise SystemExit('Final reviewed screen binding is required')
        binding = json.loads((D / 'stage_binding.json').read_text())
        if binding['selected_prefix_controls'] != args.finalize_prefix:
            raise SystemExit('Selected stage differs from finalized screen recommendation')
        assert binding['screen_manifest_sha256'] == 'c3a10cea3d4f509d51e40e3965f76119df5e2ff86274cf694dbd7d0944c12ad8'
        assert all(sha(D / record['local_path']) == record['sha256'] for record in binding['files'].values())
        assert sha(D / binding['files']['manifest.json']['local_path']) == binding['screen_manifest_sha256']
        assert binding['screen_checkpoint_sha256'] == sha(D / 'initial.pt')
        assert binding['original_prefix_transitions_sha256'] == sha(D / 'prefix_transitions.npz')
        protocol = json.loads((D / 'protocol.json').read_text())
        protocol.update(prefix_controls=args.finalize_prefix, status='frozen-ready-for-separate-explicit-execution', stage_binding=binding)
        write(D / 'protocol.json', protocol)
        write(D / 'execution_manifest.json', {str(p.relative_to(D)): sha(p) for p in sorted(D.rglob('*'))
              if p.is_file() and '__pycache__' not in p.parts and p.name not in ('execution_manifest.json', 'run.log')})
    else:
        write(D / 'preparation_manifest.json', {str(p.relative_to(D)): sha(p) for p in sorted(D.rglob('*'))
              if p.is_file() and '__pycache__' not in p.parts and p.name not in ('preparation_manifest.json', 'run.log')})
    print(json.dumps(dict(prepared=True, frozen_prefix=args.finalize_prefix, executed=False, source_checkpoint_sha256=sha(D / 'initial.pt'))))


if __name__ == '__main__':
    main()
