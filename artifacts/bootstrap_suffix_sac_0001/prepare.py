"""Snapshot inputs without constructing a learner or executing physics.

Run once without arguments for review; --finalize-prefix N freezes the selected
protocol after review. Neither operation launches the numerical experiment.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

D = Path(__file__).resolve().parent
S = D.parent / 'bootstrap_0005/value_calibration/initial_0001'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--finalize-prefix', type=int)
    args = parser.parse_args()
    if (D / 'execution_manifest.json').exists() or (D / 'runtime.json').exists():
        raise SystemExit('Already frozen or started; do not overwrite this generation')
    if not (D / 'input_bindings.json').exists():
        shutil.copytree(S / 'source', D / 'source', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        bindings = {}
        for name, original in {
            'initial.pt': S / 'checkpoint.pt',
            'initial.pt.metadata.json': S / 'checkpoint.pt.metadata.json',
            'source_config.json': S / 'config.json',
            'initial_grasp.npz': S / 'initial_grasp.npz',
            'source_report.json': S / 'report.json',
            'prefix_transitions.npz': S / 'deterministic_strict/transitions.npz',
            'prefix_physics.npz': S / 'deterministic_strict/physics.npz',
            'prefix_report.json': S / 'deterministic_strict/report.json',
        }.items():
            shutil.copyfile(original, D / name)
            bindings[name] = dict(source=str(original), sha256=sha(original))
            assert sha(D / name) == sha(original)
        write(D / 'input_bindings.json', bindings)
        config = json.loads((D / 'source_config.json').read_text())
        config['environment']['initial_grasp'] = str(D / 'initial_grasp.npz')
        write(D / 'config.json', config)
        write(D / 'protocol.json', dict(
            schema_version=1, kind='diagnostic-prefix-assisted-suffix-sac',
            status='prepared-awaiting-prefix-and-root-execution-authorization',
            prefix_controls=None, fixed_prefix_stage=True, automatic_stage_advancement=False,
            train_seed=17081, physical_reset_seed=42,
            warmup_complete_episodes=256, critic_pretrain_updates=500,
            online_policy_interactions=2000, checkpoints=[500, 1000, 2000],
            batch_size=256, utd=1, replay_capacity=100000,
            raw_reward_divisor=200.0, initial_alpha=.05 / 200,
            pretrain_alpha=.02 / 200, initial_std=.05, critic_lr=3e-4,
            actor_lr=3e-4, actor_warmup_updates=5000,
            eval_deterministic_seed=42, eval_phase_stochastic_seeds=list(range(17101, 17106)),
            eval_final_cold_stochastic_seeds=list(range(17111, 17116)),
            retention_prefix_controls=8, retention_stochastic_seeds=list(range(17121, 17126)),
            retention_evaluation_points=['pretraining_baseline', 'online_2000'],
            evaluation_points=['pretraining_baseline', 'online_500', 'online_1000', 'online_2000'],
            physical_horizon_including_prefix=100,
            policy_interface={'kind': 'absolute_joint_target'},
            replay='only executed policy suffix transitions; raw rewards; uniform transition sampling',
            pretrain='unchanged SACAgent.pretrain_critic_step; actual next actions; sampled rewards / 200',
            online='unchanged SACAgent.update; sampled rewards / 200; one update per actual policy step',
            sampling='singleton batch (1,60) CUDA float32 actor.sample; direct returned log probability',
            normalization_caveat='Curriculum and reward/temperature scaling both change; not a pure curriculum causal comparison. Alpha scaling preserves intended relative reward/entropy weight, not bit-exact algorithm invariance.',
            complete_episode_definition='environment terminated or horizon truncated; truncated return tail remains unknown',
            warmup_alpha_for_calibration=.05 / 200,
            pretrain_calibration_alpha_caveat='Empirical calibration uses frozen initial alpha; SoftSARSA uses separately specified .02/200',
            no_prefix_rows_in_replay=True, no_trajectory_export=True,
            eligible_for_primitive=False, main_mpc_dataset_eligible=False,
            new_bc_updates=0, head_refinement=False, extra_epochs=False,
            post_success_continuation=False, resume_implementation=False,
            checkpoint_semantics='Exact evidence snapshot at completed optimizer boundary; no automatic resume implementation',
            future_checkpoint_use='Any later genuinely trained 60D checkpoint requires a new frozen cold-policy collection; assisted episode evidence is never exported as a primitive source.',
            source_bindings=bindings))
    bindings = json.loads((D / 'input_bindings.json').read_text())
    assert all(sha(D / name) == row['sha256'] for name, row in bindings.items())
    if args.finalize_prefix is not None:
        if not 0 <= args.finalize_prefix <= 8:
            raise SystemExit('Prefix must be an actual pre-success boundary 0..8')
        protocol = json.loads((D / 'protocol.json').read_text())
        protocol['prefix_controls'] = args.finalize_prefix
        if args.finalize_prefix != 7:
            raise SystemExit('Root selected the single fixed prefix-7 stage for this generation')
        if not (D / 'rollback_binding.json').exists():
            raise SystemExit('Finalized rollback report/hash binding is required before freeze')
        protocol['rollback_binding'] = json.loads((D / 'rollback_binding.json').read_text())
        protocol['status'] = 'frozen-ready-for-separate-explicit-execution'
        write(D / 'protocol.json', protocol)
        files = {str(p.relative_to(D)): sha(p) for p in sorted(D.rglob('*'))
                 if p.is_file() and '__pycache__' not in p.parts and p.name not in ('execution_manifest.json', 'run.log')}
        write(D / 'execution_manifest.json', files)
    else:
        write(D / 'preparation_manifest.json', {str(p.relative_to(D)): sha(p) for p in sorted(D.rglob('*'))
              if p.is_file() and '__pycache__' not in p.parts and p.name not in ('preparation_manifest.json', 'run.log')})
    print(json.dumps(dict(prepared=True, prefix_finalized=args.finalize_prefix,
                         executed=False, initial_sha256=sha(D / 'initial.pt'))))


if __name__ == '__main__':
    main()
