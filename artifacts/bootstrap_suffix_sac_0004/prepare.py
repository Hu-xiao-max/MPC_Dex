"""Prepare and freeze the entropy-free-demonstration prefix-6 SAC stage; never execute."""
from pathlib import Path
import hashlib
import json
import shutil

D = Path(__file__).resolve().parent
P = D.parent / 'bootstrap_suffix_sac_0003'
EXCLUDED = ('execution_manifest.json', 'run.log')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def copy(source, local, expected, bindings, **extra):
    assert sha(source) == expected, source
    (D / local).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, D / local)
    bindings[local] = dict(source=str(source), sha256=expected, **extra)


def main():
    if (D / 'execution_manifest.json').exists() or (D / 'runtime.json').exists():
        raise SystemExit('This stage is already frozen or started')
    bindings = {}
    prior_execution = json.loads((P / 'execution_manifest.json').read_text())
    prior_manifest = json.loads((P / 'manifest.json').read_text())
    # The runner-written manifest omits nested files named manifest.json; hash those directly.
    assert all(prior_manifest.get(name, sha(P / name)) == digest for name, digest in prior_execution.items())
    sources = {name: digest for name, digest in prior_execution.items() if name.startswith('source/')}
    assert sources and all(sha(P / name) == digest for name, digest in sources.items())
    shutil.copytree(P / 'source', D / 'source', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    copied = {str(p.relative_to(D)) for p in (D / 'source').rglob('*') if p.is_file()}
    assert copied == set(sources) and all(sha(D / name) == digest for name, digest in sources.items())
    bindings['source/'] = dict(source=str(P / 'source'), files=len(sources), inherited_execution_manifest_sha256=sha(P / 'execution_manifest.json'))
    # Identical inherited learner, prefix evidence and demonstration rows as SAC0003, hash-bound through its freeze.
    for name in ('initial.pt', 'initial.pt.metadata.json', 'inherited_state.json', 'prior_replay_lineage_only.npz',
                 'source_config.json', 'initial_grasp.npz', 'prefix_transitions.npz', 'prefix_physics.npz', 'prefix_report.json',
                 'mpc_correction_pairs.npz', 'mpc_correction_pairs.metadata.json', 'mpc_recovery_transitions.npz', 'input_bindings.json'):
        local = name if name != 'input_bindings.json' else 'inputs/sac0003/input_bindings.json'
        copy(P / name, local, prior_execution[name], bindings, inherited_from='bootstrap_suffix_sac_0003')
    for name in ('summary.json', 'protocol.json', 'warmup_record.json', 'demonstration_buffer.json',
                 'demonstration_calibration_before.json', 'demonstration_calibration_after_pretraining.json',
                 'demonstration_calibration_online_2000.json', 'evaluations/online_2000/summary.json'):
        copy(P / name, 'inputs/sac0003/' + name.replace('/', '_'), prior_manifest[name], bindings)
    copy(P / 'findings.md', 'inputs/sac0003/findings.md', prior_manifest['findings.md'], bindings)
    copy(P / 'manifest.json', 'inputs/sac0003/manifest.json', sha(P / 'manifest.json'), bindings)
    assert sha(D / 'initial.pt') == 'e4a73d890963a70afc9337db7d29d345c38e192218a7b81dbd51cd6589232892'
    assert sha(D / 'prefix_transitions.npz') == '7b75bef17b559652bc7733b081dc03e4edaf012f46bfd0d11a349b5d4d816d8f'
    inherited = json.loads((D / 'inherited_state.json').read_text())
    assert inherited['learner_counters'] == dict(bc_updates=12000, pretrain_updates=500, online_updates=2000)
    prior_summary = json.loads((D / 'inputs/sac0003/summary.json').read_text())
    assert prior_summary['status'] == 'complete_sac_stage' and prior_summary['counts']['online_assisted_force_successes'] == 2
    prior_final = json.loads((D / 'inputs/sac0003/evaluations_online_2000_summary.json').read_text())
    assert not any(r['finger_supported_rotation_success'] for r in prior_final['reports'])
    distortion = json.loads((D / 'inputs/sac0003/demonstration_calibration_after_pretraining.json').read_text())
    assert distortion['episodes']['mpc']['critic_normalized_bias'][0] > 10
    write(D / 'input_bindings.json', bindings)
    config = json.loads((D / 'source_config.json').read_text())
    config['environment']['initial_grasp'] = str(D / 'initial_grasp.npz')
    write(D / 'config.json', config)
    write(D / 'protocol.json', dict(
        schema_version=1, kind='diagnostic-prefix-assisted-suffix-sac-mixed-demonstration-entropy-free-continuation',
        status='frozen-ready-for-separate-explicit-execution',
        prefix_controls=6, fixed_prefix_stage=True, automatic_stage_advancement=False,
        motivation='SAC0003 mixed the same 8 demonstration rows with the unchanged SoftSARSA target; the entropy term with recorded MPC next actions of actor log density down to -155331 inflated their critic targets (29.4 versus 1.08) and prefix-7 retention fell from 6/6 to 0/6. This stage repeats SAC0003 from the same SAC0001 checkpoint with one objective change.',
        objective_change='Critic pretraining computes the same Soft-SARSA target for on-policy rows, but demonstration rows bootstrap entropy-free: target = r + discount * min target Q(next_obs, recorded next action), with no -alpha*log pi term. Online SAC updates are the unchanged SACAgent.update, which samples next actions from the current actor for every row.',
        objective_implementation='runner-local entropy_masked_pretrain_step using the archived agent\'s _batch, _critic_step and _polyak; pretrain_updates counter incremented identically; archived learner source unchanged',
        differences_from_sac0003=['demonstration rows entropy-free in critic pretraining', 'prefix-7 retention evaluated at every checkpoint, not only baseline and final', 'new sampling seeds'],
        train_seed=17491, demonstration_seed=17492, physical_reset_seed=42,
        warmup_complete_episodes=256,
        learning_gate='Verified off-policy demonstration rows in the fixed demonstration buffer; warm-up policy successes are recorded but not required.',
        demonstration_rows=8, demonstration_rows_mpc=5, demonstration_rows_recorded=3, demonstration_episodes=2,
        demonstration_sources=dict(mpc='mpc_correction_pairs.npz rows 0-4: actual corrected native MPC controls 7-11 of the recovery episode, terminal success',
                                   recorded='prefix_transitions.npz rows 6-8: recorded source controls 7-9, terminal success'),
        demonstration_exclusions='the six prefix rows of either episode never enter any buffer; no successor action is fabricated; successors within each demonstration are the actual next executed controls',
        offline_fraction=.3, mixing='runner mixed_batch_with_origin reproducing mpc_dex.replay.sample_mixed draw order (demonstration then online, online-rng permutation) and adding a per-row origin mask: 77 demonstration + 179 online rows per 256 batch',
        critic_pretrain_updates=500, online_policy_interactions=2000, checkpoints=[500, 1000, 2000],
        inherited_counters=inherited['learner_counters'],
        expected_final_counters=dict(bc_updates=12000, pretrain_updates=1000, online_updates=4000),
        inherited_optimizer_steps=dict(critic=2500, actor=2000, alpha=2000),
        expected_final_optimizer_steps=dict(critic=5000, actor=4000, alpha=4000),
        batch_size=256, utd=1, replay_capacity=100000,
        raw_reward_divisor=200., inherited_learned_alpha=inherited['normalized_alpha'],
        pretrain_alpha=.0001, actor_warmup_updates=5000,
        eval_deterministic_seed=42, eval_phase_stochastic_seeds=list(range(17501, 17506)),
        eval_final_cold_stochastic_seeds=list(range(17511, 17516)),
        retention_prefix_controls=7, retention_stochastic_seeds=list(range(17521, 17526)),
        retention_evaluation_points=['pretraining_baseline', 'online_500', 'online_1000', 'online_2000'],
        retention_hard_stop=False,
        evaluation_points=['pretraining_baseline', 'online_500', 'online_1000', 'online_2000'],
        physical_horizon_including_prefix=100, policy_interface={'kind': 'absolute_joint_target'},
        learner_import='Full exact network, target, optimizer, counter, learned alpha and learned std import from the SAC0001 final checkpoint; no resets; SAC0003 checkpoints are not used',
        replay='Fresh empty online replay for actual new-stage policy suffix transitions; separate immutable demonstration buffer; SAC0001 replay copied as lineage only',
        prior_open_replay_semantics='Preserve source open row unchanged; never import, finish, discard in source, or connect it to a new reset',
        pretrain='entropy_masked_pretrain_step on mixed batches: on-policy rows unchanged SoftSARSA with pretrain alpha .0001; demonstration rows entropy-free; sampled raw rewards / 200',
        online='Unchanged SACAgent.update on mixed batches; sampled raw rewards / 200; one update per actual policy step; cumulative LR ramp',
        sampling='Singleton batch (1,60) CUDA float32 actor.sample; direct sampled log probability; new global seed 17491',
        demonstration_calibration='Critic values on the 8 fixed rows versus entropy-free discounted raw returns / 200 (now the pretraining target for those rows), deterministic actor-mean distance to both control-7 labels, actor log density of demonstration actions, at baseline, after pretraining and at each online checkpoint',
        normalization_caveat='Reward/entropy units remain those of SAC0001; inherited learned alpha is not reset. The 30/70 repetition of 8 rows and the changed reset curriculum remain confounds; only the pretraining target changed relative to SAC0003.',
        complete_episode_definition='Environment terminated or horizon truncated; any truncated return tail remains unknown',
        warmup_alpha_for_calibration=inherited['normalized_alpha'],
        pretrain_calibration_alpha_caveat='Report inherited learned online alpha separately from unchanged .0001 SoftSARSA objective.',
        no_prefix_rows_in_replay=True, no_trajectory_export=True,
        eligible_for_primitive=False, main_mpc_dataset_eligible=False,
        new_bc_updates=0, head_refinement=False, extra_epochs=False,
        post_success_continuation=False, resume_implementation=False,
        checkpoint_semantics='New curriculum stage with fresh reset/replay/RNG; full learner continuation, not exact environment resume',
        future_checkpoint_use='A later trained 60D checkpoint requires a new frozen cold-policy collection; assisted and demonstration-mixed episodes are never primitive evidence.',
        source_bindings=bindings))
    write(D / 'execution_manifest.json', {str(p.relative_to(D)): sha(p) for p in sorted(D.rglob('*'))
          if p.is_file() and '__pycache__' not in p.parts and p.name not in EXCLUDED})
    print(json.dumps(dict(prepared=True, frozen=True, executed=False, source_checkpoint_sha256=sha(D / 'initial.pt'),
                          execution_manifest_sha256=sha(D / 'execution_manifest.json'))))


if __name__ == '__main__':
    main()
