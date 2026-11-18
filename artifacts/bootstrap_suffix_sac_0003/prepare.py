"""Prepare and freeze the mixed-demonstration prefix-6 SAC stage; never execute."""
from pathlib import Path
import hashlib
import json
import shutil

D = Path(__file__).resolve().parent
S = D.parent / 'bootstrap_suffix_sac_0002'
R = D.parent / 'mpc_prefix6_handoff_recovery_0001'
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
    stage = json.loads((S / 'manifest.json').read_text())
    stage_execution = json.loads((S / 'execution_manifest.json').read_text())
    sources = {name: digest for name, digest in stage_execution.items() if name.startswith('source/')}
    assert sources and all(stage[name] == digest for name, digest in sources.items())
    assert all(sha(S / name) == digest for name, digest in sources.items())
    shutil.copytree(S / 'source', D / 'source', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    copied = {str(p.relative_to(D)) for p in (D / 'source').rglob('*') if p.is_file()}
    assert copied == set(sources) and all(sha(D / name) == digest for name, digest in sources.items())
    bindings['source/'] = dict(source=str(S / 'source'), files=len(sources), inherited_execution_manifest_sha256=sha(S / 'execution_manifest.json'))
    # Inherited learner and prefix evidence: the SAC0001 final checkpoint, unchanged through SAC0002.
    for name in ('initial.pt', 'initial.pt.metadata.json', 'inherited_state.json', 'prior_replay_lineage_only.npz',
                 'source_config.json', 'initial_grasp.npz', 'prefix_transitions.npz', 'prefix_physics.npz', 'prefix_report.json',
                 'warmup_gate.json', 'summary.json', 'protocol.json', 'stage_binding.json', 'input_bindings.json'):
        local = name if name in ('initial.pt', 'initial.pt.metadata.json', 'inherited_state.json', 'prior_replay_lineage_only.npz',
                                 'source_config.json', 'initial_grasp.npz', 'prefix_transitions.npz', 'prefix_physics.npz', 'prefix_report.json') \
            else 'inputs/sac0002/' + name
        copy(S / name, local, stage[name], bindings, inherited_manifest_key=name)
    copy(S / 'findings.md', 'inputs/sac0002/findings.md', sha(S / 'findings.md'), bindings, note='postdates the runner-written manifest')
    copy(S / 'manifest.json', 'inputs/sac0002/manifest.json', sha(S / 'manifest.json'), bindings)
    assert sha(D / 'initial.pt') == 'e4a73d890963a70afc9337db7d29d345c38e192218a7b81dbd51cd6589232892'
    assert stage['checkpoints/warmup_complete/agent.pt'] == sha(S / 'checkpoints/warmup_complete/agent.pt')
    assert sha(D / 'prefix_transitions.npz') == '7b75bef17b559652bc7733b081dc03e4edaf012f46bfd0d11a349b5d4d816d8f'
    assert json.loads((D / 'inputs/sac0002/warmup_gate.json').read_text())['observed_force_successes'] == 0
    inherited = json.loads((D / 'inherited_state.json').read_text())
    assert inherited['learner_counters'] == dict(bc_updates=12000, pretrain_updates=500, online_updates=2000)
    # Off-policy demonstration rows: the executed MPC corrections and the recorded continuation.
    recovery = json.loads((R / 'manifest.json').read_text())
    copy(R / 'actual_mpc_correction_pairs.npz', 'mpc_correction_pairs.npz', recovery['actual_mpc_correction_pairs.npz'], bindings)
    copy(R / 'actual_mpc_correction_pairs.metadata.json', 'mpc_correction_pairs.metadata.json', recovery['actual_mpc_correction_pairs.metadata.json'], bindings)
    copy(R / 'recovery/transitions.npz', 'mpc_recovery_transitions.npz', recovery['recovery/transitions.npz'], bindings)
    for name in ('recovery/independent_audit/report.json', 'final_report.json', 'findings.md', 'protocol.json', 'execution_manifest.json'):
        copy(R / name, 'inputs/mpc_recovery/' + name.replace('/', '_'), recovery[name], bindings)
    copy(R / 'manifest.json', 'inputs/mpc_recovery/manifest.json', sha(R / 'manifest.json'), bindings)
    final = json.loads((D / 'inputs/mpc_recovery/final_report.json').read_text())
    assert final['strict_force_success'] and final['mpc_corrections'] == 5 and final['prefix_controls'] == 6
    assert final['correction_pairs_sha256'] == bindings['mpc_correction_pairs.npz']['sha256']
    assert final['trajectory_sha256'] == bindings['mpc_recovery_transitions.npz']['sha256']
    assert json.loads((D / 'inputs/mpc_recovery/recovery_independent_audit_report.json').read_text())['strict_force_success']
    prefix_report = json.loads((D / 'prefix_report.json').read_text())
    assert prefix_report['finger_supported_rotation_success'] and prefix_report['steps'] == 9
    write(D / 'input_bindings.json', bindings)
    config = json.loads((D / 'source_config.json').read_text())
    config['environment']['initial_grasp'] = str(D / 'initial_grasp.npz')
    write(D / 'config.json', config)
    write(D / 'protocol.json', dict(
        schema_version=1, kind='diagnostic-prefix-assisted-suffix-sac-mixed-demonstration-continuation',
        status='frozen-ready-for-separate-explicit-execution',
        prefix_controls=6, fixed_prefix_stage=True, automatic_stage_advancement=False,
        motivation='SAC0002 found 0/256 frozen-policy successes at prefix 6; mpc_prefix6_handoff_recovery_0001 then produced a verified 5-correction turn from the identical handoff. This stage learns from those off-policy rows plus the recorded continuation, mixed with actual on-policy suffix rows.',
        train_seed=17391, demonstration_seed=17392, physical_reset_seed=42,
        warmup_complete_episodes=256,
        learning_gate='Verified off-policy demonstration rows in the fixed demonstration buffer; warm-up policy successes are recorded but not required.',
        demonstration_rows=8, demonstration_rows_mpc=5, demonstration_rows_recorded=3, demonstration_episodes=2,
        demonstration_sources=dict(mpc='mpc_correction_pairs.npz rows 0-4: actual corrected native MPC controls 7-11 of the recovery episode, terminal success',
                                   recorded='prefix_transitions.npz rows 6-8: recorded source controls 7-9, terminal success'),
        demonstration_exclusions='the six prefix rows of either episode never enter any buffer; no successor action is fabricated; prefix-6 suffix successors within each demonstration are the actual next executed controls',
        offline_fraction=.3, mixing='mpc_dex.replay.sample_mixed with the main-method 30/70 split: 77 demonstration + 179 online rows per 256 batch; demonstration rows are sampled with replacement from only 8 rows (about 9.6 expected draws each per batch)',
        critic_pretrain_updates=500, online_policy_interactions=2000, checkpoints=[500, 1000, 2000],
        inherited_counters=inherited['learner_counters'],
        expected_final_counters=dict(bc_updates=12000, pretrain_updates=1000, online_updates=4000),
        inherited_optimizer_steps=dict(critic=2500, actor=2000, alpha=2000),
        expected_final_optimizer_steps=dict(critic=5000, actor=4000, alpha=4000),
        batch_size=256, utd=1, replay_capacity=100000,
        raw_reward_divisor=200., inherited_learned_alpha=inherited['normalized_alpha'],
        pretrain_alpha=.0001, actor_warmup_updates=5000,
        eval_deterministic_seed=42, eval_phase_stochastic_seeds=list(range(17401, 17406)),
        eval_final_cold_stochastic_seeds=list(range(17411, 17416)),
        retention_prefix_controls=7, retention_stochastic_seeds=list(range(17421, 17426)),
        retention_evaluation_points=['pretraining_baseline', 'online_2000'],
        evaluation_points=['pretraining_baseline', 'online_500', 'online_1000', 'online_2000'],
        physical_horizon_including_prefix=100, policy_interface={'kind': 'absolute_joint_target'},
        learner_import='Full exact network, target, optimizer, counter, learned alpha and learned std import from the SAC0001 final checkpoint; no resets',
        replay='Fresh empty online replay for actual new-stage policy suffix transitions; separate immutable demonstration buffer; SAC0001 replay copied as lineage only',
        prior_open_replay_semantics='Preserve source open row unchanged; never import, finish, discard in source, or connect it to a new reset',
        pretrain='Unchanged SACAgent.pretrain_critic_step on mixed batches; actual next actions for both buffers; sampled raw rewards / 200',
        online='Unchanged SACAgent.update on mixed batches; sampled raw rewards / 200; one update per actual policy step; cumulative LR ramp',
        sampling='Singleton batch (1,60) CUDA float32 actor.sample; direct sampled log probability; new global seed 17391',
        demonstration_calibration='Critic values on the 8 fixed rows versus entropy-free discounted raw returns / 200, plus deterministic actor-mean distance to both control-7 labels and actor log density of demonstration actions, at baseline, after pretraining and at each online checkpoint',
        normalization_caveat='Reward/entropy units remain those of SAC0001; inherited learned alpha is not reset. Mixing ratio, demonstration source and collection seed all change; this is not an isolated causal comparison.',
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
