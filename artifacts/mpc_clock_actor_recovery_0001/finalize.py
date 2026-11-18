"""Read-only numerical verification and artifact packaging; no physics or learning."""
from pathlib import Path
import json
import hashlib
import numpy as np

D = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    assert json.loads((D / 'summary.json').read_text())['status'] == 'complete'
    inputs = json.loads((D / 'inputs_manifest.json').read_text())
    assert all(sha(D / name) == digest for name, digest in inputs.items())
    provenance = json.loads((D / 'input_provenance.json').read_text())
    assert all(sha(D / name) == item['sha256'] == sha(item['source']) for name, item in provenance.items())
    folder = D / 'recovery'
    report = json.loads((folder / 'report.json').read_text())
    audit = json.loads((folder / 'independent_audit/report.json').read_text())
    trace = load(folder / 'transitions.npz')
    replay = load(folder / 'independent_audit/transitions.npz')
    source = load(D / 'inputs/clock_actor/independent_audit/transitions.npz')
    checks = {}
    for key in source:
        expected = source[key] if key.startswith('initial_') else source[key][:10 if key == 'control_times' else 9]
        observed = trace[key] if key.startswith('initial_') else trace[key][:10 if key == 'control_times' else 9]
        checks[key] = bool(np.array_equal(expected, observed))
    assert all(checks.values())
    assert 'integration_state' not in source and 'initial_integration_state' not in source
    assert all(np.array_equal(trace[key], replay[key]) for key in trace)
    physics = load(folder / 'physics.npz')
    assert np.array_equal(physics['samples'], load(folder / 'independent_audit/physics.npz')['samples'])
    assert np.array_equal(physics['samples'][:450],
                          load(D / 'inputs/clock_actor/independent_audit/physics.npz')['samples'][:450])
    steps = len(trace['actions'])
    assert len(physics['samples']) == 50*steps
    assert trace['joint_positions'].shape == (steps + 1, 16)
    assert np.array_equal(trace['obs'][:, :60], trace['environment_obs'])
    assert np.array_equal(trace['next_obs'][:, :60], trace['environment_next_obs'])
    assert np.array_equal(trace['obs'][:, 60], trace['decision_times'].astype(np.float32))
    assert np.array_equal(trace['next_obs'][:, 60], trace['control_times'][1:].astype(np.float32))
    col = {name: physics['samples'][:, index] for index, name in enumerate(physics['columns'])}
    valid = ((col['raw_angle'] >= 2*np.pi) & (col['force_qualified_angle'] >= 2*np.pi)
             & (col['force_retained'] > 0) & (col['palm_support_violation'] == 0))
    first = np.flatnonzero(valid)
    fields = ('obs', 'actions', 'next_obs', 'rewards', 'terminated', 'truncated', 'qpos', 'qvel',
              'targets', 'integration_state', 'environment_obs', 'environment_next_obs', 'decision_times')
    pairs = {key: trace[key][9:] for key in fields}
    pairs.update(actual_post_control_times=trace['control_times'][10:],
                 source_control_1based=np.arange(10, steps + 1))
    assert np.array_equal(pairs['actions'], np.clip(pairs['actions'], -.98, .98))
    mask = load(folder / 'control_origins.npz')['expert_correction_mask']
    assert np.array_equal(mask, np.arange(steps) >= 9)
    np.savez_compressed(D / 'actual_mpc_correction_pairs.npz', **pairs)
    serialized = load(D / 'actual_mpc_correction_pairs.npz')
    assert all(np.array_equal(serialized[key], pairs[key]) for key in pairs)
    projected = [item for item in report['reference_records'] if item['projected']]
    pair_metadata = dict(kind='diagnostic-actual-feedback-state-mpc-correction-supervised-pairs',
        rows=steps - 9, actor_prefix_rows_excluded=9,
        data_sha256=sha(D / 'actual_mpc_correction_pairs.npz'), source_trajectory='recovery/transitions.npz',
        source_trajectory_sha256=sha(folder / 'transitions.npz'),
        observation_dimension=61, raw_environment_observation_dimension=60,
        clock='float32 actual reset-relative decision elapsed seconds; observed, not synthetic step timestamps',
        policy_interface=json.loads((D / 'inputs/clock_actor/policy_interface.json').read_text()),
        no_next_actions_fabricated=True, not_a_cold_reset_replay_dataset=True,
        source_actor_checkpoint_sha256=sha(D / 'inputs/clock_actor/candidate.pt'),
        source_actor_trace_sha256=sha(D / 'inputs/clock_actor/independent_audit/transitions.npz'),
        production_eligible=False, main_demonstrations=0, SAC_successes=0, learning_updates=0,
        row_provenance=[dict(row=index - 9, source_transition_row_0based=index,
            source_control_1based=index + 1, origin='actually_executed_corrected_native_mpc')
            for index in range(9, steps)])
    write(D / 'actual_mpc_correction_pairs.metadata.json', pair_metadata)
    final = dict(status='complete', strict_force_success=audit['strict_force_success'],
        source_actor_checkpoint_sha256=sha(D / 'inputs/clock_actor/candidate.pt'),
        source_actor_trace_sha256=sha(D / 'inputs/clock_actor/independent_audit/transitions.npz'),
        source_actor_manifest_sha256=sha(D / 'inputs/clock_actor/manifest.json'),
        source_prefix_field_checks=checks, source_prefix_physics_bitexact=True,
        original_source_integration_state_available=False,
        new_full_integration_and_all_transition_fields_independently_replayed_bitexact=True,
        measured_clock_verified=True, actor_prefix_controls=9, mpc_corrections=steps - 9,
        total_controls=steps, physics_samples=len(physics['samples']),
        handoff_seconds=float(trace['control_times'][9]),
        handoff_force_qualified_angle=report['branch_force_qualified_angle'],
        final_seconds=float(trace['control_times'][-1]), final_raw_angle=audit['final_raw_angle'],
        final_retained_angle=audit['final_qualified_angle'], final_force_qualified_angle=audit['final_force_qualified_angle'],
        maximum_force_qualified_angle=audit['max_force_qualified_angle'],
        first_force_fullturn_physics_seconds=float(col['time'][first[0]]) if len(first) else None,
        final_loaded_fingers=audit['final_force_bearing_fingers'], final_axis_tilt_radians=audit['final_axis_tilt_radians'],
        independent_drop=audit['independent_drop'], palm_support=audit['independent_palm_support_violation'],
        no_external_support=audit['no_external_support'], projected_reference_batches=len(projected),
        maximum_reference_projection_radians=max((item['projection_max_radians'] for item in projected), default=0.),
        live_physics_never_projected_or_reset_at_handoff=True, correction_epsilon_002_clamp_unchanged=True,
        total_planning_seconds=sum(item.get('planning_seconds', 0.) for item in report['commands']),
        wall_seconds=report['wall_seconds'], no_post_success_continuation_or_hold=True,
        no_alternate_prefix_or_parameter_sweep=True, production_eligible=False, main_demonstrations=0,
        SAC_successes=0, learning_updates=0, copied_and_original_input_hashes_unchanged=True,
        trajectory_sha256=sha(folder / 'transitions.npz'), physics_sha256=sha(folder / 'physics.npz'),
        independent_audit_sha256=sha(folder / 'independent_audit/report.json'),
        correction_pairs_sha256=sha(D / 'actual_mpc_correction_pairs.npz'),
        correction_pairs_metadata_sha256=sha(D / 'actual_mpc_correction_pairs.metadata.json'))
    write(D / 'final_report.json', final)
    text = f'''# One fixed clock-actor-state MPC recovery

**The prescribed handoff succeeds after {steps - 9} actual MPC corrections.** The complete actor-prefix-plus-MPC episode terminates at {final['final_seconds']:.1f} seconds after {steps} controls, with {audit['final_force_qualified_angle']:.9f} rad of independent force-qualified rotation, {audit['final_force_bearing_fingers']} loaded fingers, {audit['final_axis_tilt_radians']:.6f} rad tilt and no palm support, external support or drop. First physics qualification occurs at {final['first_force_fullturn_physics_seconds']:.3f} seconds. This is assisted feedback recovery, not a successful learned actor or SAC policy.

The original polished 61D clock actor's actual policy-only rollout dropped after 19 controls, with peak force-qualified rotation 6.060199 rad. This one experiment cold-resets the identical strict 40 mm physical environment, replays exactly its first nine recorded controls and hands control to MPC at .9 seconds and {report['branch_force_qualified_angle']:.9f} force-qualified rad. Every available source field is bit-identical: 61D observations/next observations, raw 60D environment observations, actual measured decision/control times, actions, rewards, terminal flags, qpos/qvel, targets, initial states, and all 450 prefix physics samples. The source audit did not save full integration states, so no comparison to missing original integration data is claimed. New full integration states are recorded throughout the recovery and match a separate independent physical replay, along with every other new transition field and all {len(physics['samples'])} physics samples.

The planner uses the same approved measured 40 mm PCA prior and corrected native implementation as the five-noisy-state recovery protocol: eight workers, 350 candidates, ten elites, three iterations, six knots, .5-second horizon, .1-second controls, retained progress, perpendicular weight 100, position weight 1000, principal weight 1 and fixed CEM seed 42. The .98 search restriction, warm nominal and bounded-target reference candidates are unchanged. At most twenty corrections were allowed, with no alternate handoff or parameter choice. Only a planning reference is projected when the current actual target lies outside the search bounds; live physics, current target and accumulated rotation are untouched. The maximum reference projection was {final['maximum_reference_projection_radians']:.9g} rad. Every planning call leaves live integration state unchanged.

`actual_mpc_correction_pairs.npz` contains exactly {steps - 9} actual MPC rows with their 61D clock observations, raw 60D observations, executed actions, actual next observations, rewards, flags, physical/integration states and measured times. The nine actor-prefix commands are excluded as expert labels. The sidecar preserves exact source row indices and hashes; no successor action is invented. Every MPC correction action is bitwise unchanged by epsilon .02 clipping. These are diagnostic supervised pairs from visited mid-episode states, not a cold-reset replay dataset or main-stage demonstrations.

The clock is recorded from actual elapsed simulator time; it is not used by MPC, and no environment/physics/production observation source changed. No BC or SAC learning occurred. No hold or further policy continuation was performed after the first qualified turn. This establishes one recoverable actual actor-visited state and supplies five corrective labels; whether another imitation round can use them or retain the object afterward remains untested. Source checkpoint, complete original failed trace, all inputs, frozen numerical sources, native build provenance, commands, runtime, branch state, new execution and independent replay are archived and hash-verified.
'''
    (D / 'findings.md').write_text(text)
    write(D / 'manifest.json', {str(path.relative_to(D)): sha(path) for path in sorted(D.rglob('*'))
        if path.is_file() and '__pycache__' not in path.parts and path != D / 'manifest.json'})
    print(json.dumps(dict(manifest_sha256=sha(D / 'manifest.json'),
        final_report_sha256=sha(D / 'final_report.json'), findings_sha256=sha(D / 'findings.md'),
        trajectory_sha256=final['trajectory_sha256'], pairs_sha256=final['correction_pairs_sha256'],
        pairs_metadata_sha256=final['correction_pairs_metadata_sha256']), indent=2))


if __name__ == '__main__':
    main()
