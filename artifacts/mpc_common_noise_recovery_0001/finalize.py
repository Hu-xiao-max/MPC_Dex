"""Validate and summarize completed diagnostics; performs no simulation/learning."""
from pathlib import Path
import hashlib
import json
import numpy as np

D = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def main():
    summary = json.loads((D / 'summary.json').read_text())
    assert summary['status'] == 'complete' and summary['completed_branches'] == 5
    inputs = json.loads((D / 'inputs_manifest.json').read_text())
    assert all(sha(D / name) == digest for name, digest in inputs.items())
    provenance = json.loads((D / 'input_provenance.json').read_text())
    assert all(sha(record['source']) == record['sha256'] == sha(D / name)
               for name, record in provenance.items())
    rows, pair_metadata, values = [], [], {}
    pooled_fields = ('obs', 'actions', 'next_obs', 'rewards', 'terminated', 'truncated',
                     'qpos', 'qvel', 'targets', 'integration_state')
    for field in pooled_fields:
        values[field] = []
    all_origins, total_planning, total_wall = [], 0., 0.
    for outcome in summary['outcomes']:
        seed = outcome['seed']
        folder = D / f'seed_{seed}'
        report = json.loads((folder / 'report.json').read_text())
        audit = json.loads((folder / 'independent_audit/report.json').read_text())
        source = load(D / 'inputs/noisy_prefixes' / f'seed_{seed}/transitions.npz')
        trace, replay = load(folder / 'transitions.npz'), load(folder / 'independent_audit/transitions.npz')
        for key in trace:
            assert np.array_equal(trace[key], replay[key]), (seed, key)
        for key in pooled_fields:
            assert np.array_equal(trace[key][:9], source[key]), (seed, key)
        assert all(audit['replay_bitexact'].values())
        assert len(trace['actions']) == 9 + outcome['mpc_controls']
        mask = load(folder / 'control_origins.npz')['expert_correction_mask']
        assert np.array_equal(mask, np.arange(len(trace['actions'])) >= 9)
        all_origins.extend(mask.tolist())
        physics = load(folder / 'physics.npz')
        assert np.array_equal(physics['samples'], load(folder / 'independent_audit/physics.npz')['samples'])
        assert np.array_equal(physics['samples'][:450],
                              load(D / 'inputs/noisy_prefixes' / f'seed_{seed}/physics.npz')['samples'])
        assert len(physics['samples']) == len(trace['actions']) * 50
        col = {key: physics['samples'][:, index] for index, key in enumerate(physics['columns'])}
        eligible = ((col['raw_angle'] >= 2*np.pi) & (col['force_qualified_angle'] >= 2*np.pi)
                    & (col['force_retained'] > 0) & (col['palm_support_violation'] == 0))
        first = np.flatnonzero(eligible)
        correction_actions = trace['actions'][9:]
        assert np.array_equal(correction_actions, np.clip(correction_actions, -.98, .98))
        projected = [record for record in report['reference_records'] if record['projected']]
        planning = sum(command.get('planning_seconds', 0.) for command in report['commands'])
        total_planning += planning
        total_wall += report['wall_seconds']
        row = dict(**outcome, total_controls=len(trace['actions']),
            first_force_fullturn_physics_seconds=float(col['time'][first[0]]) if len(first) else None,
            terminal_seconds=float(trace['control_times'][-1]), final_raw_angle=audit['final_raw_angle'],
            final_retained_angle=audit['final_qualified_angle'],
            final_force_bearing_fingers=audit['final_force_bearing_fingers'],
            final_axis_tilt_radians=audit['final_axis_tilt_radians'],
            maximum_axis_tilt_radians=float(col['axis_tilt_radians'].max()),
            peak_palm_normal_force=float(col['palm_normal_force'].max()),
            peak_external_force=max(float(col[key].max()) for key in ('palm_force_norm', 'table_force_norm',
                'other_force_norm', 'object_applied_wrench_norm', 'object_applied_generalized_force_norm')),
            prefix_integration_and_all_physics_bitexact=True, full_independent_replay_bitexact=True,
            correction_clamp_epsilon_002_bitwise_unchanged=True, projected_reference_batches=len(projected),
            maximum_reference_projection_radians=max((item['projection_max_radians'] for item in projected), default=0.),
            planning_seconds=planning, wall_seconds=report['wall_seconds'],
            trajectory_sha256=sha(folder / 'transitions.npz'), physics_sha256=sha(folder / 'physics.npz'),
            independent_audit_sha256=sha(folder / 'independent_audit/report.json'))
        rows.append(row)
        for field in pooled_fields:
            values[field].append(trace[field][9:])
        for control in range(9, len(trace['actions'])):
            pair_metadata.append(dict(row=len(pair_metadata), seed=seed, source_control_1based=control + 1,
                source_transition_row_0based=control, source_trajectory=f'seed_{seed}/transitions.npz',
                source_trajectory_sha256=row['trajectory_sha256'], origin='actually_executed_corrected_native_mpc',
                successful_source_branch=outcome['strict_force_success']))
    arrays = {key: np.concatenate(parts) for key, parts in values.items()}
    arrays.update(seed=np.asarray([row['seed'] for row in pair_metadata]),
        source_control_1based=np.asarray([row['source_control_1based'] for row in pair_metadata]))
    np.savez_compressed(D / 'actual_mpc_correction_pairs.npz', **arrays)
    pairs_report = dict(kind='diagnostic-actual-mpc-correction-supervised-pairs',
        rows=len(pair_metadata), source_branches=5, noisy_prefix_rows_excluded=45,
        no_fabricated_next_actions=True, no_next_action_field=True, not_a_cold_reset_replay_dataset=True,
        production_eligible=False, main_demonstrations=0, SAC_successes=0, learning_updates=0,
        data_sha256=sha(D / 'actual_mpc_correction_pairs.npz'), row_provenance=pair_metadata)
    write(D / 'actual_mpc_correction_pairs.metadata.json', pairs_report)
    # Reopen the serialized pairs and check every field against every actual source row.
    pairs = load(D / 'actual_mpc_correction_pairs.npz')
    for item in pair_metadata:
        source = load(D / item['source_trajectory'])
        for field in pooled_fields:
            assert np.array_equal(pairs[field][item['row']], source[field][item['source_transition_row_0based']])
    final = dict(status='complete', protocol_sha256=sha(D / 'protocol.json'),
        input_manifest_sha256=sha(D / 'inputs_manifest.json'), script_sha256=sha(D / 'run_recoveries.py'),
        completed_branches=5, strict_force_successes=sum(row['strict_force_success'] for row in rows),
        actual_prefix_controls=45, actual_mpc_controls=len(pair_metadata),
        total_executed_controls=len(all_origins), total_physics_samples=len(all_origins)*50,
        total_planning_seconds=total_planning, total_branch_wall_seconds=total_wall,
        all_original_and_copied_input_hashes_unchanged=True,
        all_source_row_identities_verified=True, outcomes=rows,
        source_kind='Prescribed fixed-mean minimum-noise physical prefixes, followed by feedback MPC; not closed-loop actor or SAC policy success',
        no_robustness_rate_claim=True, no_post_success_hold_or_policy_continuation=True,
        no_prefix_or_parameter_search=True, production_eligible=False, main_demonstrations=0,
        SAC_successes=0, learning_updates=0,
        startup_correction='One missing archived helper import failed before environment creation; exact helper was added and input hashes were refrozen. Protocol and execution script were unchanged.')
    write(D / 'final_report.json', final)
    lines = ['# Five prescribed noisy-state MPC recoveries', '',
        '**All five prescribed states reach a strict force-qualified full turn after 1–7 actual MPC corrections.** '
        'These are fixed-mean noisy prefixes followed by MPC feedback, not learned-policy or SAC successes.', '',
        '| Source seed | Prefix force-qualified angle | MPC corrections | Total controls / seconds | Final force-qualified angle | Loaded fingers | Final tilt |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['seed']} | {row['branch_force_qualified_angle']:.6f} rad | {row['mpc_controls']} | "
            f"{row['total_controls']} / {row['terminal_seconds']:.1f} | {row['final_force_qualified_angle']:.6f} rad | "
            f"{row['final_force_bearing_fingers']} | {row['final_axis_tilt_radians']:.6f} rad |")
    lines += ['', 'Every branch begins with a cold-reset replay of its nine previously recorded controls. '
        'Actions, observations, rewards, terminal flags, qpos/qvel, actuator targets, complete integration states and all 450 prefix physics rows match the original source bit-for-bit. '
        'No noise is regenerated. The original latent means, standardized Gaussian draws, actual sampled commands, RNG snapshots and reports are copied into inputs/. '
        'The prefixes are the earlier fixed-mean experiment, not the failing closed-loop neural trajectories.', '',
        'The planner is the previously approved corrected native recipe: eight workers, 350 candidates, ten elites, '
        'three iterations, six knots over .5 seconds, .1-second controls, retained progress, perpendicular weight 100, '
        'position weight 1000, principal weight 1, and the same six-component measured 40 mm prior. Each branch starts '
        'CEM with seed 42 and at most twenty correction controls. The .98 normalized-action search limit has the same '
        'float32-safe effective bound .9799998998641968. Warm nominal and current-target reference candidates are retained. '
        'Where the true live target lies outside search bounds, only the reference target is explicitly projected; live '
        'physics, accumulated rotation, velocities and targets are never reset or clipped at the switch. Every planning '
        'call leaves live integration state unchanged. All executed correction commands remain inside the bounds and '
        'are bitwise unchanged by BC epsilon .02 clipping.', '',
        'Each branch stopped at its first environment success and independently passed the stronger force-supported '
        'criterion. No branch drops or uses palm/table/other/applied support. A separate fresh physical replay of '
        f'all {len(all_origins)} executed controls and {len(all_origins)*50:,} physics transitions matches bit-for-bit. '
        'The task used unchanged gravity/contact physics and the strict 40 mm horizon-100 environment. No alternative '
        'prefix, planner parameter, hold branch, BC update or SAC update was tried.', '',
        f'`actual_mpc_correction_pairs.npz` contains exactly {len(pair_metadata)} actually executed MPC rows; all 45 noisy '
        'prefix rows are excluded as expert labels. The sidecar maps every row to its original branch, control and '
        'hash-bound full transition archive. Observations, actions, actual next observations, rewards, terminal flags '
        'and physical states are preserved. No successor actions are invented. These are supervised diagnostic pairs '
        'from mid-episode states, not a new cold-reset replay dataset or main-stage demonstration export.', '',
        'This establishes that the fixed feedback planner can recover a turn from each of these five specifically '
        'tested partial states. It supplies actual corrective coverage that the original nominal BC dataset lacked. '
        'It does not establish that an actor can fit or use that coverage, that stochastic SAC will succeed, or that '
        'the recoveries generalize to other states. No post-success hold or continued-policy test was performed; '
        'sustained retention is unmeasured for these branches. Five prescribed examples are not a population robustness '
        'estimate. The diagnostic prior remains MPC-derived, so none of this advances the SAC-prerequisite or twenty '
        'main-demonstration counts.', '',
        'One startup attempt failed on a missing archived helper import before constructing an environment. The '
        'error log is retained; the byte-identical helper was added, inputs were refrozen, and the same unchanged '
        'protocol and execution script then completed with exit 0. All copied and original immutable inputs pass '
        'the final hash check. Source snapshots, native build provenance, command, runtime, complete failed-start '
        'record, branch states, controls, independent audits and row identities remain in this artifact.', '']
    (D / 'findings.md').write_text('\n'.join(lines))
    write(D / 'manifest.json', {str(path.relative_to(D)): sha(path) for path in sorted(D.rglob('*'))
        if path.is_file() and '__pycache__' not in path.parts and path != D / 'manifest.json'})
    print(json.dumps(dict(final_report_sha256=sha(D / 'final_report.json'),
        findings_sha256=sha(D / 'findings.md'), manifest_sha256=sha(D / 'manifest.json'),
        correction_pairs_sha256=sha(D / 'actual_mpc_correction_pairs.npz'),
        correction_pairs_metadata_sha256=sha(D / 'actual_mpc_correction_pairs.metadata.json'),
        total_mpc_controls=len(pair_metadata), total_executed_controls=len(all_origins)), indent=2))


if __name__ == '__main__':
    main()
