"""Finalize existing physical results only; do not import or run a simulator."""
from pathlib import Path
import hashlib
import json
import numpy as np

D = Path(__file__).resolve().parent
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
read = lambda name: json.loads((D / name).read_text())
write = lambda name, value: (D / name).write_text(json.dumps(value, indent=2) + '\n')
assert not (D / 'report.json').exists()
inputs = read('inputs_manifest.json')
changed = [name for name, expected in inputs.items() if sha(D / name) != expected]
assert changed == ['run.log'], changed
selection = read('selection.json')
assert selection['boundary'] == 1 + int(np.argmax(selection['boundary_force_angles'])) == 26
with np.load(D / 'strict_audit/transitions.npz') as data, np.load(D / 'source_transitions.npz') as source:
    assert len(data['actions']) == 46
    for key in ('obs', 'actions', 'rewards', 'next_obs', 'terminated', 'truncated', 'qpos', 'qvel', 'targets'):
        np.testing.assert_array_equal(data[key][:26], source[key][:26])
    for key in ('initial_qpos', 'initial_qvel', 'initial_ctrl', 'initial_joint_positions'):
        np.testing.assert_array_equal(data[key], source[key])
    np.testing.assert_array_equal(data['actions'][26:], np.repeat(source['actions'][25:26], 20, axis=0))
    np.testing.assert_array_equal(data['targets'][26:], np.repeat(source['targets'][25:26], 20, axis=0))
with np.load(D / 'strict_audit/physics.npz') as data, np.load(D / 'source_physics.npz') as source:
    np.testing.assert_array_equal(data['samples'][:1300], source['samples'][:1300])
    after = data['samples'][1300:]
    columns = data['columns'].tolist()
    assert len(after) == 1000
    after_metrics = dict(minimum_loaded_fingers=int(after[:,columns.index('force_bearing_fingers')].min()),
        maximum_force_qualified_angle=float(after[:,columns.index('force_qualified_angle')].max()),
        minimum_force_qualified_angle=float(after[:,columns.index('force_qualified_angle')].min()),
        maximum_axis_tilt_radians=float(after[:,columns.index('axis_tilt_radians')].max()))
audit = read('strict_audit/report.json')
assert not audit['finger_supported_rotation_success'] and not audit['independent_drop']
assert audit['no_external_support'] and not audit['independent_palm_support_violation']
report = dict(purpose='One explicitly post-hoc maximum recorded control-boundary hold branch',
    production_eligible=False, policy_only_success=False, training_updates=0, planning_calls=0,
    selection=selection, prefix_bitexact_verified_before_hold=True, prefix_verification=read('prefix_verification.json'),
    strict_audit=audit, independent_success_replay=None, post_success_hold_2s=None,
    conditional_audits_not_run_reason='The single prescribed branch did not achieve strict success.',
    executed_hold_controls=20, held_command_bitexact=True, executed_targets_constant=True,
    after_selected_boundary=after_metrics,
    environment_contract=read('source_report.json')['environment_contract'],
    postprocessing_error='The execution script included changing run.log among immutable inputs. Only run.log changed. This read-only finalizer verifies all source/input hashes and saved transitions without rerunning physics.',
    immutable_inputs_verified={name:digest for name,digest in inputs.items() if name!='run.log'},
    original_execution_input_manifest_sha256=sha(D / 'inputs_manifest.json'),
    strict_transitions_sha256=sha(D / 'strict_audit/transitions.npz'),
    strict_physics_sha256=sha(D / 'strict_audit/physics.npz'),
    finalize_script_sha256=sha(Path(__file__)))
write('report.json', report)
(D / 'README.md').write_text('''# One post-hoc maximum-boundary hold branch

This explicitly post-hoc branch selected **control boundary 26**, the maximum force-qualified angle among the existing 28 recorded boundaries. At 2.6 s it held 6.260594098 rad, with three loaded fingers and tilt 0.192784449 rad. Selection was recorded before execution in selection.json; no other prefix or target was tried.

The cold-reset prefix matched all 26 recorded transitions and 1,300 physics rows bit-for-bit, verified before issuing the first extra hold. The exact float32 command from control 26 was then held for the entire authorized 20-control (2 s) budget, without replanning, blending, clipping, state edits, or target changes.

**No strict full turn occurred.** Final force-qualified rotation was 5.962360321 rad, with two loaded fingers and tilt 0.264846964 rad. The whole-episode peak remained 6.265469558 rad from the original prefix, below 2π. No drop or palm/table/other support occurred. The success-conditioned independent replay and extra 2 s aftermath hold were therefore not run.

Only this one physical branch executed. The final immutable-input assertion initially included the mutable run.log and failed after all physical outputs were saved. finalize.py verified that run.log was the sole changed entry, checked all saved prefix/action/target data, and produced the final report without rerunning physics. Original execution code/log/input hashes are preserved. This is partial diagnostic data, with no policy, SAC, primitive, or production-success claim.
''')
write('manifest.json', {str(path.relative_to(D)):sha(path) for path in sorted(D.rglob('*'))
    if path.is_file() and '__pycache__' not in path.parts and path.name != 'manifest.json'})
print(json.dumps(dict(report_sha256=sha(D/'report.json'),transitions_sha256=report['strict_transitions_sha256'],
                     held=after_metrics),indent=2))
