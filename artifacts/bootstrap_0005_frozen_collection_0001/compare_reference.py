"""Compare completed collection against an existing audit; no physics or learning."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np

D = Path(__file__).resolve().parent
REFERENCE = D.parent/'bootstrap_0005/value_calibration/final_2000_0001/deterministic_strict'
sha = lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
report=json.loads((D/'report.json').read_text())
assert report['status']=='complete' and report['completed_episodes']==5
assert report['successful_episodes']==0 and report['eligible_for_primitive'] is False
assert not (D/'successful_joint_trajectories.npz').exists()
ref=D/'reference_audit';ref.mkdir(exist_ok=False)
for name in ('transitions.npz','physics.npz','report.json','commands.json'):
    shutil.copyfile(REFERENCE/name,ref/name)
shutil.copyfile(REFERENCE.parent/'runtime.json',ref/'runtime.json')
baseline=np.load(ref/'transitions.npz');physics=np.load(ref/'physics.npz')
runtime=json.loads((ref/'runtime.json').read_text())
assert runtime['checkpoint_sha256']==report['checkpoint']['sha256']
comparisons=[]
fields=['obs','actions','rewards','next_obs','terminated','truncated','qpos','qvel','targets','integration_state']
for record in report['episodes']:
    data=np.load(D/record['evidence']['transitions']['path'])
    observed=np.load(D/record['evidence']['physics']['path'])
    errors={key:float(np.max(abs(data[key].astype(float)-baseline[key].astype(float)))) for key in fields}
    assert all(value==0 for value in errors.values())
    np.testing.assert_array_equal(observed['samples'],physics['samples'])
    np.testing.assert_array_equal(observed['columns'],physics['columns'])
    np.testing.assert_array_equal(data['joint_positions'][0],data['initial_qpos'][:16])
    np.testing.assert_array_equal(data['joint_positions'][1:],data['qpos'][:,:16])
    comparisons.append(dict(episode=record['episode'],seed=record['seed'],controls=record['steps'],
        transition_max_errors=errors,all_350_physics_samples_bitexact=True,initial_terminal_joint_frames_verified=True,
        force_qualified_angle=record['force_qualified_angle'],failure_reason=record['failure_reason'],
        peak_palm_support_force=record['peak_palm_support_force']))
result=dict(purpose='Actual collector integration on one immutable SAC-updated checkpoint',
    collector_exit_code=2,eligible_for_primitive=False,no_eligible_joint_archive=True,new_learning_updates=0,
    checkpoint=report['checkpoint'],learner_counters=report['learner_counters'],comparisons=comparisons,
    reference_path=str(REFERENCE),reference_sha256={name:sha(ref/name) for name in ('transitions.npz','physics.npz','report.json','runtime.json')},
    collector_report_sha256=sha(D/'report.json'),
    unique_initial_physics_states=report['unique_initial_physics_states'],
    unique_measured_joint_trajectories=report['unique_measured_joint_trajectories'],
    runtime_note='Collector disables CuDNN TF32 while reference permits it; this actor uses no CuDNN convolution. Actual commands, full integration states and all physics compare bit-exact.',
    robustness_or_success_claim=False)
(D/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
(D/'findings.md').write_text('''# Real frozen-SAC collector integration

The new collector executed exactly five deterministic attempts from archived bootstrap0005 checkpoint6, with BC12000, critic-pretraining0 and actual online SAC1745 updates. It used the unchanged40mm horizon100 contract, CUDA singleton inference, one intra/inter-op thread and TF32 disabled. The checkpoint and numerical sources were copied before execution.

All five attempts fail at control7 due to force-bearing palm support. Each ends at force-qualified rotation1.0032547032686967rad and peak palm normal force1.7862867488820415N. The collector returns2, preserves all five failed episodes, sets eligible_for_primitive=false, and emits no successful_joint_trajectories.npz.

Each attempt matches the previously independent final-checkpoint deterministic audit exactly: observations, actions, rewards, next observations, terminal flags, qpos, qvel, targets, full MuJoCo integration state, and all350physics samples have zero difference. Initial and terminal measured joint frames are retained. Seeds42–46 produce one unique initial state and one unique measured trajectory; these are five repeated executions, not reset-diversity or robustness evidence.

This verifies the real collector and its refusal to export a failed SAC policy. It does not establish a successful SAC primitive or run any new learning. compare_reference.py only reads saved arrays. comparison.json records source hashes and exact comparisons.
''')
if (D/'manifest.json').exists():shutil.copyfile(D/'manifest.json',D/'collection_manifest.json')
(D/'manifest.json').write_text(json.dumps({str(path.relative_to(D)):sha(path) for path in sorted(D.rglob('*'))
    if path.is_file() and '__pycache__' not in path.parts and path.name!='manifest.json'},indent=2)+'\n')
print(json.dumps(dict(comparison_sha256=sha(D/'comparison.json'),report_sha256=sha(D/'report.json')),indent=2))
