"""Freeze completed run evidence; no simulator, model inference, or learning."""
from pathlib import Path
import hashlib,json
import numpy as np

D=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads((D/path).read_text())
def write(path,value):(D/path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

execution=read('execution_result.json');state=read('run/run.json')
assert execution['exit_code']==0 and execution['environment_steps']==2000 and not execution['archive_errors']
assert state['status']=='budget_exhausted' and state['online_updates']==1745
assert state['bc_updates']==12000 and state['pretrain_updates']==state['bootstrap_bc_updates']==0
assert state['demonstration_successes']==1 and state['successes']==state['eval_successes']==0
inputs=read('inputs_manifest.json')
assert all(sha(D/path)==digest for path,digest in inputs.items())
rows=[json.loads(line) for line in (D/'run/metrics.jsonl').read_text().splitlines()]
episodes=[r for r in rows if r['event']=='episode']
evaluations=[r for r in rows if r['event']=='evaluation']
checkpoints=[]
for folder in sorted((D/'checkpoints').iterdir()):
    assert folder.is_dir() and not folder.name.endswith('.copying')
    proof=json.loads((folder/'archive_manifest.json').read_text())
    assert all(sha(folder/path)==digest for path,digest in proof['file_hashes'].items())
    checkpoints.append(dict(directory=str(folder.relative_to(D)),environment_steps=proof['environment_steps'],
        online_updates=proof['online_updates'],checkpoint_sha256=proof['checkpoint_sha256'],
        archive_manifest_sha256=sha(folder/'archive_manifest.json'),state_status=proof['state_status']))
assert {p['environment_steps'] for p in checkpoints}=={0,500,1000,1500,2000}
audits={}
for name in ['initial_0001','mid_1000_0001','final_2000_0001']:
    folder=D/'value_calibration'/name
    report=json.loads((folder/'report.json').read_text())
    assert report['status']=='complete' and report['frozen_weights_verified']
    assert len(report['episodes'])==6 and all(r['terminated'] and not r['truncated'] for r in report['episodes'])
    audits[name]=dict(report_path=str(folder/'report.json'),report_sha256=sha(folder/'report.json'),
        checkpoint_sha256=report['checkpoint_sha256'],
        deterministic={k:report['episodes'][0][k] for k in ['finger_supported_rotation_success','environment_failure_reason',
            'steps','final_force_qualified_angle','checkpoint_online_updates','bc_origin_success','sac_updated_policy_success']},
        stochastic_sample_count=report['stochastic_sample_count'],
        stochastic_sample_successes=report['stochastic_sample_successes'],
        mean_initial_action_critic_bias_vs_soft_q=report['mean_matched_action_critic_bias_vs_soft_q'],
        policy_movement=report['policy_movement'],no_time_limit_tails=True)
integrity=dict(archived_inputs_unchanged_at_finalization=True,
    captured_existing_live_files_matched_after_exit=True,
    live_comparison_note='All captured existing source/config/asset file hashes were compared immediately after exit and before root was told deferred source edits could resume; subsequent workspace edits are outside this completed run.',
    initial_actor_source_sha256=read('runtime.json')['immutable_input_actor_sha256'],
    seed_replay_sha256=read('runtime.json')['immutable_seed_sha256'],
    all_archived_checkpoint_files_verified=True,archive_errors=[])
write('terminal_integrity.json',integrity)
summary=dict(status='complete',production_eligible=False,eligible_SAC_primitive=False,
    budget_interactions=2000,actual_online_interactions=state['environment_steps'],actual_SAC_updates=state['online_updates'],
    imported_historical_BC_updates=12000,new_BC_updates=0,critic_pretraining_updates=0,main_stage_updates=0,
    MPC_demonstrations=1,new_SAC_training_successes=0,new_SAC_evaluation_successes=0,
    independent_initial_BC_origin_deterministic_success=audits['initial_0001']['deterministic']['bc_origin_success'],
    BC_origin_primitive_eligible=False,completed_training_episodes=len(episodes),
    episode_failure_reasons={label:sum(r['failure_reason']==label for r in episodes) for label in ['drop','palm_support']},
    physical_drop_count=sum(r['drop'] for r in episodes),
    simultaneous_drop_and_palm_count=sum(r['drop'] and r['palm_support_violation'] for r in episodes),
    mean_completed_episode_return=float(np.mean([r['return_value'] for r in episodes])),
    longest_completed_episode_controls=max(r['steps'] for r in episodes),
    max_episode_end_retained_angle=max(r['retained_angle'] for r in episodes),
    angle_limit_note='Episode-end statistic; runner does not record a full per-physics training trajectory.',
    replay_transitions=state['replay_transitions'],successful_trajectory_exports=state['successful_trajectory_episodes'],
    scheduled_evaluations=evaluations,evaluation_caveat='Each set repeats the same physical cold reset5times; not an independent robustness-rate estimate.',
    preserved_checkpoints=checkpoints,final_generation='checkpoint_00000006',
    actual_final_learner_config=state['learner_config'],runtime=read('runtime.json')['inference_runtime'],
    final_training_metrics=state['latest_metrics'],independent_value_audits=audits,
    independent_value_archive_manifest_sha256=sha(D/'value_calibration/manifest.json'),
    integrity=integrity,automatic_extension=False,exit_code=execution['exit_code'],
    interpretation='Fresh bounded SAC did not retain the BC deterministic turn or produce a new qualified turn; cold-reset critic optimism persists on actual terminal stochastic rollouts. No main-stage or primitive promotion.')
write('summary.json',summary)
manifest={str(p.relative_to(D)):sha(p) for p in sorted(D.rglob('*'))
    if p.is_file() and '__pycache__' not in p.parts and p!=D/'manifest.json'
    and 'value_calibration' not in p.relative_to(D).parts}
write('manifest.json',dict(scope='Managed run/config/source/checkpoint files; independent value_calibration subtree has its own manifests and report hashes above.',files=manifest))
print(json.dumps(dict(status='complete',files=len(manifest),summary_sha256=sha(D/'summary.json'),
                     manifest_sha256=sha(D/'manifest.json'))))
