"""Verify the completed strict branch, then execute its authorized success hold.

The original script completed and saved all physical controls before its
postprocessing requested the optional teacher 'targets' field. This script
preserves that failure and the single strict execution; it does not rerun it.
"""
from dataclasses import replace
import json
from pathlib import Path
import sys

D=Path(__file__).resolve().parent;RECOVERY=D.parent;ROOT=RECOVERY.parents[1]
sys.path.insert(0,str(RECOVERY));sys.path.insert(0,str(RECOVERY/'source'))
import numpy as np
from mpc_dex.env import EnvConfig,AllegroRotationEnv
from mpc_dex.contracts import environment_contract,sha256
from scripts.audit_policy import write_json
from scripts.audit_recorded_controls import RecordedPolicy
from audit_helpers import run_audit


def load(path):
    with np.load(path,allow_pickle=False) as archive:return {k:archive[k].copy() for k in archive.files}


def main():
    sources=dict(teacher=ROOT/'artifacts/mpc_matched40_bound098_0001/transitions.npz',
        teacher_physics=ROOT/'artifacts/mpc_matched40_bound098_0001/physics.npz',
        corrections=RECOVERY/'transitions.npz',config=RECOVERY/'config.json',
        auditor=RECOVERY/'source/scripts/audit_policy.py',environment=RECOVERY/'source/mpc_dex/env.py',
        audit_helper=RECOVERY/'audit_helpers.py',recovery_manifest=RECOVERY/'manifest.json')
    inputs={name:dict(path=str(path),sha256=sha256(path)) for name,path in sources.items()}
    teacher=load(D/'source_teacher_transitions.npz');recovery=load(D/'source_recovery_transitions.npz')
    assert sha256(D/'source_teacher_transitions.npz')==inputs['teacher']['sha256']
    assert sha256(D/'source_recovery_transitions.npz')==inputs['corrections']['sha256']
    assert sha256(D/'config.json')==inputs['config']['sha256']
    ec=EnvConfig(**json.loads((D/'config.json').read_text())['environment'])
    actions=np.vstack((teacher['actions'][:6],recovery['actions'][6:9]))
    executed=load(D/'strict_audit/transitions.npz')
    assert len(executed['actions'])==len(actions)==9
    assert np.array_equal(executed['actions'],actions) and abs(actions).max()<=.98
    keys=['obs','actions','rewards','next_obs','qpos','qvel']
    for key in keys:assert np.array_equal(executed[key][:6],teacher[key][:6]),key
    assert not executed['terminated'][:6].any() and not executed['truncated'][:6].any()
    original_physics=load(sources['teacher_physics']);physics=load(D/'strict_audit/physics.npz')
    assert np.array_equal(original_physics['columns'],physics['columns'])
    assert np.array_equal(original_physics['samples'][:300],physics['samples'][:300])
    env=AllegroRotationEnv(ec)
    try:assert np.array_equal(executed['targets'],env.to_joint_targets(actions))
    finally:env.close()
    result=json.loads((D/'strict_audit/report.json').read_text())
    assert result['finger_supported_rotation_success']
    report=dict(purpose='One compatibility branch: original bounded teacher controls 1-6 then actual saved MPC recovery corrections 7-9',
        production_eligible=False,policy_only_success=False,training_updates=0,planning_calls=0,
        reset_or_clipping_at_switch=False,teacher_prefix_transitions_bitexact=True,
        teacher_prefix_verified_fields=keys,teacher_prefix_physics_bitexact=True,verified_prefix_physics_rows=300,
        executed_targets_match_actual_action_mapping=True,teacher_continuation_rows_used=False,
        strict_branch_execution_count=1,initial_postprocessing_error='Original teacher archive contains obs/actions/rewards/next_obs/qpos/qvel only; available fields verified exactly, executed targets verified directly from actions, and actual terminal flags preserved.',
        inputs=inputs,source_script_sha256=sha256(D/'compatibility.py'),finalize_script_sha256=sha256(__file__),
        environment_contract=environment_contract(ec),requested_controls=9,executed_controls=9,
        used_correction_controls=3,action_abs_max=float(abs(actions).max()),strict_audit=result)
    write_json(D/'report.json',report)
    held=np.vstack((actions,np.repeat(actions[-1:],20,axis=0)))
    hold_ec=replace(ec,terminate_on_success=False,horizon_steps=max(ec.horizon_steps,len(held)))
    hold,_,_=run_audit(RecordedPolicy(held),hold_ec,D/'post_success_hold_2s',len(held))
    hold_physics=load(D/'post_success_hold_2s/physics.npz')
    assert np.array_equal(physics['samples'],hold_physics['samples'][:450])
    columns={name:hold_physics['samples'][:,i] for i,name in enumerate(hold_physics['columns'])}
    hold.update(requested_hold_seconds=2.,completed_hold_seconds=max(0.,hold['simulated_seconds']-.9),
        completed_two_seconds_without_failure=bool(hold['steps']==29 and not hold['independent_drop'] and not hold['independent_palm_support_violation']),
        minimum_hold_loaded_fingers=int(columns['force_bearing_fingers'][450:].min()),
        maximum_hold_axis_tilt_radians=float(columns['axis_tilt_radians'][450:].max()),
        force_retained_every_hold_physics_step=bool(columns['force_retained'][450:].all()))
    write_json(D/'post_success_hold_2s/report.json',hold);report['post_success_hold_2s']=hold
    write_json(D/'report.json',report)
    assert all(sha256(sources[name])==record['sha256'] for name,record in inputs.items())
    write_json(D/'manifest.json',{str(p.relative_to(D)):sha256(p) for p in sorted(D.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p.name not in ('manifest.json','finalize.log')})
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
