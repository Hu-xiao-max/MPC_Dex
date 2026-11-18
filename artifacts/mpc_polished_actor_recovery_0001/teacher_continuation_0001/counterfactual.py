"""One actual actor prefix followed verbatim by the original bounded teacher."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import sys

D=Path(__file__).resolve().parent
RECOVERY=D.parent
ROOT=RECOVERY.parents[1]
sys.path.insert(0,str(RECOVERY));sys.path.insert(0,str(RECOVERY/'source'))
import numpy as np
from mpc_dex.env import EnvConfig
from mpc_dex.contracts import environment_contract,sha256
from scripts.audit_policy import write_json
from scripts.audit_recorded_controls import RecordedPolicy
from audit_helpers import run_audit


def main():
    teacher_source=ROOT/'artifacts/mpc_matched40_bound098_0001/transitions.npz'
    sources=dict(actor=RECOVERY/'actor_transitions.npz',teacher=teacher_source,
                 config=RECOVERY/'config.json',auditor=RECOVERY/'source/scripts/audit_policy.py',
                 environment=RECOVERY/'source/mpc_dex/env.py',audit_helper=RECOVERY/'audit_helpers.py',
                 recovery_manifest=RECOVERY/'manifest.json')
    inputs={name:dict(path=str(path),sha256=sha256(path)) for name,path in sources.items()}
    shutil.copyfile(sources['actor'],D/'source_actor_transitions.npz')
    shutil.copyfile(teacher_source,D/'source_teacher_transitions.npz')
    shutil.copyfile(sources['config'],D/'config.json')
    ec=EnvConfig(**json.loads((D/'config.json').read_text())['environment'])
    assert ec.terminate_on_success and ec.object_diameter==.04
    with np.load(D/'source_actor_transitions.npz',allow_pickle=False) as a:
        actor={k:a[k].copy() for k in a.files}
    with np.load(D/'source_teacher_transitions.npz',allow_pickle=False) as a:
        teacher={k:a[k].copy() for k in a.files}
    assert len(teacher['actions'])==60 and actor['actions'].dtype==teacher['actions'].dtype==np.float32
    assert abs(teacher['actions']).max()<=.98
    actions=np.vstack((actor['actions'][:6],teacher['actions'][6:]))
    result,executed,_=run_audit(RecordedPolicy(actions),ec,D/'strict_audit',len(actions))
    with np.load(D/'strict_audit/transitions.npz',allow_pickle=False) as a:
        for key in ('obs','actions','rewards','next_obs','qpos','qvel','targets','terminated','truncated'):
            assert np.array_equal(a[key][:6],actor[key][:6]),key
        assert np.array_equal(a['actions'],actions[:len(executed)])
    with np.load(D/'strict_audit/physics.npz',allow_pickle=False) as a,np.load(RECOVERY/'actor_physics.npz',allow_pickle=False) as b:
        assert np.array_equal(a['samples'][:300],b['samples'][:300])
    report=dict(purpose='One counterfactual: six actual polished-actor controls followed by original bounded teacher rows 7-60 verbatim until strict terminal or budget',
        production_eligible=False,policy_only_success=False,training_updates=0,planning_calls=0,
        reset_or_clipping_at_switch=False,actor_prefix_bitexact=True,inputs=inputs,
        source_script_sha256=sha256(__file__),environment_contract=environment_contract(ec),
        requested_controls=len(actions),executed_controls=len(executed),
        used_teacher_controls=max(0,len(executed)-6),unused_teacher_tail=len(actions)-len(executed),
        strict_audit=result)
    write_json(D/'report.json',report)
    print(json.dumps(report,indent=2),flush=True)
    if result['finger_supported_rotation_success']:
        held=np.vstack((executed,np.repeat(executed[-1:],20,axis=0)))
        hold_ec=replace(ec,terminate_on_success=False,horizon_steps=max(ec.horizon_steps,len(held)))
        hold,_,_=run_audit(RecordedPolicy(held),hold_ec,D/'post_success_hold_2s',len(held))
        with np.load(D/'strict_audit/physics.npz',allow_pickle=False) as a,np.load(D/'post_success_hold_2s/physics.npz',allow_pickle=False) as b:
            assert np.array_equal(a['samples'],b['samples'][:len(a['samples'])])
        hold.update(requested_hold_seconds=2.,completed_hold_seconds=max(0.,hold['simulated_seconds']-.1*len(executed)),
            completed_two_seconds_without_failure=bool(hold['steps']==len(held) and not hold['independent_drop'] and not hold['independent_palm_support_violation']))
        write_json(D/'post_success_hold_2s/report.json',hold);report['post_success_hold_2s']=hold
        write_json(D/'report.json',report)
        print(json.dumps({'post_success_hold_2s':hold},indent=2),flush=True)
    assert all(sha256(sources[name])==record['sha256'] for name,record in inputs.items())
    write_json(D/'manifest.json',{str(p.relative_to(D)):sha256(p) for p in sorted(D.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p.name not in ('manifest.json','counterfactual.log')})


if __name__=='__main__':main()
