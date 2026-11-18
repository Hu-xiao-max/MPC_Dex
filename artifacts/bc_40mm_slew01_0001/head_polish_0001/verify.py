"""Verify exact checkpoint changes and actual Torch actions; no new physics."""
from pathlib import Path
import json,sys
D=Path(__file__).resolve().parent
sys.path.insert(0,str(D.parent/'source'))
import numpy as np
import torch
from mpc_dex.learning import SACAgent
from mpc_dex.contracts import sha256
from scripts.audit_policy import write_json

torch.set_num_threads(1);torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32=False
baseline=torch.load(D/'baseline.pt',map_location='cpu',weights_only=True)
polished=torch.load(D/'polished.pt',map_location='cpu',weights_only=True)
changed=[]


def compare(a,b,path):
    if type(a)!=type(b):changed.append(path+'[type]');return
    if isinstance(a,torch.Tensor):
        if a.dtype!=b.dtype or a.shape!=b.shape or a.numpy().tobytes()!=b.numpy().tobytes():changed.append(path)
    elif isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:compare(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i))
    elif a!=b:changed.append(path)


compare(baseline,polished,'checkpoint')
assert set(changed)=={'checkpoint/actor/mean_head.weight','checkpoint/actor/mean_head.bias'},changed
with np.load(D/'solve_inputs_and_result.npz') as a:obs,actions=a['observations'].copy(),a['actions'].copy()
predictions={};fit={}
for name in ['baseline','polished']:
    agent=SACAgent.load(D/(name+'.pt'),'cuda')
    predictions[name]=np.asarray([agent.act(o,deterministic=True) for o in obs])
    difference=predictions[name].astype(np.float64)-actions.astype(np.float64)
    fit[name]=dict(action_mse=float(np.mean(difference**2)),action_max_abs_error=float(abs(difference).max()),
                   first_action_max_abs_error=float(abs(difference[0]).max()),
                   action_max_abs_error_by_step=abs(difference).max(1).tolist())
np.savez_compressed(D/'actual_torch_teacher_predictions.npz',observations=obs,teacher_actions=actions,**predictions)
with np.load(D/'independent_audit/transitions.npz') as a,np.load(D.parent/'strict_first_success/transitions.npz') as t:
    growth=[]
    for i in range(min(len(a['actions']),len(t['actions']))):
        growth.append(dict(step=i+1,observation_max_error=float(abs(a['obs'][i]-t['obs'][i]).max()),
            action_max_error=float(abs(a['actions'][i]-t['actions'][i]).max()),
            next_qpos_max_error=float(abs(a['qpos'][i]-t['qpos'][i]).max()),
            next_qvel_max_error=float(abs(a['qvel'][i]-t['qvel'][i]).max())))
proof=dict(changed_checkpoint_paths=changed,all_other_payload_values_bitwise_unchanged=True,
           actual_torch_action_fit=fit,actual_error_growth=growth,source_script_sha256=sha256(__file__))
write_json(D/'proof.json',proof)
report=json.loads((D/'report.json').read_text())
assert sha256(D/'polish.py')==report['source_script_sha256']
report.update(actual_torch_action_fit=fit,
    fit_metric_note='baseline_fit/polished_fit reconstruct tanh using NumPy; actual_torch_action_fit uses the exact singleton SACAgent.act Torch route and is authoritative at float32 precision.',
    changed_checkpoint_paths=changed,all_other_payload_values_bitwise_unchanged=True,
    proof_source_sha256=sha256(__file__))
write_json(D/'report.json',report)
print(json.dumps(proof,indent=2))
