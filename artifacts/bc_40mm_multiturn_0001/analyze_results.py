"""Analyze fixed checkpoint fits and executed error growth; no new updates."""
from pathlib import Path
import json,sys
D=Path(__file__).resolve().parent
sys.path.insert(0,str(D/'source'))
import numpy as np
import torch
from mpc_dex.learning import SACAgent
from mpc_dex.contracts import sha256
from scripts.audit_policy import write_json

torch.set_num_threads(1);torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32=False
report=json.loads((D/'report.json').read_text())
with np.load(D/'teacher_transitions.npz') as a:
    obs,actions,qpos,qvel=[a[k].copy() for k in ['obs','actions','qpos','qvel']]
analysis=dict(purpose='Fixed-checkpoint fit and actual rollout error analysis; zero training/physics updates',
    unique_training_observations=int(len(np.unique(obs,axis=0))),total_observations=len(obs),conditions=[])
assert report['conditions'][0]['initial_actor_sha256']==report['conditions'][1]['initial_actor_sha256']
for condition in report['conditions']:
    folder=D/condition['label'];count=condition['training_rows'];rows=[]
    for result in condition['results']:
        update=result['update'];agent=SACAgent.load(folder/f'checkpoint_{update:05d}.pt','cuda')
        predictions=np.asarray([agent.act(o,deterministic=True) for o in obs])
        errors=predictions-actions
        with np.load(folder/f'teacher_fit_{update:05d}.npz') as f:
            latent=f['predicted_latent_means']-f['latent_targets']
        with np.load(folder/f'audit_{update:05d}'/'transitions.npz') as a:
            actual=[]
            for i in range(min(len(a['actions']),len(actions))):
                actual.append(dict(step=i+1,
                    observation_max_error=float(abs(a['obs'][i]-obs[i]).max()),
                    action_max_error=float(abs(a['actions'][i]-actions[i]).max()),
                    next_qpos_max_error=float(abs(a['qpos'][i]-qpos[i]).max()),
                    next_qvel_max_error=float(abs(a['qvel'][i]-qvel[i]).max())))
        row=dict(update=update,first_action_max_error=float(abs(errors[0]).max()),
            first19_action_mse=float(np.mean(errors[:19]**2)),
            training_action_mse=float(np.mean(errors[:count]**2)),
            training_action_max_error=float(abs(errors[:count]).max()),
            largest_training_action_error_step=int(np.unravel_index(abs(errors[:count]).argmax(),errors[:count].shape)[0]+1),
            largest_training_action_error_joint=int(np.unravel_index(abs(errors[:count]).argmax(),errors[:count].shape)[1]),
            teacher_latent_mse=float(np.mean(latent**2)),teacher_latent_max_error=float(abs(latent).max()),
            training_action_max_error_by_step=abs(errors[:count]).max(1).tolist(),actual_rollout=actual)
        rows.append(row)
    analysis['conditions'].append(dict(label=condition['label'],training_rows=count,evaluations=rows))
write_json(D/'fit_diagnosis.json',analysis)
assert sha256(D/'run.py')==report['run_source_sha256']
assert all(sha256(D/k)==v for k,v in json.loads((D/'inputs_manifest.json').read_text()).items())
report.update(analysis_source_sha256=sha256(__file__),source_and_input_hashes_verified=True,
    aftermath_performed=False,aftermath_reason='No clone passed independently qualified strict full-turn audit')
write_json(D/'report.json',report)
print(json.dumps(analysis,indent=2))
