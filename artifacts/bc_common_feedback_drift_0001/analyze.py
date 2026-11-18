"""Read-only attribution of executed noise versus conditional-mean drift."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np

D=Path(__file__).resolve().parent
F=D.parent/'bc_common_continuation_0001'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files={'nominal':F/'head_polish/independent_audit/transitions.npz'}
files.update({str(seed):F/f'bootstrap_exploration_minstd_0001/stochastic_seed_{seed}/transitions.npz' for seed in range(17021,17026)})
inputs={}
for key,path in files.items():
    shutil.copyfile(path,D/f'{key}.npz');inputs[str(path)]=sha(path)
with np.load(files['nominal'],allow_pickle=False) as a:
    nominal={k:a[k].copy() for k in a.files}
results=[]
for seed in range(17021,17026):
    with np.load(files[str(seed)],allow_pickle=False) as a:trial={k:a[k].copy() for k in a.files}
    rows=[]
    for i in range(min(9,len(trial['actions']))):
        noise=trial['actions'][i].astype(float)-trial['conditional_mean_actions'][i]
        drift=trial['conditional_mean_actions'][i].astype(float)-nominal['actions'][i]
        obs=trial['obs'][i].astype(float)-nominal['obs'][i]
        rows.append(dict(step=i+1,action_noise_max=float(abs(noise).max()),
            action_noise_l2=float(np.linalg.norm(noise)),conditional_mean_drift_max=float(abs(drift).max()),
            conditional_mean_drift_l2=float(np.linalg.norm(drift)),
            drift_to_direct_noise_l2_ratio=float(np.linalg.norm(drift)/np.linalg.norm(noise)) if np.linalg.norm(noise)>0 else None,
            observation_max=float(abs(obs).max()),
            joint_velocity_observation_delta_max_rad_per_s=float(abs(obs[25:41]).max()),
            previous_action_delta_max=float(abs(obs[44:60]).max())))
    assert rows[0]['conditional_mean_drift_max']==0 and rows[0]['observation_max']==0
    results.append(dict(seed=seed,rows=rows))
assert all(sha(Path(p))==h for p,h in inputs.items())
report=dict(purpose='Read-only comparison of direct sampled action noise and conditional neural mean drift against the successful deterministic trajectory',
    inference='The causal contribution of feedback requires the separate fixed-mean physical comparison; these measured drift ratios alone are not a causal proof.',
    action_units='normalized joint-target actions for both numerator and denominator',
    observation_slices='60-entry rotation contract; velocity[25:41],previous_action[44:60]',
    no_new_training_or_physics=True,production_eligible=False,inputs=inputs,results=results)
(D/'report.json').write_text(json.dumps(report,indent=2)+'\n')
(D/'manifest.json').write_text(json.dumps({p.name:sha(p) for p in D.iterdir() if p.is_file() and p.name not in ('manifest.json','run.log')},indent=2)+'\n')
for result in results:
    print(json.dumps(dict(seed=result['seed'],step2=result['rows'][1]),indent=2))
