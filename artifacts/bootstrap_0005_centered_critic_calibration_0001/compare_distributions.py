from pathlib import Path
import numpy as np,json,shutil
base=Path('/home/autel001/code/mpc_dex/artifacts');old=base/'bootstrap_0005_frozen_critic_calibration_0001';out=base/'bootstrap_0005_centered_critic_calibration_0001'
z=np.load(out/'reconstructed_preupdate_controls.npz');start=np.r_[True,np.diff(z['episode_id'])!=0]
def stats(a):
 a=np.asarray(a,dtype=np.float64);return dict(n=len(a),mean=float(a.mean()),variance=float(a.var()),std=float(a.std()),quantiles=dict(zip(('0','10','25','50','75','90','100'),np.quantile(a,[0,.1,.25,.5,.75,.9,1]).tolist())))
held=json.loads((out/'assessment_after.json').read_text())['heldout_episodes']
result=dict(status='complete',variance_definition='Population variance of recorded finite samples; not uncertainty of a true policy-value estimate',training_initial_unique_observations=len(np.unique(z['obs'][start],axis=0)),
 targets=dict(training_all_256=stats(z['empirical_soft_q_targets']),training_episode_initial_78=stats(z['empirical_soft_q_targets'][start]),heldout_episode_initial_5=stats([x['initial_soft_return'] for x in held])),predictions=[])
for artifact,directory in [('uncentered',old),('centered',out)]:
 cap=json.loads((directory/'capacity_analysis.json').read_text())
 for phase,assess_name in [('initial','assessment_before.json'),('after_500','assessment_after.json')]:
  records=next(x['critics'] for x in cap['checkpoints'] if x['label']==phase);a=json.loads((directory/assess_name).read_text())
  for i,r in enumerate(records):
   result['predictions'].append(dict(artifact=artifact,phase=phase,critic=i+1,training_episode_initial_78=stats(r['training_initial_state_q']),heldout_episode_initial_5=stats([e['initial_q'][i] for e in a['heldout_episodes']]),training_episode_initial_rmse=r['training_initial_state_rmse'],training_all_bias=r['training_bias'],training_all_rmse=r['training_rmse']))
(out/'return_distribution_comparison.json').write_text(json.dumps(result,indent=2)+'\n');shutil.copyfile(__file__,out/'compare_distributions.py')
print(json.dumps({'targets':result['targets'],'fitted_predictions':[r for r in result['predictions'] if r['phase']=='after_500']},indent=2))
