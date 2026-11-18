from pathlib import Path
import sys,json,hashlib,shutil
out=Path('/home/autel001/code/mpc_dex/artifacts/bootstrap_0005_frozen_critic_calibration_0001');sys.path.insert(0,str(out/'source'))
import numpy as np,torch
from mpc_dex.learning import SACAgent
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
with np.load(out/'reconstructed_preupdate_controls.npz') as z:
 mask=z['eligible_complete_terminal'];obs=z['obs'][mask];act=z['actions'][mask];targets=z['empirical_soft_q_targets'][mask];episode=z['episode_id'][mask]
heldout=[]
for seed in range(17041,17046):
 with np.load(out/f'inputs/heldout_{seed}/transitions.npz') as z:heldout.append((z['obs'][0].copy(),z['actions'][0].copy()))
x=torch.as_tensor(np.concatenate((obs,act),axis=1),device='cuda',dtype=torch.float32)
hx=torch.as_tensor(np.asarray([np.r_[a,b] for a,b in heldout]),device='cuda',dtype=torch.float32)
report=dict(status='complete',no_additional_updates=True,training_rows=len(targets),training_target=dict(min=float(targets.min()),max=float(targets.max()),mean=float(targets.mean()),std=float(targets.std()),quantiles=np.quantile(targets,[0,.1,.25,.5,.75,.9,1]).tolist()),checkpoints=[])
original=SACAgent.load(out/'inputs/initial_agent.pt','cuda');final=SACAgent.load(out/'diagnostic_calibrated_critics.pt','cuda')
for name,agent in [('initial',original),('after_500',final)]:
 records=[]
 with torch.no_grad():
  for i,net in enumerate(agent.critics.networks):
   features=net[0](x);hfeatures=net[0](hx);q=net[1](features).squeeze(-1);hq=net[1](hfeatures).squeeze(-1)
   weight=net[1].weight[0];bias=float(net[1].bias);l1=float(weight.abs().sum());lower=bias-l1;upper=bias+l1
   delta=weight-original.critics.networks[i][1].weight[0]
   residual=q.cpu().numpy().astype(np.float64)-targets
   records.append(dict(critic=i+1,training_q_min=float(q.min()),training_q_max=float(q.max()),training_q_mean=float(q.mean()),training_bias=float(residual.mean()),training_mae=float(np.mean(abs(residual))),training_rmse=float(np.sqrt(np.mean(residual**2))),training_initial_state_rmse=float(np.sqrt(np.mean(residual[np.r_[True,np.diff(episode)!=0]]**2))),
    output_bias=bias,output_weight_l1=l1,output_weight_min=float(weight.min()),output_weight_max=float(weight.max()),output_weight_rms=float(torch.sqrt(torch.mean(weight**2))),output_weight_delta_max=float(delta.abs().max()),output_weight_delta_mean_abs=float(delta.abs().mean()),output_bias_delta=bias-float(original.critics.networks[i][1].bias),
    exact_head_global_q_lower_bound=lower,exact_head_global_q_upper_bound=upper,fraction_training_targets_below_global_bound=float(np.mean(targets<lower)),
    best_possible_train_rmse_with_fixed_output_head_bounds=float(np.sqrt(np.mean((np.clip(targets,lower,upper)-targets)**2))),
    penultimate_features_mean_abs=float(features.abs().mean()),penultimate_features_abs_gt_099_fraction=float((features.abs()>.99).float().mean()),penultimate_features_min=float(features.min()),penultimate_features_max=float(features.max()),
    heldout_initial_q=hq.cpu().tolist(),heldout_q_distance_from_global_lower=(hq-lower).cpu().tolist(),heldout_penultimate_mean_abs=hfeatures.abs().mean(1).cpu().tolist(),
    training_initial_state_q=q.cpu().numpy()[np.r_[True,np.diff(episode)!=0]].tolist(),training_initial_state_targets=targets[np.r_[True,np.diff(episode)!=0]].tolist()))
 report['checkpoints'].append(dict(label=name,critics=records))
(out/'capacity_analysis.json').write_text(json.dumps(report,indent=2)+'\n');shutil.copyfile(__file__,out/'inspect_capacity.py')
print(json.dumps({'targets':report['training_target'],'checkpoints':[{'label':v['label'],'critics':[{k:r[k] for k in ('critic','training_q_min','training_q_max','training_q_mean','training_bias','training_rmse','output_weight_l1','output_bias','output_weight_delta_max','output_weight_delta_mean_abs','output_bias_delta','exact_head_global_q_lower_bound','fraction_training_targets_below_global_bound','best_possible_train_rmse_with_fixed_output_head_bounds','penultimate_features_mean_abs','penultimate_features_abs_gt_099_fraction')} for r in v['critics']]} for v in report['checkpoints']]},indent=2))
