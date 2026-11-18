from pathlib import Path
import sys,json,shutil,hashlib
base=Path('/home/autel001/code/mpc_dex/artifacts');old=base/'bootstrap_0005_frozen_critic_calibration_0001';out=base/'bootstrap_0005_centered_critic_calibration_0001';sys.path.insert(0,str(out/'source'))
import numpy as np,torch
from mpc_dex.learning import SACAgent
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
z=np.load(out/'reconstructed_preupdate_controls.npz');obs=z['obs'];actual=z['actions'];rng=torch.load(out/'inputs/initial_torch_rng.pt',weights_only=True)
report=dict(status='complete',no_updates=True,observations='Same256 actual pre-update observations for every probe',actions='Same256 actual executed actions, reproduced with saved initial RNG and sequential trainer-shaped actor.sample calls',interpretation='Frozen loss derivative magnitudes only; no claim about physically correct action direction.',results=[])
for label,path in [('original_initial',out/'inputs/initial_agent.pt'),('centered_initial',out/'centered_initial.pt'),('uncentered_500',old/'diagnostic_calibrated_critics.pt'),('centered_500',out/'diagnostic_calibrated_critics.pt')]:
 agent=SACAgent.load(path,'cuda');torch.set_rng_state(rng['cpu']);torch.cuda.set_rng_state_all(rng['cuda']);action_rows=[];logp_rows=[]
 for i,row in enumerate(obs):
  a,lp=agent.actor.sample(torch.as_tensor(row[None,:],device='cuda',dtype=torch.float32));np.testing.assert_array_equal(a.detach().cpu().numpy()[0],actual[i]);action_rows.append(a);logp_rows.append(lp)
 actions=torch.cat(action_rows);logp=torch.cat(logp_rows);tensor=torch.as_tensor(obs,device='cuda',dtype=torch.float32)
 q=agent.critics(tensor,actions).mean(0);eloss=(agent.alpha.detach()*logp).mean();qloss=-q.mean();named=list(agent.actor.named_parameters());params=[v for k,v in named]
 eg=torch.autograd.grad(eloss,params,retain_graph=True);qg=torch.autograd.grad(qloss,[*params,actions]);actiongrad=qg[-1];qg=qg[:-1]
 groups={}
 for group in ('trunk','mean_head','log_std_head'):
  e=torch.cat([g.reshape(-1) for (k,v),g in zip(named,eg) if k.startswith(group+'.')]);g=torch.cat([g.reshape(-1) for (k,v),g in zip(named,qg) if k.startswith(group+'.')])
  groups[group]=dict(entropy_gradient_l2=float(e.norm()),negative_q_gradient_l2=float(g.norm()),combined_gradient_l2=float((e+g).norm()))
 report['results'].append(dict(label=label,checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),all_executed_actions_bitexact=True,groups=groups,negative_q_action_gradient_l2=float(actiongrad.norm()),negative_q_action_gradient_mean_abs=float(actiongrad.abs().mean())))
print(json.dumps(report,indent=2));(out/'gradient_comparison.json').write_text(json.dumps(report,indent=2)+'\n');shutil.copyfile(__file__,out/'compare_gradients.py')
