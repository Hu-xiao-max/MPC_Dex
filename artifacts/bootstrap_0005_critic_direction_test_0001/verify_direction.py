from pathlib import Path
import sys,json,hashlib,shutil
out=Path('/home/autel001/code/mpc_dex/artifacts/bootstrap_0005_critic_direction_test_0001');sys.path.insert(0,str(out/'source'))
import numpy as np,torch
from mpc_dex.learning import SACAgent
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
agent=SACAgent.load(out/'inputs/checkpoint.pt','cuda');r=json.loads((out/'report.json').read_text());checks=[]
for seedrecord in r['seeds']:
 seed=seedrecord['seed'];folder=out/f'seed_{seed}';base=np.load(out/f'inputs/baseline_{seed}/transitions.npz');base_noise=np.load(folder/'baseline_executed_gaussian_draws.npy');intervention=np.load(folder/'intervention.npz')
 tensor=torch.as_tensor(base['obs'][0],device='cuda',dtype=torch.float32);a0=torch.as_tensor(base['actions'][0],device='cuda',dtype=torch.float32).requires_grad_(True);g=torch.autograd.grad(agent.critics(tensor,a0).mean(),a0)[0];np.testing.assert_array_equal(g.detach().cpu().numpy(),intervention['gradient'])
 with torch.no_grad():
  for label,sign in [('plus',1),('minus',-1)]:np.testing.assert_array_equal((a0+sign*.005*g/g.norm()).clamp(-1,1).cpu().numpy(),intervention[label])
 for branch in seedrecord['branches']:
  label=branch['intervention'];d=folder/label;z=np.load(d/'transitions.npz');p=np.load(d/'physics.npz');rng=torch.load(d/'torch_rng_initial.pt',weights_only=True);torch.set_rng_state(rng['cpu']);torch.cuda.set_rng_state_all(rng['cuda'])
  with torch.no_grad():
   for i,ob in enumerate(z['obs']):
    obs=torch.as_tensor(ob,device='cuda',dtype=torch.float32);action,logp=agent.actor.sample(obs)
    np.testing.assert_array_equal(action.cpu().numpy(),z['policy_sampled_actions'][i]);np.testing.assert_array_equal(logp.cpu().numpy(),z['policy_sample_log_probs'][i])
    np.testing.assert_array_equal(z['actions'][i],intervention[label] if i==0 else z['policy_sampled_actions'][i])
  end=torch.load(d/'torch_rng_final.pt',weights_only=True);assert torch.equal(torch.get_rng_state(),end['cpu']) and all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),end['cuda']))
  n=min(len(base_noise),len(z['actions']));np.testing.assert_array_equal(z['gaussian_draws'][:n],base_noise[:n])
  T=len(z['actions']);discount=np.power(r['gamma'],np.arange(T));G=float(discount@z['rewards']-r['alpha']*(discount[1:]@z['policy_sample_log_probs'][1:]));assert abs(G-branch['observed_soft_q_return'])<1e-10
  assert z['terminated'][-1] and not z['truncated'][-1] and not z['executed_log_prob_valid'][0] and z['executed_log_prob_valid'][1:].all()
  nq=z['qpos'].shape[1];nv=z['qvel'].shape[1];np.testing.assert_array_equal(p['fullphysics'][49::50,1:1+nq],z['qpos']);np.testing.assert_array_equal(p['fullphysics'][49::50,1+nq:1+nq+nv],z['qvel']);np.testing.assert_array_equal(p['integration_states'][49::50],z['integration_state'])
  checks.append(dict(seed=seed,branch=label,steps=T,actions_density_rng_exact=True,gaussian_baseline_common_steps=n,terminal_return_exact=True,physics_control_states_exact=True,achieved_radius=branch['achieved_action_l2_radius'],clipped_dimensions=branch['physical_bound_clipped_dimensions']))
(out/'verification.json').write_text(json.dumps(dict(status='verified',branches=checks),indent=2)+'\n');shutil.copyfile(__file__,out/'verify_direction.py');print(json.dumps(checks,indent=2))
