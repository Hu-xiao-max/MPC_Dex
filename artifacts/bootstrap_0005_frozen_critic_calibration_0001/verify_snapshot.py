from pathlib import Path
import sys,json,hashlib,shutil
out=Path('/home/autel001/code/mpc_dex/artifacts/bootstrap_0005_frozen_critic_calibration_0001');sys.path.insert(0,str(out/'source'))
import numpy as np,torch
from mpc_dex.learning import SACAgent
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
agent=SACAgent.load(out/'diagnostic_calibrated_critics.pt','cuda');checks=[]
for seed in range(17041,17046):
 d=out/f'inputs/heldout_{seed}';z=np.load(d/'transitions.npz');rng=torch.load(d/'torch_rng_initial.pt',weights_only=True)
 torch.set_rng_state(rng['cpu']);torch.cuda.set_rng_state_all(rng['cuda'])
 with torch.no_grad():
  for i,ob in enumerate(z['obs']):
   action,logp=agent.actor.sample(torch.as_tensor(ob[None,:],device='cuda',dtype=torch.float32))
   np.testing.assert_array_equal(action[0].cpu().numpy(),z['actions'][i]);np.testing.assert_array_equal(logp[0].cpu().numpy(),z['log_probs'][i])
 end=torch.load(d/'torch_rng_final.pt',weights_only=True);assert torch.equal(torch.get_rng_state(),end['cpu']) and all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),end['cuda']))
 checks.append(dict(seed=seed,actions_and_actual_logprobs_bitexact_under_trainer_shape=True,final_rng_bitexact=True))
source=torch.load(out/'inputs/initial_agent.pt',weights_only=True,map_location='cpu');final=torch.load(out/'diagnostic_calibrated_critics.pt',weights_only=True,map_location='cpu')
assert all(torch.equal(v,final['actor'][k]) for k,v in source['actor'].items());assert torch.equal(source['log_alpha'],final['log_alpha'])
for k in ('bc_updates','pretrain_updates','online_updates','std_initialized'):assert source[k]==final[k]
assert {float(x['step']) for x in final['critic_optimizer']['state'].values()}=={500.}
assert len(final['critic_optimizer']['state'])==len(final['critics'])
for k in ('bc_optimizer','actor_optimizer','alpha_optimizer'):assert source[k]==final[k]
with np.load(out/'reconstructed_preupdate_controls.npz') as z,np.load(out/'physics.npz') as p:
 nq=z['qpos'].shape[1];nv=z['qvel'].shape[1];np.testing.assert_array_equal(p['fullphysics'][49::50,1:1+nq],z['qpos']);np.testing.assert_array_equal(p['fullphysics'][49::50,1+nq:1+nq+nv],z['qvel']);np.testing.assert_array_equal(p['integration_states'][49::50],z['integration_state'])
r=dict(status='verified',heldout_policy_checks=checks,actor_alpha_counters_optimizer_states_bitexact=True,all_critic_adam_parameters_step_500=True,critic_parameter_states=len(final['critic_optimizer']['state']),fullphysics_control_state_alignment_exact=True)
(out/'verification.json').write_text(json.dumps(r,indent=2)+'\n');shutil.copyfile(__file__,out/'verify_snapshot.py');print(json.dumps(r,indent=2))
