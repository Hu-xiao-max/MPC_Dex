from pathlib import Path
import sys,json,hashlib
root=Path('/home/autel001/code/mpc_dex/artifacts/bootstrap_0005/value_calibration')
sys.path.insert(0,str(root/'initial_0001/source'))
import numpy as np,torch
from mpc_dex.learning import SACAgent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False
checks={}
for name in ('initial_0001','mid_1000_0001','final_2000_0001'):
 d=root/name;r=json.loads((d/'report.json').read_text());agent=SACAgent.load(d/'checkpoint.pt','cuda')
 manifest=json.loads((d/'manifest.json').read_text());assert all(sha(d/k)==v for k,v in manifest.items())
 for e in r['episodes']:
  ep=d/('deterministic_strict' if e['deterministic'] else f"stochastic_seed_{e['seed']}")
  z=np.load(ep/'transitions.npz');p=np.load(ep/'physics.npz')
  rng=torch.load(ep/'torch_rng_initial.pt',weights_only=True);torch.set_rng_state(rng['cpu']);torch.cuda.set_rng_state_all(rng['cuda'])
  with torch.no_grad():
   for i,ob in enumerate(z['obs']):
    o=torch.as_tensor(ob,device='cuda',dtype=torch.float32);mean,logstd=agent.actor(o)
    if e['deterministic']:action=mean.tanh();logp=agent.actor._log_prob(mean,logstd,mean)
    else:action,logp=agent.actor.sample(o)
    for got,key in [(action,'actions'),(mean,'raw_mean'),(mean.tanh(),'mean_actions'),(logstd.exp(),'latent_std'),(logp,'log_probs'),(agent.critics(o,action),'critic_q'),(agent.target_critics(o,action),'target_critic_q')]:
     np.testing.assert_array_equal(got.cpu().numpy(),z[key][i],err_msg=str((name,e['seed'],i,key)))
  end=torch.load(ep/'torch_rng_final.pt',weights_only=True);assert torch.equal(torch.get_rng_state(),end['cpu'])
  assert all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),end['cuda']))
  value=e['value_calibration'];gamma=value['gamma'];alpha=value['alpha'];T=len(z['rewards'])
  raw=sum((gamma**t)*float(z['rewards'][t]) for t in range(T))
  soft=raw-alpha*sum((gamma**t)*float(z['log_probs'][t]) for t in range(1,T))
  assert abs(raw-value['realized_discounted_reward'])<1e-10 and abs(soft-value['realized_soft_q_return'])<1e-10
  assert z['terminated'][-1] and not z['truncated'][-1] and not value['missing_continuation_tail']
  nq=z['qpos'].shape[1];nv=z['qvel'].shape[1]
  np.testing.assert_array_equal(p['fullphysics'][49::50,1:1+nq],z['qpos'])
  np.testing.assert_array_equal(p['fullphysics'][49::50,1+nq:1+nq+nv],z['qvel'])
  np.testing.assert_array_equal(p['integration_states'][49::50],z['integration_state'])
  assert len(p['samples'])==len(p['fullphysics'])==50*T
 checks[name]=dict(manifest_files=len(manifest),episodes=6,all_actions_logprobs_critics_rng_bitexact=True,discounted_returns_verified=True,fullphysics_control_states_exact=True,all_true_terminal=True)
print(json.dumps(checks,indent=2))
