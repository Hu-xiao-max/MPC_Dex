"""Ten fixed-radius first-action interventions with paired policy noise."""
from pathlib import Path
import hashlib,json,shutil,sys
from datetime import datetime,timezone
BASE=Path('/home/autel001/code/mpc_dex/artifacts');SOURCE=BASE/'bootstrap_0005_centered_critic_calibration_0001';OUT=BASE/'bootstrap_0005_critic_direction_test_0001'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 manifest=json.loads((SOURCE/'manifest.json').read_text());assert all(sha(SOURCE/p)==h for p,h in manifest.items())
 OUT.mkdir(exist_ok=False);shutil.copyfile(__file__,OUT/'audit_direction.py');shutil.copytree(SOURCE/'source',OUT/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'));(OUT/'inputs').mkdir()
 inputs={}
 def capture(p,name):
  dest=OUT/'inputs'/name;dest.parent.mkdir(parents=True,exist_ok=True);h=sha(p);shutil.copyfile(p,dest);assert sha(p)==sha(dest)==h;inputs[name]=dict(source_path=str(p),sha256=h)
 for orig,dest in [('diagnostic_calibrated_critics.pt','checkpoint.pt'),('diagnostic_calibrated_critics.pt.metadata.json','checkpoint.pt.metadata.json'),('inputs/initial_agent.pt','initial_agent.pt'),('inputs/initial_grasp.npz','initial_grasp.npz'),('config.json','source_config.json'),('manifest.json','critic_source_manifest.json')]:capture(SOURCE/orig,dest)
 for seed in range(17041,17046):
  for name in ('transitions.npz','physics.npz','report.json','torch_rng_initial.pt','torch_rng_final.pt'):capture(SOURCE/f'inputs/heldout_{seed}'/name,f'baseline_{seed}/{name}')
 config=json.loads((OUT/'inputs/source_config.json').read_text());config['environment']['initial_grasp']=str(OUT/'inputs/initial_grasp.npz');write(OUT/'config.json',config);write(OUT/'input_bindings.json',inputs)
 sys.path.insert(0,str(OUT/'source'))
 import numpy as np,torch,mujoco
 from mpc_dex.learning import SACAgent
 from mpc_dex.env import EnvConfig,AllegroRotationEnv
 from mpc_dex.contracts import validate_checkpoint,environment_contract
 from scripts import audit_policy
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 ec=EnvConfig(**config['environment']);assert ec.terminate_on_success and ec.horizon_steps==100 and not config['bootstrap'].get('control')
 validate_checkpoint(OUT/'inputs/checkpoint.pt',ec);agent=SACAgent.load(OUT/'inputs/checkpoint.pt','cuda')
 initial=torch.load(OUT/'inputs/initial_agent.pt',map_location='cpu',weights_only=True)
 assert all(torch.equal(v.detach().cpu(),initial['actor'][k]) for k,v in agent.actor.state_dict().items()) and torch.equal(agent.log_alpha.detach().cpu(),initial['log_alpha'])
 assert agent.bc_updates==12000 and agent.pretrain_updates==agent.online_updates==0
 agent.actor.requires_grad_(False);agent.critics.requires_grad_(False);agent.log_alpha.requires_grad_(False)
 alpha=float(agent.alpha.detach());gamma=agent.config.discount;radius=.005
 def reset_rng(record):torch.set_rng_state(record['cpu']);torch.cuda.set_rng_state_all(record['cuda'])
 def rng_state():return dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
 def rng_equal(a,b):return torch.equal(a['cpu'],b['cpu']) and all(torch.equal(x,y) for x,y in zip(a['cuda'],b['cuda']))
 @torch.no_grad()
 def sample(obs):
  tensor=torch.as_tensor(obs,device='cuda',dtype=torch.float32);mean,logstd=agent.actor(tensor)
  noise=torch.distributions.Normal(torch.zeros_like(mean),torch.ones_like(mean)).rsample()
  raw=mean+noise*logstd.exp();action=raw.tanh();logp=agent.actor._log_prob(mean,logstd,raw)
  return action.cpu().numpy(),float(logp),noise.cpu().numpy(),mean.cpu().numpy(),logstd.exp().cpu().numpy()
 @torch.no_grad()
 def q_value(obs,action):return agent.critics(torch.as_tensor(obs,device='cuda',dtype=torch.float32),torch.as_tensor(action,device='cuda',dtype=torch.float32)).cpu().numpy()
 report=dict(status='running',purpose='Physical first-action directionality test of fitted critic gradient',radius=radius,normalized_action_norm='L2',gradient='gradient of mean of both centered fitted critics with respect to original actually sampled first action',
  prespecified_seeds=list(range(17041,17046)),branches_per_seed=['plus','minus'],maximum_controls_per_branch=100,
  physics_interventions_only=True,policy_only=False,sac_success_claim=False,production_eligible=False,new_learning_updates=0,
  actor_source_online_updates=0,critic_custom_updates=500,alpha=alpha,gamma=gamma,
  clipping='Only actual normalized physical bounds[-1,1]; record achieved radius and clipped dimensions',
  entropy_convention='Initial intervention has no current-action entropy in Q; use only actual future raw-sample log probabilities at t>=1',
  noise_pairing='Assert identical standardized Gaussian draws only on commonly executed steps. Beyond baseline termination, continue original seeded RNG stream without inventing baseline actions or draws.',
  truncation='Nonterminal100-step limits yield observed partial returns, without a terminal Q-return comparison or invented tail.',seeds=[])
 runtime=dict(created_at=datetime.now(timezone.utc).isoformat(),torch_version=str(torch.__version__),numpy_version=np.__version__,mujoco_version=mujoco.__version__,device='cuda',gpu=torch.cuda.get_device_name(),torch_threads=1,torch_interop_threads=1,matmul_tf32=False,cudnn_tf32=False,environment_contract=environment_contract(ec),critic_checkpoint_sha256=sha(OUT/'inputs/checkpoint.pt'),source_manifest_sha256=sha(SOURCE/'manifest.json'))
 write(OUT/'runtime.json',runtime);write(OUT/'report.json',report)
 for seed in report['prespecified_seeds']:
  folder=OUT/f'seed_{seed}';folder.mkdir();baseline_folder=OUT/f'inputs/baseline_{seed}'
  with np.load(baseline_folder/'transitions.npz') as z:baseline={k:z[k].copy() for k in z.files}
  baseline_report=json.loads((baseline_folder/'report.json').read_text());initial_rng=torch.load(baseline_folder/'torch_rng_initial.pt',weights_only=True);final_rng=torch.load(baseline_folder/'torch_rng_final.pt',weights_only=True)
  reset_rng(initial_rng);baseline_noise=[]
  for i,obs in enumerate(baseline['obs']):
   action,logp,noise,mean,std=sample(obs);np.testing.assert_array_equal(action,baseline['actions'][i]);np.testing.assert_array_equal(np.asarray(logp),baseline['log_probs'][i]);baseline_noise.append(noise)
  assert rng_equal(rng_state(),final_rng);baseline_noise=np.asarray(baseline_noise);np.save(folder/'baseline_executed_gaussian_draws.npy',baseline_noise)
  tensor=torch.as_tensor(baseline['obs'][0],device='cuda',dtype=torch.float32);a0=torch.as_tensor(baseline['actions'][0],device='cuda',dtype=torch.float32).clone().requires_grad_(True)
  q=agent.critics(tensor,a0);gradient=torch.autograd.grad(q.mean(),a0)[0];norm=gradient.norm();assert torch.isfinite(norm) and norm>0
  direction=gradient/norm;forced={}
  with torch.no_grad():
   for label,sign in [('plus',1),('minus',-1)]:forced[label]=(a0.detach()+sign*radius*direction).clamp(-1,1).cpu().numpy()
  np.savez_compressed(folder/'intervention.npz',original_action=baseline['actions'][0],gradient=gradient.detach().cpu().numpy(),unit_direction=direction.detach().cpu().numpy(),plus=forced['plus'],minus=forced['minus'])
  seed_record=dict(seed=seed,baseline_action_logprob_final_rng_bitexact=True,baseline_steps=len(baseline['actions']),baseline=baseline_report,
   baseline_centered_critic_q=q_value(baseline['obs'][0],baseline['actions'][0]).tolist(),gradient_norm=float(norm),branches=[])
  saved_branch_noise={}
  for label in ('plus','minus'):
   output=folder/label;output.mkdir();reset_rng(initial_rng);torch.save(rng_state(),output/'torch_rng_initial.pt')
   env=AllegroRotationEnv(ec);records={k:[] for k in ('obs','actions','policy_sampled_actions','policy_sample_log_probs','executed_log_prob_valid','gaussian_draws','mean_actions','latent_std','next_obs','rewards','terminated','truncated','qpos','qvel','targets','integration_state')};initial_values={};physics_states=[];physics_integration=[]
   spec=mujoco.mjtState.mjSTATE_INTEGRATION;fullspec=mujoco.mjtState.mjSTATE_FULLPHYSICS
   def state(specification):
    values=np.empty(mujoco.mj_stateSize(env.model,specification));mujoco.mj_getState(env.model,env.data,values,specification);return values
   original_reset,original_step=env.reset,env.step;oldcapture=audit_policy.PhysicsObserver.capture
   def capture_physics(observer):oldcapture(observer);physics_states.append(state(fullspec));physics_integration.append(state(spec))
   def reset(*a,**kw):
    result=original_reset(*a,**kw);np.testing.assert_array_equal(result[0],baseline['obs'][0]);np.testing.assert_array_equal(state(spec),baseline['initial_integration_state'])
    initial_values.update(obs=result[0].copy(),qpos=env.data.qpos.copy(),qvel=env.data.qvel.copy(),ctrl=env.data.ctrl.copy(),integration_state=state(spec));return result
   def step(action):
    result=original_step(action);ob,reward,terminated,truncated,info=result
    for k,v in dict(next_obs=ob.copy(),rewards=reward,terminated=terminated,truncated=truncated,qpos=env.data.qpos.copy(),qvel=env.data.qvel.copy(),targets=env.data.ctrl.copy(),integration_state=state(spec)).items():records[k].append(v)
    return result
   class Policy:
    def act(self,obs,deterministic=False):
     index=len(records['actions']);action,logp,noise,mean,std=sample(obs)
     if index<len(baseline_noise):np.testing.assert_array_equal(noise,baseline_noise[index])
     if index==0:np.testing.assert_array_equal(action,baseline['actions'][0]);executed=forced[label].copy()
     else:executed=action.copy()
     for k,v in dict(obs=obs.copy(),actions=executed,policy_sampled_actions=action,policy_sample_log_probs=logp,executed_log_prob_valid=index>0,gaussian_draws=noise,mean_actions=np.tanh(mean),latent_std=std).items():records[k].append(v)
     return executed
   env.reset,env.step=reset,step;audit_policy.PhysicsObserver.capture=capture_physics
   try:result,physics,commands=audit_policy.audit_episode(env,Policy(),seed,100,False)
   finally:env.close();audit_policy.PhysicsObserver.capture=oldcapture
   arrays={k:np.asarray(v) for k,v in records.items()};T=len(arrays['actions']);discounts=np.power(gamma,np.arange(T));raw_return=float(discounts@arrays['rewards']);soft_return=float(raw_return-alpha*(discounts[1:]@arrays['policy_sample_log_probs'][1:]));terminal=bool(arrays['terminated'][-1]);delta=forced[label].astype(np.float64)-baseline['actions'][0].astype(np.float64)
   qforced=q_value(baseline['obs'][0],forced[label]);qbase=np.asarray(seed_record['baseline_centered_critic_q']);baseline_return=baseline_report['value_calibration']['realized_soft_q_return']
   result.update(intervention=label,policy_only=False,sac_success=False,new_learning_updates=0,initial_action_q=qforced.tolist(),initial_action_q_mean=float(qforced.mean()),predicted_q_mean_delta=float(qforced.mean()-qbase.mean()),
    achieved_action_l2_radius=float(np.linalg.norm(delta)),actual_action_delta=delta.tolist(),physical_bound_clipped_dimensions=int(np.sum((forced[label]==-1)|(forced[label]==1))),
    realized_discounted_reward=raw_return,observed_soft_q_return=soft_return,terminal_return_comparison_eligible=terminal,
    soft_return_delta_vs_baseline=float(soft_return-baseline_return) if terminal else None,missing_continuation_tail=not terminal,missing_tail_discount=0. if terminal else float(gamma**T),
    baseline_common_executed_gaussian_steps=min(T,len(baseline_noise)),baseline_common_gaussian_draws_bitexact=True,
    future_policy='Original unchanged BC actor with std.05 and original seeded Gaussian stream; intervention only atcontrol1',
    first_logprob_note='policy_sample_log_probs[0] refers to the original unexecuted sample and is excluded from soft-Q return; all t>=1 values correspond to actual executed policy actions')
   assert len(physics)==len(physics_states)==len(physics_integration)==50*T
   np.savez_compressed(output/'transitions.npz',**arrays,**{'initial_'+k:v for k,v in initial_values.items()});np.savez_compressed(output/'physics.npz',samples=physics,columns=np.asarray(audit_policy.PHYSICS_COLUMNS),fullphysics=np.asarray(physics_states),integration_states=np.asarray(physics_integration));write(output/'commands.json',commands);write(output/'report.json',result);torch.save(rng_state(),output/'torch_rng_final.pt')
   saved_branch_noise[label]=arrays['gaussian_draws'];seed_record['branches'].append(result)
   print(json.dumps(dict(seed=seed,branch=label,steps=T,success=result['finger_supported_rotation_success'],failure=result['environment_failure_reason'],predicted_delta=result['predicted_q_mean_delta'],actual_soft_delta=result['soft_return_delta_vs_baseline'],force_angle=result['max_force_qualified_angle'])),flush=True)
  n=min(map(len,saved_branch_noise.values()));np.testing.assert_array_equal(saved_branch_noise['plus'][:n],saved_branch_noise['minus'][:n]);seed_record['plus_minus_common_executed_gaussian_steps']=n;seed_record['plus_minus_common_draws_bitexact']=True
  report['seeds'].append(seed_record);write(OUT/'report.json',report)
 assert len(report['seeds'])==5 and all(len(s['branches'])==2 for s in report['seeds'])
 assert all(torch.equal(v.detach().cpu(),initial['actor'][k]) for k,v in agent.actor.state_dict().items()) and torch.equal(agent.log_alpha.detach().cpu(),initial['log_alpha'])
 assert all(sha(SOURCE/p)==h for p,h in manifest.items())
 report.update(status='complete',new_physical_intervention_episodes=10,new_baseline_physical_episodes=0,all_original_artifacts_unchanged=True,actor_alpha_bitexact=True)
 write(OUT/'report.json',report);write(OUT/'manifest.json',{str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p!=OUT/'manifest.json'})
if __name__=='__main__':main()
