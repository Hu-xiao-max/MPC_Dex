"""Bounded artifact-only fixed-policy empirical-return critic calibration."""
from pathlib import Path
import argparse,hashlib,json,shutil,sys,math
from datetime import datetime,timezone
ROOT=Path('/home/autel001/code/mpc_dex');RUN=ROOT/'artifacts/bootstrap_0005'
OUT=ROOT/'artifacts/bootstrap_0005_frozen_critic_calibration_0001'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def manifest():
 write(OUT/'manifest.json',{str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p!=OUT/'manifest.json'})

def prepare():
 OUT.mkdir(exist_ok=False);(OUT/'inputs').mkdir();shutil.copyfile(__file__,OUT/'calibrate.py')
 source=RUN/'source';source_hashes={str(p.relative_to(source)):sha(p) for p in sorted(source.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
 shutil.copytree(source,OUT/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 assert all(sha(source/p)==h==sha(OUT/'source'/p) for p,h in source_hashes.items())
 write(OUT/'source_hashes.json',source_hashes)
 inputs={}
 def capture(path,name):
  p=Path(path);h=sha(p);d=OUT/'inputs'/name;d.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(p,d);assert sha(p)==sha(d)==h
  inputs[name]=dict(source_path=str(p),sha256=h)
 initial=RUN/'checkpoints/checkpoint_00000001'
 for name in ('agent.pt','agent.pt.metadata.json','environment_0.npz','state.json','torch_rng.pt','replay_0.npz','archive_manifest.json'):capture(initial/name,'initial_'+name)
 capture(RUN/'checkpoints/checkpoint_00000006/replay_0.npz','actual_final_replay.npz');capture(RUN/'config.json','source_config.json')
 for seed in range(17041,17046):
  for name in ('report.json','transitions.npz','physics.npz','torch_rng_initial.pt','torch_rng_final.pt'):
   capture(RUN/f'value_calibration/initial_0001/stochastic_seed_{seed}'/name,f'heldout_{seed}/{name}')
 capture(RUN/'value_calibration/initial_0001/report.json','heldout_source_report.json')
 config=json.loads((OUT/'inputs/source_config.json').read_text());capture(config['environment']['initial_grasp'],'initial_grasp.npz')
 config['environment']['initial_grasp']=str(OUT/'inputs/initial_grasp.npz');write(OUT/'config.json',config);write(OUT/'input_bindings.json',inputs)
 sys.path.insert(0,str(OUT/'source'))
 import numpy as np,torch,mujoco
 from mpc_dex.learning import SACAgent
 from mpc_dex.env import EnvConfig,AllegroRotationEnv
 from mpc_dex.contracts import validate_checkpoint
 from mpc_dex.replay import ReplayBuffer
 from scripts.audit_policy import PhysicsObserver,PHYSICS_COLUMNS
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 ec=EnvConfig(**config['environment']);assert ec.terminate_on_success and ec.horizon_steps==100 and not config['bootstrap'].get('control')
 validate_checkpoint(OUT/'inputs/initial_agent.pt',ec)
 agent=SACAgent.load(OUT/'inputs/initial_agent.pt','cuda');assert agent.bc_updates==12000 and agent.pretrain_updates==agent.online_updates==0
 gamma=agent.config.discount;alpha=float(agent.alpha.detach());assert abs(alpha-.05)<1e-8
 allrows=ReplayBuffer.load(OUT/'inputs/actual_final_replay.npz').transitions();seedrows=ReplayBuffer.load(OUT/'inputs/initial_replay_0.npz').transitions()
 assert len(seedrows['actions'])==9
 assert all(np.array_equal(seedrows[k],allrows[k][:9]) for k in seedrows)
 recorded={k:v[9:9+256] for k,v in allrows.items()};assert len(recorded['obs'])==256
 agent_state={k:v.detach().clone() for k,v in agent.actor.state_dict().items()}
 env=AllegroRotationEnv(ec);obs,_=env.reset(seed=17031)
 spec=mujoco.mjtState.mjSTATE_INTEGRATION;fullspec=mujoco.mjtState.mjSTATE_FULLPHYSICS
 def state(specification):
  a=np.empty(mujoco.mj_stateSize(env.model,specification));mujoco.mj_getState(env.model,env.data,a,specification);return a
 with np.load(OUT/'inputs/initial_environment_0.npz',allow_pickle=False) as saved:
  np.testing.assert_array_equal(state(spec),saved['integration_state']);np.testing.assert_array_equal(obs,saved['observation'])
 rng=torch.load(OUT/'inputs/initial_torch_rng.pt',weights_only=True);torch.set_rng_state(rng['cpu']);torch.cuda.set_rng_state_all(rng['cuda'])
 records={k:[] for k in ('obs','actions','next_obs','actual_rewards','recorded_rewards','terminated','truncated','actual_log_probs','raw_mean','latent_std','qpos','qvel','ctrl','integration_state','episode_id')}
 episodes=[];physics=[];fullphysics=[];physics_integration=[];physics_episode=[];ep_start=0;episode_id=0
 oldcapture=PhysicsObserver.capture
 def capture_physics(observer):
  oldcapture(observer);fullphysics.append(state(fullspec));physics_integration.append(state(spec));physics_episode.append(episode_id)
 PhysicsObserver.capture=capture_physics;observer=PhysicsObserver(env)
 try:
  for i in range(256):
   np.testing.assert_array_equal(obs,recorded['obs'][i],err_msg=f'obs{i}')
   with torch.no_grad():
    tensor=torch.as_tensor(obs[None,:],device='cuda',dtype=torch.float32);mean,logstd=agent.actor(tensor);a,logp=agent.actor.sample(tensor)
   action=a.cpu().numpy()[0];np.testing.assert_array_equal(action,recorded['actions'][i],err_msg=f'action{i}')
   next_obs,reward,terminated,truncated,info=env.step(action)
   np.testing.assert_array_equal(next_obs,recorded['next_obs'][i],err_msg=f'next_obs{i}')
   assert np.float32(reward)==recorded['rewards'][i] and terminated==bool(recorded['terminated'][i]) and truncated==bool(recorded['truncated'][i])
   values=dict(obs=obs.copy(),actions=action.copy(),next_obs=next_obs.copy(),actual_rewards=reward,recorded_rewards=recorded['rewards'][i],terminated=terminated,truncated=truncated,
    actual_log_probs=float(logp[0]),raw_mean=mean[0].cpu().numpy(),latent_std=logstd[0].exp().cpu().numpy(),qpos=env.data.qpos.copy(),qvel=env.data.qvel.copy(),ctrl=env.data.ctrl.copy(),integration_state=state(spec),episode_id=episode_id)
   for k,v in values.items():records[k].append(v)
   obs=next_obs
   if terminated or truncated:
    assert len(observer.rows)==50*(i+1-ep_start);physics.extend(observer.rows)
    episodes.append(dict(episode=episode_id,start=ep_start,end=i+1,transitions=i+1-ep_start,terminated=terminated,truncated=truncated,success=bool(info['success']),drop=bool(info['drop']),palm_support_violation=bool(info['palm_support_violation']),failure_reason=info['failure_reason'],independent_drop=observer.independent_drop,independent_palm_support=observer.palm_support_violation,force_qualified_angle=observer.force_qualified_angle,raw_angle=observer.raw_angle,peak_palm_force=max(row[36] for row in observer.rows)))
    observer.close();ep_start=i+1;episode_id+=1
    if i<255:obs,_=env.reset();observer=PhysicsObserver(env)
  if ep_start<256:
   physics.extend(observer.rows);episodes.append(dict(episode=episode_id,start=ep_start,end=256,transitions=256-ep_start,terminated=False,truncated=False,incomplete_prefix=True))
 finally:
  observer.close();PhysicsObserver.capture=oldcapture;env.close()
 arrays={k:np.asarray(v) for k,v in records.items()};eligible=np.zeros(256,dtype=bool);targets=np.full(256,np.nan)
 for ep in episodes:
  if not ep['terminated']:continue
  start,end=ep['start'],ep['end'];eligible[start:end]=True;targets[end-1]=arrays['actual_rewards'][end-1]
  for t in range(end-2,start-1,-1):targets[t]=arrays['actual_rewards'][t]+gamma*(targets[t+1]-alpha*arrays['actual_log_probs'][t+1])
 assert eligible.any() and np.isfinite(targets[eligible]).all() and len(physics)==len(fullphysics)==256*50
 assert all(torch.equal(v,agent.actor.state_dict()[k]) for k,v in agent_state.items())
 np.savez_compressed(OUT/'reconstructed_preupdate_controls.npz',**arrays,eligible_complete_terminal=eligible,empirical_soft_q_targets=targets)
 np.savez_compressed(OUT/'physics.npz',samples=np.asarray(physics),columns=np.asarray(PHYSICS_COLUMNS),fullphysics=np.asarray(fullphysics),integration_states=np.asarray(physics_integration),episode_ids=np.asarray(physics_episode))
 torch.save(dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()),OUT/'reconstruction_final_rng.pt')
 report=dict(status='data_verified',first_online_controls=256,first_update_after_control=256,seed_rows_excluded=9,
  complete_terminal_episodes=sum(ep['terminated'] for ep in episodes),eligible_transitions=int(eligible.sum()),excluded_incomplete_or_truncated_transitions=int((~eligible).sum()),episodes=episodes,
  action_reconstruction='Exact trainer batch(1,60) actor.sample with initial saved CPU/all-CUDA RNG; every action matches bit-for-bit; direct returned logprob, no inversion or clamp',
  physics_reconstruction='Exact initial integration state; every obs/next_obs/flag exact; float32(actual reward) equals archived replay reward; all12800 physics steps retained',
  actual_reward_precision='Targets use re-executed float64 actual rewards; source replay stores rounded float32 rewards, both archived',
  target='terminal G[t]=r[t]; nonterminal G[t]=r[t]+gamma*(G[t+1]-alpha*logpi[t+1]); complete true-terminal episodes only',
  heldout_seeds=list(range(17041,17046)),heldout_used_for_fitting=False,gamma=gamma,alpha=alpha,
  planned_critic_updates=500,batch_size=256,critic_lr=agent.config.critic_lr,polyak_tau=agent.config.tau,fit_numpy_seed=17051,
  actor_frozen=True,alpha_frozen=True,bc_updates=12000,pretrain_updates=0,online_updates=0,custom_critic_calibration_updates=0,
  runtime=dict(torch_version=str(torch.__version__),numpy_version=np.__version__,mujoco_version=mujoco.__version__,device='cuda',gpu=torch.cuda.get_device_name(),torch_threads=1,torch_interop_threads=1,matmul_tf32=False,cudnn_tf32=False))
 write(OUT/'protocol.json',report);manifest();print(json.dumps({k:report[k] for k in ('status','complete_terminal_episodes','eligible_transitions','excluded_incomplete_or_truncated_transitions')}),flush=True)


def fit():
 sys.path.insert(0,str(OUT/'source'))
 import numpy as np,torch
 from mpc_dex.learning import SACAgent
 from mpc_dex.contracts import write_checkpoint_metadata
 from mpc_dex.env import EnvConfig
 existing=json.loads((OUT/'manifest.json').read_text());assert all(sha(OUT/k)==h for k,h in existing.items())
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 agent=SACAgent.load(OUT/'inputs/initial_agent.pt','cuda');config=json.loads((OUT/'config.json').read_text());ec=EnvConfig(**config['environment'])
 protocol=json.loads((OUT/'protocol.json').read_text());assert protocol['status']=='data_verified' and protocol['custom_critic_calibration_updates']==0
 with np.load(OUT/'reconstructed_preupdate_controls.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
 indices=np.flatnonzero(data['eligible_complete_terminal']);batch_obs=torch.as_tensor(data['obs'][indices],device='cuda',dtype=torch.float32);batch_actions=torch.as_tensor(data['actions'][indices],device='cuda',dtype=torch.float32);targets=torch.as_tensor(data['empirical_soft_q_targets'][indices],device='cuda',dtype=torch.float32)
 actor_before={k:v.detach().clone() for k,v in agent.actor.state_dict().items()};alpha_before=agent.log_alpha.detach().clone()
 agent.actor.requires_grad_(False);agent.log_alpha.requires_grad_(False)
 heldout=[]
 for seed in protocol['heldout_seeds']:
  with np.load(OUT/f'inputs/heldout_{seed}/transitions.npz',allow_pickle=False) as z:rows={k:z[k].copy() for k in z.files}
  assert rows['terminated'][-1] and not rows['truncated'][-1]
  T=len(rows['rewards']);g=np.empty(T);g[-1]=rows['rewards'][-1]
  for t in range(T-2,-1,-1):g[t]=rows['rewards'][t]+protocol['gamma']*(g[t+1]-protocol['alpha']*rows['log_probs'][t+1])
  source_report=json.loads((OUT/f'inputs/heldout_{seed}/report.json').read_text());assert abs(g[0]-source_report['value_calibration']['realized_soft_q_return'])<1e-10
  heldout.append((seed,rows,g))
 def assess():
  results=[]
  with torch.no_grad():
   for seed,rows,g in heldout:
    q=[];tq=[]
    for ob,action in zip(rows['obs'],rows['actions']):
     ob=torch.as_tensor(ob,device='cuda',dtype=torch.float32);action=torch.as_tensor(action,device='cuda',dtype=torch.float32)
     q.append(agent.critics(ob,action).cpu().numpy());tq.append(agent.target_critics(ob,action).cpu().numpy())
    q=np.asarray(q,dtype=np.float64);tq=np.asarray(tq,dtype=np.float64)
    results.append(dict(seed=seed,transitions=len(g),initial_q=q[0].tolist(),initial_soft_return=float(g[0]),initial_bias=(q[0]-g[0]).tolist(),trajectory_q=q.tolist(),target_trajectory_q=tq.tolist(),actual_soft_returns=g.tolist(),all_transition_rmse=np.sqrt(np.mean((q-g[:,None])**2,axis=0)).tolist()))
   fit_predictions=agent.critics(batch_obs,batch_actions).cpu().numpy();fit_targets=targets.cpu().numpy();training_mse=np.mean((fit_predictions-fit_targets[None,:])**2,axis=1).tolist()
  errors=np.asarray([r['initial_bias'] for r in results]);return dict(heldout_episodes=results,heldout_initial_bias=errors.mean(0).tolist(),heldout_initial_mae=np.abs(errors).mean(0).tolist(),heldout_initial_rmse=np.sqrt(np.mean(errors**2,axis=0)).tolist(),training_mse=training_mse)
 before=assess();write(OUT/'assessment_before.json',before);rng=np.random.default_rng(protocol['fit_numpy_seed']);metrics=[]
 for update in range(1,501):
  chosen=rng.integers(len(indices),size=256);selection=torch.as_tensor(chosen,device='cuda')
  loss=agent._critic_step(dict(obs=batch_obs[selection],actions=batch_actions[selection]),targets[selection]);agent._polyak()
  if update==1 or update%50==0:metrics.append(dict(custom_critic_calibration_updates=update,minibatch_loss=float(loss)));print(json.dumps(metrics[-1]),flush=True)
  assert agent.bc_updates==12000 and agent.pretrain_updates==agent.online_updates==0
 after=assess();write(OUT/'assessment_after.json',after);write(OUT/'training_metrics.json',metrics)
 assert all(torch.equal(v,agent.actor.state_dict()[k]) for k,v in actor_before.items()) and torch.equal(agent.log_alpha,alpha_before)
 assert not agent.actor_optimizer.state and not agent.alpha_optimizer.state and not agent.bc_optimizer.state
 agent.save(OUT/'diagnostic_calibrated_critics.pt')
 metadata=write_checkpoint_metadata(OUT/'diagnostic_calibrated_critics.pt',ec,agent.config,stage='diagnostic-frozen-policy-empirical-return-critic-calibration',policy_interface={'kind':'absolute_joint_target'})
 metadata.update(custom_critic_calibration_updates=500,actor_online_updates=0,actor_frozen=True,alpha_frozen=True,primitive_eligible=False,main_mpc_dataset_eligible=False,sac_success_claim=False,source_checkpoint_sha256=sha(OUT/'inputs/initial_agent.pt'))
 write(OUT/'diagnostic_calibrated_critics.pt.metadata.json',metadata)
 original=torch.load(OUT/'inputs/initial_agent.pt',map_location='cpu',weights_only=True);calibrated=torch.load(OUT/'diagnostic_calibrated_critics.pt',map_location='cpu',weights_only=True)
 for name in ('actor','log_alpha','bc_updates','pretrain_updates','online_updates','std_initialized'):
  if isinstance(original[name],dict):assert all(torch.equal(v,calibrated[name][k]) for k,v in original[name].items())
  elif isinstance(original[name],torch.Tensor):assert torch.equal(original[name],calibrated[name])
  else:assert original[name]==calibrated[name]
 protocol.update(status='complete',custom_critic_calibration_updates=500,actor_trunk_mean_std_alpha_bitexact=True,new_environment_controls_for_training=0,reexecuted_source_controls_for_integrity=256,new_sac_successes=0,source_checkpoint_sha256=sha(OUT/'inputs/initial_agent.pt'),calibrated_checkpoint_sha256=sha(OUT/'diagnostic_calibrated_critics.pt'))
 write(OUT/'protocol.json',protocol)
 report=dict(status='complete',protocol=protocol,before=before,after=after,
  interpretation='Fixed-policy empirical terminal return regression on actual pre-update failures; heldout episodes never fit. No actor improvement, new SAC success, or production eligibility claim.')
 write(OUT/'report.json',report);manifest();print(json.dumps(dict(status='complete',before_bias=before['heldout_initial_bias'],after_bias=after['heldout_initial_bias'],before_rmse=before['heldout_initial_rmse'],after_rmse=after['heldout_initial_rmse'])),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','fit']);args=p.parse_args()
 (prepare if args.phase=='prepare' else fit)()
