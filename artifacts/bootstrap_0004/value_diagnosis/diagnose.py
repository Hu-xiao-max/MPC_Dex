from pathlib import Path
import sys,json,math,time
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from mpc_dex.bootstrap_env import environment_for_policy
from mpc_dex.cli import read_config
from mpc_dex.contracts import validate_checkpoint,sha256
from mpc_dex.learning import SACAgent
from mpc_dex.replay import ReplayBuffer
D=Path(__file__).resolve().parent;torch.set_num_threads(1);torch.manual_seed(41)
_,ec=read_config(ROOT/'configs/bootstrap_incremental_seeded.json');cp=D/'agent_40000.pt';metadata=validate_checkpoint(cp,ec)
a=SACAgent.load(cp,'cpu');alpha=float(a.alpha.detach());gamma=a.config.discount

def stats(x):
 x=np.asarray(x);return dict(mean=float(x.mean()),std=float(x.std()),min=float(x.min()),p10=float(np.quantile(x,.1)),median=float(np.median(x)),p90=float(np.quantile(x,.9)),max=float(x.max()))

@torch.no_grad()
def policy_stats(obs,actions):
 o=torch.as_tensor(obs,dtype=torch.float32);ac=torch.as_tensor(actions,dtype=torch.float32)
 estimates=a.critics(o,ac).numpy();target=a.target_critics(o,ac).numpy()
 mean,ls=a.actor(o)
 return dict(q_mean=estimates.mean(0),q_min=estimates.min(0),q_disagreement=estimates.max(0)-estimates.min(0),target_q_min=target.min(0),mean_actions=mean.tanh().numpy(),raw_std=ls.exp().numpy())

@torch.no_grad()
def rollout(kind,index):
 env=environment_for_policy(ec,metadata['policy_interface']);obs,_=env.reset(seed=2000000+index);torch.manual_seed(7000+index)
 observations=[];actions=[];rewards=[];logps=[];infos=[]
 for t in range(ec.horizon_steps):
  ot=torch.as_tensor(obs)
  if kind=='deterministic':
   mean,ls=a.actor(ot);action=mean.tanh();lp=a.actor._log_prob(mean,ls,mean)
  else:action,lp=a.actor.sample(ot)
  action=action.numpy();observations.append(obs.copy());actions.append(action);logps.append(float(lp))
  obs,r,term,trunc,info=env.step(action);rewards.append(r);infos.append(info)
  if term or trunc:break
 q=policy_stats(observations,actions);soft=np.zeros(len(rewards));plain=np.zeros(len(rewards));future=ordinary=0.
 for t in range(len(rewards)-1,-1,-1):
  plain[t]=rewards[t]+gamma*ordinary;ordinary=plain[t]
  soft[t]=rewards[t]+gamma*future;future=soft[t]-alpha*logps[t]
 result=dict(kind=kind,index=index,steps=len(rewards),terminated=bool(term),truncated=bool(trunc),drop=info['drop'],success=info['success'],return_value=sum(rewards),discounted_return=float(plain[0]),soft_return_excluding_initial_entropy=float(soft[0]),initial_q_mean=float(q['q_mean'][0]),initial_q_min=float(q['q_min'][0]),initial_q_disagreement=float(q['q_disagreement'][0]),initial_log_prob=logps[0],mean_log_prob=float(np.mean(logps)),max_retained_angle=max(x['retained_angle'] for x in infos),final_retained_angle=info['retained_angle'],q_mean_minus_mc_soft=stats(q['q_mean']-soft),q_min_minus_mc_soft=stats(q['q_min']-soft),critic_disagreement=stats(q['q_disagreement']),raw_policy_std=stats(q['raw_std']))
 np.savez_compressed(D/f'rollout_{kind}_{index:02d}.npz',obs=observations,actions=actions,rewards=rewards,log_probs=logps,discounted_returns=plain,soft_returns=soft,**q)
 env.close();print(json.dumps(result),flush=True);return result

@torch.no_grad()
def replay_diagnosis():
 b=ReplayBuffer.load(D/'replay_0_40000.npz');rows=b.transitions();n=len(rows['rewards']);obs=torch.as_tensor(rows['obs']);act=torch.as_tensor(rows['actions']);nobs=torch.as_tensor(rows['next_obs']);term=rows['terminated'];r=rows['rewards'];statsq=policy_stats(rows['obs'],rows['actions'])
 behavior_lp=a.actor.log_prob(obs,act).numpy();m,ls=a.actor(obs);meanq=a.critics(obs,m.tanh()).numpy()
 actorq=[];targets=[];next_qs=[];entropy_terms=[]
 for _ in range(16):
  policyact,lp=a.actor.sample(obs);actorq.append(a.critics(obs,policyact).mean(0).numpy())
  na,nlp=a.actor.sample(nobs);nq=a.target_critics(nobs,na).min(0).values.numpy();en=-alpha*nlp.numpy()
  targets.append(r+gamma*(~term)*(nq+en));next_qs.append(nq);entropy_terms.append(en)
 target=np.mean(targets,0);actq=np.mean(actorq,0)
 done=rows['terminated']|rows['truncated'];starts=np.r_[0,np.flatnonzero(done)[:-1]+1];ends=np.flatnonzero(done);episode_results=[]
 for start,end in zip(starts,ends):
  if not term[end]:continue
  rr=r[start:end+1];discounted=float(np.dot(gamma**np.arange(len(rr)),rr));episode_results.append(dict(start=int(start),end=int(end),steps=len(rr),return_value=float(rr.sum()),discounted_behavior_return=discounted,current_q_behavior_start=float(statsq['q_mean'][start]),current_q_actor_start=float(actq[start])))
 out=dict(transitions=n,terminal_fraction=float(term.mean()),truncated_fraction=float(rows['truncated'].mean()),reward=stats(r),behavior_action_q_mean=stats(statsq['q_mean']),actor_mean_action_q=stats(meanq.mean(0)),actor_sampled_action_q=stats(actq),actor_minus_behavior_q=stats(actq-statsq['q_mean']),target_next_q_min=stats(np.mean(next_qs,0)),next_entropy_term=stats(np.mean(entropy_terms,0)),mean_sampled_bellman_target=stats(target),sampled_bellman_residual=stats(target-statsq['q_mean']),target_mc_sampling_std=stats(np.std(targets,0)),terminal_q=stats(statsq['q_mean'][term]),terminal_true_reward=stats(r[term]),terminal_residual=stats(r[term]-statsq['q_mean'][term]),q_disagreement=stats(statsq['q_disagreement']),current_policy_log_prob_replay_actions=stats(behavior_lp),completed_episodes=episode_results)
 np.savez_compressed(D/'replay_values.npz',bellman_target=target,actor_sampled_q=actq,actor_mean_action_q=meanq.mean(0),behavior_current_policy_log_prob=behavior_lp,**statsq)
 return out

report=dict(checkpoint_sha256=sha256(cp),environment_steps=40000,alpha=alpha,gamma=gamma,config=a.config.__dict__,inference_device='cpu',notes=['Soft Monte Carlo Q excludes entropy of the initial action, matching SAC Bellman convention. Stochastic episodes sample the saved actor and are on-policy; deterministic continuation returns are descriptive only.','Terminal trajectories yield complete episode returns. Truncations are finite-horizon lower-evidence estimates, not infinite-horizon ground truth.','Replay behavior episode returns come from changing old policies, so they are not unbiased returns for the current actor.'],replay=replay_diagnosis(),rollouts=[])
(D/'report.json').write_text(json.dumps(report,indent=2)+'\n')
for kind,index in [('deterministic',0)]+[('stochastic',i) for i in range(12)]:
 report['rollouts'].append(rollout(kind,index));(D/'report.json').write_text(json.dumps(report,indent=2)+'\n')
rr=[x for x in report['rollouts'] if x['kind']=='stochastic' and x['terminated']]
report['stochastic_terminal_summary']={key:stats([x[key] for x in rr]) for key in ('steps','discounted_return','soft_return_excluding_initial_entropy','initial_q_mean','initial_q_min','initial_q_disagreement','max_retained_angle')}
report['complete']=True;(D/'report.json').write_text(json.dumps(report,indent=2)+'\n');print('SUMMARY',json.dumps(report['stochastic_terminal_summary']),flush=True)
