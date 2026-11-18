"""Bounded diagnostic only: frozen-policy Monte Carlo critic calibration.

This writes critic-only diagnostic weights, never a deployable SAC checkpoint.
The actor, temperature, target networks and original input artifacts are frozen.
"""
from pathlib import Path
import hashlib,json,shutil,sys,time
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from mpc_dex.learning import SACAgent
D=Path(__file__).resolve().parent;OUT=D/'calibration_0001';OUT.mkdir(exist_ok=False)
INPUTS=OUT/'inputs';INPUTS.mkdir()
torch.set_num_threads(1);torch.manual_seed(80040);rng=np.random.default_rng(80040)

def digest_file(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest_tensors(values):
 h=hashlib.sha256()
 for name,value in sorted(values.items()):
  h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def save_json(name,obj):
 p=OUT/name;tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)
files=['agent_40000.pt','agent_40000.pt.metadata.json','report.json']+[f'rollout_stochastic_{i:02d}.npz' for i in range(12)]
for name in files:shutil.copyfile(D/name,INPUTS/name)
shutil.copyfile(__file__,OUT/Path(__file__).name)
a=SACAgent.load(INPUTS/'agent_40000.pt','cpu');alpha=float(a.alpha.detach());gamma=a.config.discount
a.actor.requires_grad_(False);a.log_alpha.requires_grad_(False)
actor_before=digest_tensors(a.actor.state_dict());target_before=digest_tensors(a.target_critics.state_dict())
logalpha_before=float(a.log_alpha.detach());optimizer_before=a.online_updates
records=json.loads((INPUTS/'report.json').read_text())['rollouts'];recordmap={r['index']:r for r in records if r['kind']=='stochastic'}
episodes=[];max_target_difference=0.
for i in range(12):
 with np.load(INPUTS/f'rollout_stochastic_{i:02d}.npz',allow_pickle=False) as ar:
  if not recordmap[i]['terminated'] or recordmap[i]['truncated']:raise ValueError('Monte Carlo calibration requires complete terminal trajectories')
  obs=ar['obs'].copy();actions=ar['actions'].copy();rewards=ar['rewards'].copy();logps=ar['log_probs'].copy();targets=ar['soft_returns'].copy()
  recomputed=np.zeros_like(targets);future=0.
  for t in range(len(targets)-1,-1,-1):
   recomputed[t]=rewards[t]+gamma*future;future=recomputed[t]-alpha*logps[t]
  max_target_difference=max(max_target_difference,float(np.max(np.abs(targets-recomputed))))
  episodes.append(dict(index=i,obs=torch.tensor(obs,dtype=torch.float32),actions=torch.tensor(actions,dtype=torch.float32),targets=torch.tensor(targets,dtype=torch.float32),exact_targets=targets))
trainobs=torch.cat([e['obs'] for e in episodes[:8]]);trainact=torch.cat([e['actions'] for e in episodes[:8]]);traintarget=torch.cat([e['targets'] for e in episodes[:8]])
settings=dict(purpose='Frozen actor critic calibration diagnostic; not a trained/deployed policy',source_checkpoint='agent_40000.pt',seed=80040,device='cpu',torch_threads=1,max_updates=2000,batch_size=256,evaluation_every=250,train_episode_ids=list(range(8)),heldout_episode_ids=list(range(8,12)),sampling='Uniform over transitions in the eight training episodes; held-out trajectories never sampled',critic_lr=a.config.critic_lr,critic_optimizer='Copied Adam state from the40k checkpoint; no reset',loss='Mean squared error over both critics and actual normalized-action soft Monte Carlo Q returns; initial action entropy excluded',alpha=alpha,discount=gamma,actor_frozen=True,alpha_frozen=True,target_networks_frozen=True,train_transitions=len(trainobs),heldout_transitions=sum(len(e['obs']) for e in episodes[8:]),input_sha256={name:digest_file(INPUTS/name) for name in files},script_sha256=digest_file(OUT/Path(__file__).name),learning_source_sha256=digest_file(ROOT/'mpc_dex/learning.py'),max_mc_target_recomputation_error=max_target_difference)
save_json('settings.json',settings)

@torch.no_grad()
def evaluate(update):
 rows=[]
 for e in episodes:
  qs=a.critics(e['obs'],e['actions']).numpy();pred=qs.mean(0);y=e['exact_targets'];err=pred-y
  rows.append(dict(episode=e['index'],split='train' if e['index']<8 else 'heldout',steps=len(y),initial_q_mean=float(pred[0]),initial_mc_return=float(y[0]),initial_q_bias=float(err[0]),transition_bias=float(err.mean()),transition_mse=float(np.mean(err**2)),transition_rmse=float(np.sqrt(np.mean(err**2))),twin_disagreement=float(np.mean(np.abs(qs[0]-qs[1])))))
 summary={}
 for split in ('train','heldout'):
  rr=[r for r in rows if r['split']==split];n=sum(r['steps'] for r in rr)
  summary[split]=dict(initial_q_mean=float(np.mean([r['initial_q_mean'] for r in rr])),initial_mc_return=float(np.mean([r['initial_mc_return'] for r in rr])),initial_mean_bias=float(np.mean([r['initial_q_bias'] for r in rr])),initial_mse=float(np.mean([r['initial_q_bias']**2 for r in rr])),transition_mean_bias=float(sum(r['steps']*r['transition_bias'] for r in rr)/n),transition_mse=float(sum(r['steps']*r['transition_mse'] for r in rr)/n),episode_equal_transition_mse=float(np.mean([r['transition_mse'] for r in rr])))
 return dict(update=update,wall_seconds=time.monotonic()-started,summary=summary,episodes=rows)

started=time.monotonic();evaluations=[evaluate(0)];save_json('metrics.json',evaluations);print(json.dumps(evaluations[0]['summary']),flush=True)
for update in range(1,2001):
 ids=torch.tensor(rng.integers(len(trainobs),size=256));values={'obs':trainobs[ids],'actions':trainact[ids]}
 loss=a._critic_step(values,traintarget[ids])
 if update%250==0:
  result=evaluate(update);result['last_minibatch_mse']=float(loss);evaluations.append(result);save_json('metrics.json',evaluations);print(json.dumps(dict(update=update,**result['summary'])),flush=True)
actor_after=digest_tensors(a.actor.state_dict());target_after=digest_tensors(a.target_critics.state_dict())
if actor_after!=actor_before or target_after!=target_before or float(a.log_alpha.detach())!=logalpha_before:raise RuntimeError('A frozen component changed')
base=evaluations[0]['summary']['heldout'];final=evaluations[-1]['summary']['heldout']
report=dict(status='complete',updates=2000,wall_seconds=time.monotonic()-started,settings=settings,baseline_heldout=base,final_heldout=final,relative_heldout_transition_mse=final['transition_mse']/base['transition_mse'],relative_heldout_initial_abs_bias=abs(final['initial_mean_bias'])/abs(base['initial_mean_bias']),calibration_decision_passed=bool(final['initial_q_mean']<0 and abs(final['initial_mean_bias'])<=.5*abs(base['initial_mean_bias'])),actor_sha256_before=actor_before,actor_sha256_after=actor_after,target_critic_sha256_before=target_before,target_critic_sha256_after=target_after,log_alpha_before=logalpha_before,log_alpha_after=float(a.log_alpha.detach()),agent_online_update_counter_unchanged=a.online_updates==optimizer_before,production_mutated=False,policy_improvement_claimed=False,limitations=['Only eight training and four held-out stochastic rollouts from one40k policy/reset state. Correlated within-episode transitions and high Monte Carlo variance limit interpretation.','Held-out episode8 is a757-step low-return trajectory outside the short training episode distribution. Both transition-weighted and episode-equal metrics are reported.','This experiment changes only critics against frozen Monte Carlo targets; it does not show that online SAC remains calibrated or that the policy rotates better.'])
torch.save(dict(purpose='diagnostic critic calibration only; not a deployable SAC checkpoint',critic_state=a.critics.state_dict(),critic_optimizer=a.critic_optimizer.state_dict(),settings=settings,updates=2000),OUT/'diagnostic_critics.pt')
report['diagnostic_critics_sha256']=digest_file(OUT/'diagnostic_critics.pt');save_json('report.json',report);print('FINAL',json.dumps(report),flush=True)
