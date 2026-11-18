from pathlib import Path
import hashlib,json,shutil,sys
root=Path('/home/autel001/code/mpc_dex');run=root/'artifacts/bootstrap_0005';out=root/'artifacts/bootstrap_0005_next_step_review_0001'
out.mkdir(exist_ok=False);shutil.copyfile(__file__,out/'analyze_frozen_losses.py')
sys.path.insert(0,str(run/'source'))
import numpy as np,torch
from mpc_dex.learning import SACAgent
from mpc_dex.replay import ReplayBuffer

torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False
initial=run/'checkpoints/checkpoint_00000001';final=run/'checkpoints/checkpoint_00000006'
init_replay=ReplayBuffer.load(initial/'replay_0.npz');rows=ReplayBuffer.load(final/'replay_0.npz').transitions()
assert all(np.array_equal(init_replay.transitions()[k],rows[k][:9]) for k in init_replay.transitions())
rng=np.random.default_rng();rng.bit_generator.state=json.loads((initial/'state.json').read_text())['rng_state']
counts=rng.multinomial(256,[1.]);indices=init_replay.rng.integers(9+256,size=int(counts[0]));indices=indices[rng.permutation(256)]
obs=rows['obs'][indices];np.savez_compressed(out/'actual_initial_batch_observations.npz',indices=indices,obs=obs)
inputs=[run/'config.json',run/'run/metrics.jsonl',initial/'state.json',initial/'replay_0.npz',final/'replay_0.npz',run/'source/mpc_dex/learning.py',run/'source/scripts/train_bootstrap.py',run/'source/mpc_dex/replay.py',run/'source/mpc_dex/env.py']
report=dict(status='complete',purpose='Read-only gradient decomposition, no optimizer step or physical execution',
 batch='First 256-update replay-observation sample reconstructed from initial buffer and runner RNG; 9 seed + first256 actual online rows. Actor/critics are frozen checkpoint snapshots before any hypothetical update, not gradients captured during training.',
 observation_indices=indices.tolist(),demo_rows_in_this_batch=int(np.sum(indices<9)),gradient_seed=19061,
 runtime=dict(device='cuda',threads=1,interop=1,tf32=False,torch_version=str(torch.__version__)),checkpoints=[])
for gen in ('checkpoint_00000001','checkpoint_00000003','checkpoint_00000006'):
 p=run/'checkpoints'/gen/'agent.pt';inputs.append(p);agent=SACAgent.load(p,'cuda');before={k:v.detach().clone() for k,v in agent.actor.state_dict().items()}
 torch.manual_seed(19061);torch.cuda.manual_seed_all(19061)
 tensor=torch.as_tensor(obs,device='cuda',dtype=torch.float32);actions,logp=agent.actor.sample(tensor)
 q=agent.critics(tensor,actions).mean(dim=0);entropy_term=(agent.alpha.detach()*logp).mean();q_term=-q.mean()
 named=list(agent.actor.named_parameters());params=[v for k,v in named]
 ent=torch.autograd.grad(entropy_term,params,retain_graph=True);value=torch.autograd.grad(q_term,params)
 groups={}
 for group in ('trunk','mean_head','log_std_head'):
  e=torch.cat([g.reshape(-1) for (k,v),g in zip(named,ent) if k.startswith(group+'.')]);qg=torch.cat([g.reshape(-1) for (k,v),g in zip(named,value) if k.startswith(group+'.')])
  groups[group]=dict(entropy_grad_norm=float(e.norm()),negative_q_grad_norm=float(qg.norm()),combined_grad_norm=float((e+qg).norm()),cosine=float((e@qg)/(e.norm()*qg.norm()+1e-30)))
 index={k:i for i,(k,v) in enumerate(named)}
 record=dict(checkpoint=gen,online_updates=agent.online_updates,alpha=float(agent.alpha.detach()),entropy_loss=float(entropy_term.detach()),negative_q_loss=float(q_term.detach()),batch_sample_entropy=float(-logp.detach().mean()),groups=groups,
 log_std_bias_entropy_gradient=ent[index['log_std_head.bias']].detach().cpu().tolist(),log_std_bias_negative_q_gradient=value[index['log_std_head.bias']].detach().cpu().tolist(),
 mean_bias_entropy_gradient=ent[index['mean_head.bias']].detach().cpu().tolist(),mean_bias_negative_q_gradient=value[index['mean_head.bias']].detach().cpu().tolist())
 report['checkpoints'].append(record)
 assert all(torch.equal(v,agent.actor.state_dict()[k]) for k,v in before.items())
 print(json.dumps(record),flush=True)
report['input_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
(out/'gradient_analysis.json').write_text(json.dumps(report,indent=2)+'\n')
