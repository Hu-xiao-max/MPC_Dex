from pathlib import Path
import sys,json,numpy as np,torch,math
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from mpc_dex.learning import SACAgent
from mpc_dex.replay import ReplayBuffer
D=Path(__file__).resolve().parent;torch.set_num_threads(1)
a=SACAgent.load(D/'agent_40000.pt','cpu');b=ReplayBuffer.load(D/'replay_0_40000.npz').sample(512)
o=torch.tensor(b['next_obs']);r=torch.tensor(b['rewards']);done=torch.tensor(b['terminated']).bool();bootstrap=~done
seed=423;torch.manual_seed(seed)
with torch.no_grad():
 mean,ls=a.actor(o[bootstrap]);raw=torch.distributions.Normal(mean,ls.exp()).rsample();action=raw.tanh()
 # Equivalent change-of-variables density evaluated independently where the
 # direct Jacobian is well conditioned; saturated actions use the stable form.
 direct=(torch.distributions.Normal(mean,ls.exp()).log_prob(raw)-torch.log1p(-action.square())).sum(-1)
 stable=a.actor._log_prob(mean,ls,raw);well=(action.abs()<.999).all(-1)
 selected=torch.randperm(a.config.ensemble_size)[:a.config.online_target_critics]
 nextq=a.target_critics(o[bootstrap],action)[selected].min(0).values
 expected=r.clone();expected[bootstrap]+=a.config.discount*(nextq-a.alpha*stable)
 targets_before=[p.clone() for p in a.target_critics.parameters()]
 captured={};original=a._critic_step
 def capture(values,targets):
  captured['targets']=targets.detach().clone();return original(values,targets)
 a._critic_step=capture
 torch.manual_seed(seed)
metrics=a.update(b)
with torch.no_grad():
 polyak_error=max(float((target-((1-a.config.tau)*before+a.config.tau*current)).abs().max()) for target,before,current in zip(a.target_critics.parameters(),targets_before,a.critics.parameters()))
result=dict(batch_size=len(r),terminal_rows=int(done.sum()),target_formula_max_error=float((captured['targets']-expected).abs().max()),terminal_target_max_error=float((captured['targets'][done]-r[done]).abs().max()) if done.any() else None,log_probability_well_conditioned_rows=int(well.sum()),log_probability_direct_jacobian_max_error=float((direct[well]-stable[well]).abs().max()) if well.any() else None,polyak_max_error=polyak_error,metrics=metrics,note='One update on an independent in-memory copy. No production checkpoint, replay, source or live process changed.')
(D/'update_math.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
