from pathlib import Path
import sys,json,numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from mpc_dex.env import EnvConfig
from mpc_dex.bootstrap_env import BootstrapRotationEnv,BootstrapControlConfig
D=Path(__file__).resolve().parent
state=json.loads((D/'state_10000.json').read_text());config=EnvConfig(**state['provenance']['environment'])
rows=[]
for seed in range(10):
 e=BootstrapRotationEnv(config,BootstrapControlConfig());o,_=e.reset(seed=seed);rng=np.random.default_rng(seed);maxdelta=maxerror=0.
 for step in range(200):
  q=e.data.qpos[e.q_ids].copy();prev=e.data.ctrl.copy();a=np.tanh(rng.normal(0,.05,16));target=e.joint_targets_for_increment(a);maxdelta=max(maxdelta,np.max(np.abs(target-prev)));maxerror=max(maxerror,np.max(np.abs(target-q)));o,r,t,tr,info=e.step(a)
  if t or tr:break
 rows.append(dict(seed=seed,steps=step+1,angle=info['angle'],drop=info['drop'],contacts=info['finger_contacts'],tilt=info['axis_tilt_radians'],max_target_delta=maxdelta,max_target_error=maxerror));e.close()
(D/'increment_noise_probe.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(dict(episodes=len(rows),mean_steps=np.mean([r['steps'] for r in rows]),drops=sum(r['drop'] for r in rows),max_delta=max(r['max_target_delta'] for r in rows),max_target_error=max(r['max_target_error'] for r in rows),rows=rows),indent=2))
