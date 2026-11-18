"""Short physical continuations of recorded no-palm prefixes; diagnostic only."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from mpc_dex.env import EnvConfig,AllegroRotationEnv
D=Path(__file__).resolve().parent;P=D.parent
record=json.loads((P/'report.json').read_text());ec=EnvConfig(**record['environment_contract']['environment']);ec.initial_grasp=str(ROOT/'assets/initial_grasp.npz')
with np.load(P/'transitions.npz',allow_pickle=False) as a:commands=a['actions'].copy()
results=[]
for prefix in [17,18]:
 for mode,value in [('blend',0.),('blend',.25),('blend',.5),('blend',.75),('blend',1.),('delta',.1),('delta',.2),('delta',.4)]:
  e=AllegroRotationEnv(ec);obs,info=e.reset(seed=0);trace=[]
  for action in commands[:prefix]:
   obs,r,t,tr,info=e.step(action)
   if t or tr:raise RuntimeError('Original no-palm prefix unexpectedly terminated')
  before=e.data.ctrl.copy();nexttarget=e.to_joint_targets(commands[prefix]);delta=nexttarget-before
  target=before+ (value*delta if mode=='blend' else np.clip(delta,-value,value))
  action=e.to_action(target);maxangle=e.retained_angle
  for k in range(3):
   obs,r,t,tr,info=e.step(action);maxangle=max(maxangle,info['retained_angle'])
   trace.append({key:info[key] for key in ['angle','retained_angle','finger_contacts','axis_tilt_radians','drop','success','palm_support_violation','peak_palm_support_force']})
   if t or tr:break
  results.append(dict(prefix_steps=prefix,mode=mode,value=value,extra_steps=len(trace),max_control_endpoint_qualified_angle=maxangle,final=trace[-1],trace=trace,target_delta_max=float(np.max(np.abs(target-before)))))
  e.close()
report=dict(purpose='Bounded non-training diagnostic; physically replayed original prefixes, then held a blended/rate-limited next target for at most0.3s. No object-state mutation, geometry change, or criterion weakening.',environment_source_sha256=hashlib.sha256((ROOT/'mpc_dex/env.py').read_bytes()).hexdigest(),input_sha256=hashlib.sha256((P/'transitions.npz').read_bytes()).hexdigest(),results=results)
(D/'bounded_continuations.json').write_text(json.dumps(report,indent=2)+'\n')
for r in results:print(r['prefix_steps'],r['mode'],r['value'],r['extra_steps'],round(r['max_control_endpoint_qualified_angle'],6),r['final']['success'],r['final']['palm_support_violation'],r['final']['drop'])
