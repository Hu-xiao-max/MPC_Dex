from pathlib import Path
import sys,json,math
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from mpc_dex.env import AllegroRotationEnv,EnvConfig
from mpc_dex.learning import SACAgent
DIR=Path(__file__).resolve().parent
torch.set_num_threads(1)
state=json.loads((DIR/'state_10000.json').read_text()); ec=EnvConfig(**state['provenance']['environment'])
agent=SACAgent.load(DIR/'agent_10000.pt','cpu')

def run(mode,seed=0,max_steps=200,details=False):
 e=AllegroRotationEnv(ec);obs,_=e.reset(seed=seed);stable=e.data.ctrl.copy();stable_raw=np.arctanh(np.clip(e.to_action(stable),-.999999,.999999));rng=np.random.default_rng(seed)
 actual=e.finger_contacts;physics=[];frames=[];ret=0.
 def contacts(data=None):
  n=actual(data)
  if data is None and (not physics or e.data.time>physics[-1]['time']+1e-10):
   p=e.object_position();qa=e.object_dadr
   physics.append(dict(time=float(e.data.time),angle=float(e.angle),contacts=n,tilt=float(e.axis_tilt()),position=p.tolist(),displacement=float(np.linalg.norm(p-e.initial_position)),torque_max=float(np.max(np.abs(e.data.actuator_force))),torque_sq=float(np.sum(e.data.actuator_force**2)),joint_vel_max=float(np.max(np.abs(e.data.qvel[e.v_ids]))),object_ang_vel=float(np.linalg.norm(e.data.qvel[qa+3:qa+6])),object_lin_vel=float(np.linalg.norm(e.data.qvel[qa:qa+3]))))
  return n
 e.finger_contacts=contacts
 for k in range(max_steps):
  previous=e.data.ctrl.copy()
  if mode=='trained':action=agent.act(obs,True)
  elif mode.startswith('slew'):
   targets=e.to_joint_targets(agent.act(obs,True));limit=float(mode.split(':')[1]);action=e.to_action(previous+np.clip(targets-previous,-limit,limit))
  elif mode=='hold':action=e.to_action(stable)
  elif mode.startswith('noise'):
   action=np.tanh(stable_raw+rng.normal(0,float(mode.split(':')[1]),16))
  before=len(physics);obs,r,t,tr,info=e.step(action);ret+=r
  target=e.to_joint_targets(action)
  frames.append(dict(step=k+1,reward=r,angle=info['angle'],contacts=info['finger_contacts'],tilt=info['axis_tilt_radians'],drop=info['drop'],success=info['success'],target_delta_max=float(np.max(np.abs(target-previous))),target_delta_rms=float(np.sqrt(np.mean((target-previous)**2))),observation_slices=dict(joint_q_max=float(np.max(np.abs(obs[:16]))),object_p_max=float(np.max(np.abs(obs[16:19]))),joint_vel_max=float(np.max(np.abs(obs[25:41]))),object_vel_max=float(np.max(np.abs(obs[41:44])))),physics_start=before))
  if t or tr:break
 arr=np.array([x['angle'] for x in physics]);inc=np.diff(np.r_[0,arr]);held=np.array([x['contacts']>=2 and x['tilt']<=np.pi/6 for x in physics]);zero=np.array([x['contacts']==0 for x in physics]);badtilt=np.array([x['tilt']>np.pi/6 for x in physics]);
 def first(mask):
  ind=np.flatnonzero(mask)
  return physics[int(ind[0])] if len(ind) else None
 result=dict(mode=mode,seed=seed,steps=len(frames),return_value=ret,angle=info['angle'],drop=bool(info['drop']),success=bool(info['success']),retained_angle=float(inc[held].sum()),unretained_angle=float(inc[~held].sum()),zero_contact_angle=float(inc[zero].sum()),retained_fraction=float(held.mean()),first_zero_contact=first(zero),first_bad_tilt=first(badtilt),max_object_ang_vel=max(x['object_ang_vel'] for x in physics),max_joint_vel=max(x['joint_vel_max'] for x in physics),max_target_delta=max(x['target_delta_max'] for x in frames),max_displacement=max(x['displacement'] for x in physics))
 if details:result.update(frames=frames,physics=physics)
 e.close();return result
runs=[run('trained',max_steps=20,details=True),run('hold',max_steps=900,details=True)]
for mode in ('slew:0.02','slew:0.05','slew:0.1'):
 runs.append(run(mode,max_steps=200,details=True))
for std in (.01,.02,.05,.1):
 for seed in range(10):runs.append(run(f'noise:{std}',seed,max_steps=100))
with (DIR/'probe_results.json').open('w') as f:json.dump(dict(checkpoint='agent_10000.pt',runs=runs),f,indent=2,allow_nan=False)
for r in runs[:5]:print(json.dumps({k:v for k,v in r.items() if k not in ('physics','frames')}))
print('TRAINED FRAMES',json.dumps(runs[0]['frames']))
for std in (.01,.02,.05,.1):
 rr=[r for r in runs if r['mode']==f'noise:{std}'];print('NOISE',std,'mean_steps',np.mean([r['steps'] for r in rr]),'drops',sum(r['drop'] for r in rr),'mean_angle',np.mean([r['angle'] for r in rr]),'max_angle',max(r['angle'] for r in rr),'mean_return',np.mean([r['return_value'] for r in rr]))
