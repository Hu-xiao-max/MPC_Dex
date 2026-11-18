"""Observe the full requested hold window; failure remains latched after drop.

After environment termination, bare MuJoCo continues solely as a postmortem;
these states are never episodes, training rows, successes or demonstrations.
"""
from pathlib import Path
import json,sys
import numpy as np,mujoco
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from mpc_dex.cli import read_config
from mpc_dex.env import AllegroRotationEnv
D=Path(__file__).resolve().parent/'prefix17_hold';_,ec=read_config(D/'hold_config.json');e=AllegroRotationEnv(ec);e.reset(seed=0)
with np.load(D/'transitions.npz',allow_pickle=False) as a:actions=a['actions'].copy()
actions=np.vstack([actions,np.repeat(actions[-1:],20,axis=0)])
rows=[];failure=None;first_success=None
for index,action in enumerate(actions):
 if failure is None:
  obs,reward,done,truncated,info=e.step(action)
  if info['success'] and first_success is None:first_success=float(e.data.time)
  if done or truncated:failure=dict(control_step=index+1,time=float(e.data.time),drop=bool(info['drop']),palm_support_violation=bool(info['palm_support_violation']),reason=info['failure_reason'])
 else:
  e.data.ctrl[:]=e.to_joint_targets(action)
  for _ in range(e.substeps):mujoco.mj_step(e.model,e.data);mujoco.mj_forward(e.model,e.data)
 table=palm=0.
 for j,c in enumerate(e.data.contact):
  if e.object_geom not in (c.geom1,c.geom2):continue
  other=int(c.geom1 if c.geom2==e.object_geom else c.geom2);w=np.zeros(6);mujoco.mj_contactForce(e.model,e.data,j,w)
  if other==e.table_geom:table+=max(0.,float(w[0]))
  if e.model.geom_bodyid[other]==e.palm_id:palm+=max(0.,float(w[0]))
 rows.append([float(e.data.time),failure is not None,*e.data.xpos[e.object_id],table,palm])
np.savez_compressed(D/'full_hold_aftermath.npz',samples=np.asarray(rows),columns=np.asarray(['time','failure_latched','object_world_x','object_world_y','object_world_z','table_normal_force','palm_normal_force']))
report=dict(purpose='Full2s post-success physical hold observation; strict failure is latched and all post-failure observations are postmortem only',first_success_seconds=first_success,first_terminal_failure=failure,observation_end_seconds=float(e.data.time),post_success_observation_seconds=float(e.data.time)-first_success,completed_successful_hold=False,post_failure_samples_are_training_or_demonstration=False,final_table_normal_force=rows[-1][-2],final_palm_normal_force=rows[-1][-1])
(D/'full_hold_aftermath.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));e.close()
