from pathlib import Path
import sys,json,hashlib,math
import numpy as np,mujoco
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from mpc_dex.env import AllegroRotationEnv,EnvConfig
D=Path(__file__).resolve().parent;P=D.parent
report=json.loads((P/'report.json').read_text());ec=EnvConfig(**report['environment_contract']['environment']);ec.initial_grasp=str(ROOT/'assets/initial_grasp.npz')
env=AllegroRotationEnv(ec);env.reset(seed=0);model,data=env.model,env.data
palm_geoms=[i for i in range(model.ngeom) if model.geom_bodyid[i]==env.palm_id and (model.geom_contype[i] or model.geom_conaffinity[i])]
with np.load(P/'transitions.npz',allow_pickle=False) as a:actions=a['actions'].copy();expected_qpos=a['qpos'].copy();expected_qvel=a['qvel'].copy()
with np.load(P/'physics.npz',allow_pickle=False) as a:old=a['samples'].copy();columns=a['columns'].tolist();oldcols=dict(zip(columns,old.T))
rows=[];details=[];maxstateerror=0.;maxforceerror=0.;weight=ec.object_mass*9.81;first=None
for k in range(45):
 action=actions[min(k,len(actions)-1)];data.ctrl[:]=env.to_joint_targets(action)
 for substep in range(env.substeps):
  mujoco.mj_step(model,data);mujoco.mj_forward(model,data)
  forces={name:np.zeros(3) for name in ['finger','palm','table','other']};torques={name:np.zeros(3) for name in forces};counts={name:0 for name in forces};contacts=[]
  for j,c in enumerate(data.contact):
   if env.object_geom not in (c.geom1,c.geom2):continue
   other=int(c.geom1 if c.geom2==env.object_geom else c.geom2);body=int(model.geom_bodyid[other]);name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,body) or ''
   family='finger' if name[:2] in ['ff','mf','rf','th'] else 'palm' if body==env.palm_id else 'table' if other==env.table_geom else 'other'
   local=np.zeros(6);mujoco.mj_contactForce(model,data,j,local);sign=-1 if c.geom1==env.object_geom else 1
   f=sign*c.frame.reshape(3,3).T@local[:3];torque=sign*c.frame.reshape(3,3).T@local[3:]+np.cross(c.pos-data.xpos[env.object_id],f)
   forces[family]+=f;torques[family]+=torque;counts[family]+=1
   if family=='palm':contacts.append(dict(position_world=c.pos.tolist(),position_palm=(data.xmat[env.palm_id].reshape(3,3).T@(c.pos-data.xpos[env.palm_id])).tolist(),force_world=f.tolist(),normal_world=(sign*c.frame[:3]).tolist(),local_wrench_on_geom2=local.tolist(),penetration_meters=float(c.dist),geom_pair=[int(c.geom1),int(c.geom2)]))
  clearance=min(mujoco.mj_geomDistance(model,data,env.object_geom,g,.2,None) for g in palm_geoms)
  net=sum(forces.values(),np.zeros(3));balance=float(np.linalg.norm(net-data.qfrc_constraint[env.object_dadr:env.object_dadr+3]));maxforceerror=max(maxforceerror,balance)
  row=[float(data.time),clearance,*forces['palm'],*forces['finger'],*torques['palm'],*torques['finger'],forces['palm'][2]/weight,forces['finger'][2]/weight,float(np.linalg.norm(forces['palm'])),*env.object_position()]
  rows.append(row)
  if contacts:details.append(dict(physics_index=len(rows)-1,time=float(data.time),contacts=contacts))
  if first is None and np.linalg.norm(forces['palm'])>1e-6:first=len(rows)-1
 if k<len(actions):
  maxstateerror=max(maxstateerror,float(np.max(np.abs(data.qpos-expected_qpos[k]))),float(np.max(np.abs(data.qvel-expected_qvel[k]))))
columns2=['time','palm_clearance','palm_force_x','palm_force_y','palm_force_z','finger_force_x','finger_force_y','finger_force_z','palm_torque_x','palm_torque_y','palm_torque_z','finger_torque_x','finger_torque_y','finger_torque_z','palm_vertical_weight_ratio','finger_vertical_weight_ratio','palm_force_norm','object_palm_x','object_palm_y','object_palm_z']
a=np.array(rows);c=dict(zip(columns2,a.T));np.savez_compressed(D/'palm_force_trace.npz',samples=a,columns=np.asarray(columns2))
(D/'palm_contacts.json').write_text(json.dumps(details,indent=2)+'\n')
maxforceoriginal=float(np.max(np.abs(c['palm_force_norm'][:len(old)]-oldcols['palm_force_norm'])))
pre=int(np.argmax(oldcols['qualified_angle'][:first]));events={}
for name,i in [('first_palm',first),('just_before_palm',first-1),('max_qualified_before_palm',pre),('peak_palm_probe',int(np.argmax(c['palm_force_norm'][:len(old)]))),('final_probe',len(old)-1),('peak_palm_hold',int(np.argmax(c['palm_force_norm']))),('final_hold',len(a)-1)]:
 events[name]={key:float(values[i]) for key,values in c.items()};events[name]['physics_index']=i
 if i<len(old):events[name].update({key:float(oldcols[key][i]) for key in ['raw_angle','qualified_angle','force_qualified_angle','axis_tilt_radians','finger_contacts','force_bearing_fingers','object_displacement']})
checkpoints=[]
for t in [0,.1,.5,1.,1.3,1.4,1.5,1.6,1.7,1.8,1.816,1.85,1.9,1.91,1.912,2.,2.5,3.,4.5]:
 i=int(np.argmin(np.abs(c['time']-t)));row={key:float(c[key][i]) for key in ['time','palm_clearance','palm_vertical_weight_ratio','finger_vertical_weight_ratio','palm_force_norm','object_palm_x','object_palm_y','object_palm_z']}
 if i<len(old):row.update(qualified_angle=float(oldcols['qualified_angle'][i]),tilt=float(oldcols['axis_tilt_radians'][i]));checkpoints.append(row)
result=dict(input_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [P/'report.json',P/'physics.npz',P/'transitions.npz',P/'audit/report.json',P/'hold_audit/report.json',P/'hold_audit/physics.npz',ROOT/'artifacts/sources/paper.html',ROOT/'artifacts/sources/planner-paper.html',ROOT/'mpc_dex/env.py',ROOT/'assets/allegro/right_hand.xml']},physical_replay=dict(max_recorded_qpos_or_qvel_disagreement=maxstateerror,max_recorded_palm_force_disagreement=maxforceoriginal,max_contact_force_balance_residual=maxforceerror,physics_timestep=.002,commands_replayed=25,subsequent_unchanged_hold_commands=20),weight_newtons=weight,palm_collision_geoms=palm_geoms,events=events,time_samples=checkpoints,palm_force_fraction_probe=float((c['palm_force_norm'][:len(old)]>1e-6).mean()),palm_force_fraction_hold_segment=float((c['palm_force_norm'][len(old):]>1e-6).mean()),mean_vertical_load_ratios_probe={'palm':float(c['palm_vertical_weight_ratio'][:len(old)].mean()),'finger':float(c['finger_vertical_weight_ratio'][:len(old)].mean())},mean_vertical_load_ratios_hold_segment={'palm':float(c['palm_vertical_weight_ratio'][len(old):].mean()),'finger':float(c['finger_vertical_weight_ratio'][len(old):].mean())},no_model_geometry_or_criterion_changes=True)
(D/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));env.close()
