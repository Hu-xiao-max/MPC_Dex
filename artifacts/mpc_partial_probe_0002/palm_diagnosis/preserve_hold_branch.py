from pathlib import Path
from dataclasses import asdict
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from mpc_dex.env import AllegroRotationEnv,EnvConfig
from mpc_dex.contracts import environment_contract,sha256
from scripts.audit_recorded_controls import audit
D=Path(__file__).resolve().parent;P=D.parent;out=D/'prefix17_hold';out.mkdir(exist_ok=False)
original=json.loads((P/'report.json').read_text());ec=EnvConfig(**original['environment_contract']['environment']);ec.initial_grasp=str(ROOT/'assets/initial_grasp.npz')
with np.load(P/'transitions.npz',allow_pickle=False) as ar:source_actions=ar['actions'].copy()
actions=np.concatenate([source_actions[:17],np.repeat(source_actions[16:17],2,axis=0)])
env=AllegroRotationEnv(ec);obs,reset=env.reset(seed=0);initial_qpos=env.data.qpos.copy();initial_qvel=env.data.qvel.copy();initial_ctrl=env.data.ctrl.copy();joint=[env.data.qpos[env.q_ids].copy()];rows=[];infos=[];postq=[];postv=[]
for action in actions:
 nextobs,reward,terminated,truncated,info=env.step(action)
 rows.append((obs.copy(),action.copy(),reward,nextobs.copy(),terminated,truncated));obs=nextobs
 infos.append({k:info[k] for k in ['angle','retained_angle','finger_contacts','axis_tilt_radians','drop','success','palm_support_violation','peak_palm_support_force']});joint.append(env.data.qpos[env.q_ids].copy());postq.append(env.data.qpos.copy());postv.append(env.data.qvel.copy())
 if terminated or truncated:break
if not infos[-1]['success'] or infos[-1]['palm_support_violation'] or infos[-1]['drop']:raise RuntimeError('Expected strict recorded branch success was not reproduced')
metadata=dict(purpose='Diagnostic MPC prefix plus constant-target hold; not a SAC policy or production primitive',source='17 physically executed recorded MPC controls followed by two actual constant-target controls',source_trajectory=str((P/'transitions.npz').resolve()),source_sha256=sha256(P/'transitions.npz'),production_eligible=False,policy_only=False,task='rotation',environment=asdict(ec),environment_contract=environment_contract(ec),frames=len(joint),steps=len(rows),episode=infos[-1],next_actions_policy='No next actions stored or fabricated; adjacent rows contain the actual subsequent executed action; final row is terminal',initialization='Unmodified saved initial grasp under gravity; original object/model geometry and strict no-palm criterion')
arrays={key:np.asarray([r[i] for r in rows]) for i,key in enumerate(['obs','actions','rewards','next_obs','terminated','truncated'])}
np.savez_compressed(out/'transitions.npz',**arrays,qpos=np.asarray(postq),qvel=np.asarray(postv),joint_positions=np.asarray(joint),initial_joint_positions=joint[0],initial_qpos=initial_qpos,initial_qvel=initial_qvel,initial_ctrl=initial_ctrl,metadata=np.asarray(json.dumps(metadata)))
(out/'record.json').write_text(json.dumps(dict(metadata=metadata,info=infos),indent=2)+'\n');env.close()
config={'implementation':'independent-paper-reproduction','environment':asdict(ec)};(out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
stop=audit(out/'transitions.npz',out/'config.json',out/'strict_audit')
config['environment']['terminate_on_success']=False;(out/'hold_config.json').write_text(json.dumps(config,indent=2)+'\n')
held=audit(out/'transitions.npz',out/'hold_config.json',out/'hold_audit',hold_steps=20)
print(json.dumps(dict(strict_audit=stop,hold_audit=held),indent=2))
