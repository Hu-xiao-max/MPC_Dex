"""Snapshot and independently audit a policy in free-object contact simulation.

Both absolute and bootstrap-incremental checkpoints are routed through their
saved action/observation interface. A per-instance observer reads every 2 ms
physics state without changing any MuJoCo state or production source. Full
physics samples and executed control commands are retained, including failures.
This audit is simulation evidence, never hardware or paper-performance evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import mujoco
import numpy as np
import torch

from mpc_dex.bootstrap_env import environment_for_policy
from mpc_dex.cli import read_config
from mpc_dex.contracts import metadata_path, sha256, validate_checkpoint
from mpc_dex.learning import SACAgent


FORCE_CONVENTION='https://mujoco.readthedocs.io/en/latest/computation/#contact'
PHYSICS_COLUMNS=('time','raw_angle','qualified_angle','force_qualified_angle','twist_increment',
    'finger_contacts','force_bearing_fingers','axis_tilt_radians','object_displacement',
    'contactless_seconds','finger_force_x','finger_force_y','finger_force_z',
    'palm_force_norm','table_force_norm','other_force_norm','finger_support_weight_ratio',
    'contact_force_balance_error','object_linear_speed','object_angular_speed',
    'joint_speed_max','actuator_torque_max','object_applied_wrench_norm','object_applied_generalized_force_norm',
    'retained','force_retained','table_contacts','palm_contacts','other_contacts',
    'object_x','object_y','object_z','quaternion_w','quaternion_x','quaternion_y','quaternion_z')


def write_json(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def snapshot_checkpoint(output, checkpoint=None, run=None):
    """Copy one immutable committed generation, retrying only a cleanup race."""
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    for attempt in range(3):
        manifest=json.loads((Path(run)/'latest.json').read_text()) if run else None
        source=(Path(run)/manifest['checkpoint_directory']/'agent.pt') if run else Path(checkpoint)
        destination=output/'checkpoint.pt'
        try:
            shutil.copyfile(source,destination)
            shutil.copyfile(metadata_path(source),metadata_path(destination))
        except FileNotFoundError:
            if not run or attempt==2:raise
            continue
        metadata=json.loads(metadata_path(destination).read_text())
        if sha256(destination)!=metadata['checkpoint_sha256']:
            raise ValueError('Copied checkpoint does not match its metadata hash')
        record=dict(source_checkpoint=str(source.resolve()),checkpoint_sha256=sha256(destination),
                    metadata_sha256=sha256(metadata_path(destination)),run_manifest=manifest,
                    copied_at=datetime.now(timezone.utc).isoformat())
        write_json(output/'snapshot.json',record)
        return destination,record
    raise RuntimeError('No complete checkpoint generation could be snapshotted')


def contact_support(env):
    """World forces on the object, independently classified by contact partner.

    MuJoCo's local contact wrench acts on geom2; frame rows are local axes in
    world coordinates. See FORCE_CONVENTION. This reader never calls forward or
    step and never changes the simulator state.
    """
    model,data=env.model,env.data
    forces={name:np.zeros(3) for name in ('finger','palm','table','other')}
    counts=dict(finger=0,palm=0,table=0,other=0)
    fingers=set();loaded=set()
    for index,contact in enumerate(data.contact):
        if env.object_geom not in (contact.geom1,contact.geom2):continue
        other=int(contact.geom1 if contact.geom2==env.object_geom else contact.geom2)
        body=int(model.geom_bodyid[other]);name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,body) or ''
        finger=name[:2] if name[:2] in {'ff','mf','rf','th'} else None
        family=('finger' if finger else 'table' if other==env.table_geom else
                'palm' if body==env.palm_id else 'other')
        wrench=np.zeros(6);mujoco.mj_contactForce(model,data,index,wrench)
        force=contact.frame.reshape(3,3).T@wrench[:3]
        if contact.geom1==env.object_geom:force=-force
        forces[family]+=force;counts[family]+=1
        if finger:
            fingers.add(finger)
            if wrench[0]>1e-6:loaded.add(finger)
    return forces,counts,len(fingers),len(loaded)


class PhysicsObserver:
    """Observe one environment through a local read-only contact-count hook."""
    def __init__(self,env):
        self.env=env;self.original=env.finger_contacts;self.rows=[]
        self.last_time=float(env.data.time);self.initial_time=self.last_time
        self.previous_quaternion=env.data.xquat[env.object_id].copy()
        matrix=np.empty(9);mujoco.mju_quat2Mat(matrix,self.previous_quaternion)
        self.initial_axis=matrix.reshape(3,3)[:,1].copy()
        self.initial_position=env.data.xpos[env.object_id].copy()
        self.raw_angle=self.qualified_angle=self.force_qualified_angle=0.
        self.contactless_seconds=0.;self.independent_drop=False
        self.gravity_norm=float(np.linalg.norm(env.model.opt.gravity))
        self.up=-env.model.opt.gravity/max(self.gravity_norm,1e-12)
        self.weight=float(env.model.body_mass[env.object_id])*self.gravity_norm
        env.finger_contacts=self.contact_count

    def close(self):
        self.env.finger_contacts=self.original

    def contact_count(self,data=None):
        result=self.original(data)
        if data is None and self.env.data.time>self.last_time+1e-12:self.capture()
        return result

    def capture(self):
        env=self.env;data=env.data
        elapsed=float(data.time)-self.last_time;self.last_time=float(data.time)
        quaternion=data.xquat[env.object_id].copy()
        inverse=np.empty(4);delta=np.empty(4)
        mujoco.mju_negQuat(inverse,self.previous_quaternion);mujoco.mju_mulQuat(delta,inverse,quaternion)
        if delta[0]<0:delta=-delta
        increment=2*math.atan2(float(delta[2]),float(delta[0]))
        self.previous_quaternion=quaternion
        matrix=np.empty(9);mujoco.mju_quat2Mat(matrix,quaternion)
        tilt=math.acos(float(np.clip(matrix.reshape(3,3)[:,1]@self.initial_axis,-1,1)))
        forces,counts,fingers,loaded=contact_support(env)
        retained=fingers>=2 and tilt<=math.radians(env.config.success_axis_tolerance_degrees)
        force_retained=loaded>=2 and retained
        self.raw_angle+=increment
        self.qualified_angle+=increment if increment<0 or retained else 0.
        self.force_qualified_angle+=increment if increment<0 or force_retained else 0.
        self.contactless_seconds=self.contactless_seconds+elapsed if fingers==0 else 0.
        position=data.xpos[env.object_id].copy();displacement=float(np.linalg.norm(position-self.initial_position))
        if counts['table'] or displacement>.18 or self.contactless_seconds>=env.config.contact_loss_seconds-1e-10:
            self.independent_drop=True
        da=env.object_dadr
        net=sum(forces.values(),np.zeros(3))
        row=(float(data.time)-self.initial_time,self.raw_angle,self.qualified_angle,self.force_qualified_angle,increment,
            fingers,loaded,tilt,displacement,self.contactless_seconds,*forces['finger'],
            *(float(np.linalg.norm(forces[x])) for x in ('palm','table','other')),
            float(forces['finger']@self.up/max(self.weight,1e-12)),
            float(np.linalg.norm(net-data.qfrc_constraint[da:da+3])),
            float(np.linalg.norm(data.qvel[da:da+3])),float(np.linalg.norm(data.qvel[da+3:da+6])),
            float(np.max(np.abs(data.qvel[env.v_ids]))),float(np.max(np.abs(data.actuator_force))),
            float(np.linalg.norm(data.xfrc_applied[env.object_id])),float(np.linalg.norm(data.qfrc_applied[da:da+6])),
            retained,force_retained,counts['table'],counts['palm'],counts['other'],*position,*quaternion)
        self.rows.append(row)


def audit_episode(env,agent,seed,step_limit,deterministic=True):
    if step_limit<1:raise ValueError('Audit step limit must be positive')
    obs,info=env.reset(seed=seed)
    initial_hash=hashlib.sha256(np.r_[env.data.qpos,env.data.qvel,env.data.ctrl].tobytes()).hexdigest()
    observer=PhysicsObserver(env);commands=[];reward_total=paper_total=0.;terminated=truncated=False
    max_raw_error=max_qualified_error=0.
    try:
        for step in range(step_limit):
            previous=env.data.ctrl.copy();measured=env.data.qpos[env.q_ids].copy()
            action=agent.act(obs,deterministic=deterministic)
            obs,reward,terminated,truncated,info=env.step(action)
            target=env.data.ctrl.copy();reward_total+=reward;paper_total+=info.get('paper_reward',reward)
            expected=(step+1)*env.substeps
            if len(observer.rows)!=expected:
                raise RuntimeError(f'Physics instrumentation missed samples: {len(observer.rows)} != {expected}')
            max_raw_error=max(max_raw_error,abs(observer.raw_angle-info['angle']))
            max_qualified_error=max(max_qualified_error,abs(observer.qualified_angle-info['retained_angle']))
            commands.append(dict(step=step+1,policy_action=action.tolist(),previous_target=previous.tolist(),
                measured_joint_position=measured.tolist(),executed_target=target.tolist(),
                target_delta_max=float(np.max(np.abs(target-previous))),
                target_delta_rms=float(np.sqrt(np.mean((target-previous)**2))),
                command_rate_max=float(np.max(np.abs(target-previous))/env.config.control_seconds),
                target_error_max=float(np.max(np.abs(target-measured))),
                reward=float(reward),paper_reward=float(info.get('paper_reward',reward)),
                raw_angle=observer.raw_angle,qualified_angle=observer.qualified_angle,
                force_qualified_angle=observer.force_qualified_angle))
            if terminated or truncated:break
    finally:observer.close()
    physics=np.asarray(observer.rows,dtype=np.float64)
    col={name:physics[:,i] for i,name in enumerate(PHYSICS_COLUMNS)}
    retained=col['retained'].astype(bool);force_retained=col['force_retained'].astype(bool)
    def first_time(mask):
        indices=np.flatnonzero(mask)
        return float(col['time'][indices[0]]) if len(indices) else None
    no_external_support=all(float(np.max(col[name]))<=1e-6 for name in
        ('palm_force_norm','table_force_norm','other_force_norm','object_applied_wrench_norm','object_applied_generalized_force_norm'))
    full_turn=observer.qualified_angle>=2*np.pi and observer.raw_angle>=2*np.pi
    independent_success=bool(env.config.task=='rotation' and full_turn and retained[-1] and not observer.independent_drop)
    finger_supported_success=bool(independent_success and observer.force_qualified_angle>=2*np.pi
                                  and force_retained[-1] and no_external_support and env.model.neq==0)
    result=dict(seed=seed,initial_physics_sha256=initial_hash,steps=len(commands),physics_steps=len(physics),
        simulated_seconds=float(physics[-1,0]),environment_success=bool(info['success']),environment_drop=bool(info['drop']),
        independent_rotation_success=independent_success,finger_supported_rotation_success=finger_supported_success,
        independent_drop=observer.independent_drop,terminated=bool(terminated),truncated=bool(truncated),
        audit_limit_reached=not(terminated or truncated),return_value=reward_total,paper_return=paper_total,
        final_raw_angle=observer.raw_angle,final_qualified_angle=observer.qualified_angle,
        final_force_qualified_angle=observer.force_qualified_angle,
        max_raw_angle=float(col['raw_angle'].max()),max_qualified_angle=float(col['qualified_angle'].max()),
        max_force_qualified_angle=float(col['force_qualified_angle'].max()),
        max_qualified_angle_time_seconds=float(col['time'][col['qualified_angle'].argmax()]),
        first_zero_contact_seconds=first_time(col['finger_contacts']==0),
        first_fewer_than_two_contacts_seconds=first_time(col['finger_contacts']<2),
        first_axis_violation_seconds=first_time(col['axis_tilt_radians']>math.radians(env.config.success_axis_tolerance_degrees)),
        final_finger_contacts=int(col['finger_contacts'][-1]),final_force_bearing_fingers=int(col['force_bearing_fingers'][-1]),
        final_axis_tilt_radians=float(col['axis_tilt_radians'][-1]),
        retained_physics_fraction=float(retained.mean()),force_retained_physics_fraction=float(force_retained.mean()),
        zero_contact_physics_fraction=float((col['finger_contacts']==0).mean()),
        zero_contact_signed_twist=float(col['twist_increment'][col['finger_contacts']==0].sum()),
        no_external_support=no_external_support,
        mean_finger_support_weight_ratio=float(col['finger_support_weight_ratio'].mean()),
        max_contact_force_balance_error=float(col['contact_force_balance_error'].max()),
        max_raw_angle_disagreement=max_raw_error,max_qualified_angle_disagreement=max_qualified_error,
        max_joint_speed=float(col['joint_speed_max'].max()),max_object_angular_speed=float(col['object_angular_speed'].max()),
        max_object_displacement=float(col['object_displacement'].max()),max_actuator_torque=float(col['actuator_torque_max'].max()),
        max_command_rate=max(x['command_rate_max'] for x in commands),
        max_target_delta=max(x['target_delta_max'] for x in commands),max_target_error=max(x['target_error_max'] for x in commands),
        peak_support_forces={name:float(col[name+'_force_norm'].max()) for name in ('palm','table','other')})
    if max(max_raw_error,max_qualified_error)>1e-7:
        raise RuntimeError('Independent rotation measurements disagree with the environment')
    return result,physics,commands


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint');source.add_argument('--run',help='Snapshot the latest committed bootstrap generation')
    parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--episodes',type=int,default=5);parser.add_argument('--max-steps',type=int)
    parser.add_argument('--seed',type=int,default=1000000);parser.add_argument('--device',default='cpu')
    parser.add_argument('--stochastic',action='store_true');parser.add_argument('--torch-threads',type=int,default=1)
    args=parser.parse_args(argv)
    if args.episodes<1 or args.torch_threads<1 or (args.max_steps is not None and args.max_steps<1):
        parser.error('Episode, thread and optional step limits must be positive')
    torch.set_num_threads(args.torch_threads);torch.manual_seed(args.seed)
    output=Path(args.output);checkpoint,snapshot=snapshot_checkpoint(output,args.checkpoint,args.run)
    shutil.copyfile(__file__,output/'audit_policy.py')
    config,ec=read_config(args.config);write_json(output/'config.json',config)
    metadata=validate_checkpoint(checkpoint,ec)
    interface=metadata.get('policy_interface',{'kind':'absolute_joint_target'})
    env=environment_for_policy(ec,interface);agent=SACAgent.load(checkpoint,args.device)
    limit=min(ec.horizon_steps,args.max_steps or ec.horizon_steps)
    results=[];start=time.monotonic()
    report=dict(status='running',implementation='independent-paper-reproduction',snapshot=snapshot,
        policy_interface=interface,deterministic=not args.stochastic,config_sha256=sha256(args.config),
        inference_device=str(agent.device),torch_version=str(torch.__version__),mujoco_version=mujoco.__version__,
        model_equality_constraints=int(env.model.neq),
        audit_source_sha256=sha256(__file__),force_convention_source=FORCE_CONVENTION,
        physics_columns=PHYSICS_COLUMNS,physics_timestep=ec.timestep,control_seconds=ec.control_seconds,
        requested_episodes=args.episodes,audit_step_limit=limit,results=results,
        hardware_verified=False,paper_performance_verified=False,
        independent_success_definition='Full signed axial turn qualified at every physics step by at least two geometric finger contacts and allowed axis tilt; all reverse increments count; final retention; no independent workspace/table/contact-loss drop.',
        finger_supported_success_definition='Independent rotation success plus full force-bearing qualified turn, final two force-bearing fingers, no palm/table/other force or applied object force, and no model equality constraints.',
        instrumentation='Per-environment contact-count observer; read-only MuJoCo getters; no global monkeypatch or extra physics/forward calls.')
    write_json(output/'report.json',report)
    try:
        for episode in range(args.episodes):
            result,physics,commands=audit_episode(env,agent,args.seed+episode,limit,not args.stochastic)
            result['episode']=episode;results.append(result)
            np.savez_compressed(output/f'physics_{episode:03d}.npz',samples=physics,columns=np.asarray(PHYSICS_COLUMNS))
            write_json(output/f'commands_{episode:03d}.json',commands)
            report.update(completed_episodes=len(results),wall_seconds=time.monotonic()-start,
                independent_successes=sum(x['independent_rotation_success'] for x in results),
                finger_supported_successes=sum(x['finger_supported_rotation_success'] for x in results),
                max_qualified_angle=max(x['max_qualified_angle'] for x in results),
                unique_initial_physics_states=len({x['initial_physics_sha256'] for x in results}))
            write_json(output/'report.json',report)
            print(json.dumps(result,allow_nan=False),flush=True)
        report['status']='complete';write_json(output/'report.json',report)
        return 0
    except BaseException as error:
        report.update(status='failed',error=str(error));write_json(output/'report.json',report);raise
    finally:env.close()


if __name__=='__main__':raise SystemExit(main())
