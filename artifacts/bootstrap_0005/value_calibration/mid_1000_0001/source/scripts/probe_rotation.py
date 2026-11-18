"""Measure unsupported rotation caused only by 10 Hz joint targets.

This geometric probe is an independent feasibility experiment, not the paper
controller or a claimed full-turn policy. It never actuates or constrains the
object. It solves fingertip IK on a separate kinematic scratch state and
executes those targets in the unmodified contact-dynamics environment.
"""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from scipy.optimize import least_squares
from mpc_dex.env import AllegroRotationEnv,EnvConfig


def run_probe(out: Path, target: float, seconds: float, preload: float, follow: float, preload_seconds: float=0.):
    env=AllegroRotationEnv(EnvConfig(initial_grasp='assets/initial_grasp.npz',horizon_steps=300))
    obs,_=env.reset(seed=0)
    kd=mujoco.MjData(env.model)
    mujoco.mj_copyData(kd,env.model,env.data)
    sites=np.array([env.model.site(f+'_tip_site').id for f in ('ff','mf','rf','th')])
    origin=env.data.xpos[env.palm_id].copy(); R=env.data.xmat[env.palm_id].reshape(3,3).copy()
    center=env.object_position().copy(); baseline=env.data.ctrl.copy()
    kd.qpos[env.q_ids]=baseline;mujoco.mj_forward(env.model,kd)
    offsets=(kd.site_xpos[sites]-origin)@R-center
    observations=[];next_observations=[];rewards=[];positions=[];actions=[];angles=[];contacts=[];tilts=[];qpositions=[];qvels=[];errors=[]
    last=baseline.copy();min_contact=4;max_tilt=0.;bad_support=False
    nt=round(seconds/env.config.control_seconds)
    start=time.monotonic()
    for t in range(nt+30):
        theta=target*min(1.,(t+1)/nt)
        c,s=np.cos(theta),np.sin(theta)
        rotation=np.array([[c,0,s],[0,1,0],[-s,0,c]])
        v=offsets.copy();v[:,[0,2]]*=1+(preload-1)*min(1.,(t+1)*env.config.control_seconds/max(.1,preload_seconds))
        desired=center+(env.object_position()-center)*follow+v@rotation.T
        def residual(q):
            kd.qpos[env.q_ids]=q;mujoco.mj_kinematics(env.model,kd)
            tip=(kd.site_xpos[sites]-origin)@R
            return np.r_[100*(tip-desired).ravel(),.01*(q-last)]
        fit=least_squares(residual,np.clip(last,env.low+1e-5,env.high-1e-5),bounds=(env.low+1e-6,env.high-1e-6),max_nfev=50,ftol=1e-6,xtol=1e-6,gtol=1e-6)
        last=fit.x; action=env.to_action(last)
        next_obs,reward,done,trunc,info=env.step(action)
        observations.append(obs);next_observations.append(next_obs);rewards.append(reward);obs=next_obs
        min_contact=min(min_contact,info['finger_contacts']);max_tilt=max(max_tilt,info['axis_tilt_radians'])
        for contact in env.data.contact:
            if env.object_geom in (contact.geom1,contact.geom2):
                other=contact.geom2 if contact.geom1==env.object_geom else contact.geom1
                if other==env.table_geom or env.model.geom_bodyid[other]==env.palm_id:bad_support=True
        angles.append(info['angle']);positions.append(info['object_position']);actions.append(action);contacts.append(info['finger_contacts']);tilts.append(info['axis_tilt_radians']);qpositions.append(env.data.qpos.copy());qvels.append(env.data.qvel.copy());errors.append(np.linalg.norm(fit.fun[:12])/100)
        if done or trunc:break
    report=dict(target=target,seconds=seconds,preload=preload,follow=follow,preload_seconds=preload_seconds,final_angle=angles[-1],max_angle=max(angles),minimum_contacts=min_contact,maximum_axis_tilt=max_tilt,final_contacts=contacts[-1],final_tilt=tilts[-1],dropped=info['drop'],forbidden_support=bad_support,elapsed_steps=len(angles),max_ik_error=max(errors),wall_seconds=time.monotonic()-start)
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez(Path(str(out)+'.npz'),observations=np.array(observations),next_observations=np.array(next_observations),rewards=np.array(rewards),actions=np.array(actions),angle=np.array(angles),position=np.array(positions),contacts=np.array(contacts),tilt=np.array(tilts),qpos=np.array(qpositions),qvel=np.array(qvels),ik_error=np.array(errors),report_json=np.array(json.dumps(report)))
    Path(str(out)+'.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    return report


def run_gait(out: Path, order: str, lift: float, gait_seconds: float=1., cycles: int=2):
    env=AllegroRotationEnv(EnvConfig(initial_grasp='assets/initial_grasp.npz',horizon_steps=1000))
    obs,_=env.reset(seed=0)
    kd=mujoco.MjData(env.model);mujoco.mj_copyData(kd,env.model,env.data)
    sites=np.array([env.model.site(f+'_tip_site').id for f in ('ff','mf','rf','th')])
    origin=env.data.xpos[env.palm_id].copy();R=env.data.xmat[env.palm_id].reshape(3,3).copy()
    center=env.object_position().copy();last=env.data.ctrl.copy()
    kd.qpos[env.q_ids]=last;mujoco.mj_forward(env.model,kd)
    offsets=(kd.site_xpos[sites]-origin)@R-center;offsets[:,[0,2]]*=.97
    phases=[];theta=np.zeros(4)
    for cycle in range(cycles):
        end=np.ones(4)*1.2
        for s in np.linspace(0,1,41)[1:]:phases.append((theta*(1-s)+end*s,np.ones(4),'advance'))
        theta=end.copy()
        for finger in map(int,order):
            end=theta.copy();end[finger]-=np.pi/3
            for s in np.linspace(0,1,round(gait_seconds*10)+1)[1:]:
                radius=np.ones(4);radius[finger]+=lift*np.sin(np.pi*s)/np.linalg.norm(offsets[finger,[0,2]])
                phases.append((theta*(1-s)+end*s,radius,'reset_'+str(finger)))
            theta=end.copy()
        for _ in range(10):phases.append((theta.copy(),np.ones(4),'hold'))
    for _ in range(30):phases.append((theta.copy(),np.ones(4),'hold'))
    rows={k:[] for k in ('observations','next_observations','rewards','actions','angle','position','contacts','tilt','qpos','qvel','phase','ik_error')}
    forbidden=False
    for theta,radius,phase in phases:
        desired=offsets.copy()
        for finger in range(4):
            c,s=np.cos(theta[finger]),np.sin(theta[finger]);rot=np.array([[c,0,s],[0,1,0],[-s,0,c]])
            v=offsets[finger].copy();v[[0,2]]*=radius[finger]
            desired[finger]=center+.8*(env.object_position()-center)+rot@v
        def residual(q):
            kd.qpos[env.q_ids]=q;mujoco.mj_kinematics(env.model,kd)
            return np.r_[100*((kd.site_xpos[sites]-origin)@R-desired).ravel(),.01*(q-last)]
        fit=least_squares(residual,np.clip(last,env.low+1e-5,env.high-1e-5),bounds=(env.low+1e-6,env.high-1e-6),max_nfev=50,ftol=1e-6,xtol=1e-6,gtol=1e-6)
        last=fit.x;action=env.to_action(last)
        next_obs,reward,done,trunc,info=env.step(action)
        for contact in env.data.contact:
            if env.object_geom in (contact.geom1,contact.geom2):
                other=contact.geom2 if contact.geom1==env.object_geom else contact.geom1
                forbidden|=bool(other==env.table_geom or env.model.geom_bodyid[other]==env.palm_id)
        row=dict(observations=obs,next_observations=next_obs,rewards=reward,actions=action,angle=info['angle'],position=info['object_position'],contacts=info['finger_contacts'],tilt=info['axis_tilt_radians'],qpos=env.data.qpos.copy(),qvel=env.data.qvel.copy(),phase=phase,ik_error=np.linalg.norm(fit.fun[:12])/100)
        for key,value in row.items():rows[key].append(value)
        obs=next_obs
        if done or trunc:break
    report=dict(order=order,lift=lift,gait_seconds=gait_seconds,cycles=cycles,final_angle=rows['angle'][-1],max_angle=max(rows['angle']),minimum_contacts=min(rows['contacts']),maximum_tilt=max(rows['tilt']),final_contacts=rows['contacts'][-1],final_tilt=rows['tilt'][-1],dropped=info['drop'],forbidden_support=forbidden,elapsed_steps=len(rows['angle']),total_steps=len(phases),phase_at_end=rows['phase'][-1])
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez(Path(str(out)+'.npz'),**{k:np.array(v) for k,v in rows.items()},report_json=np.array(json.dumps(report)))
    Path(str(out)+'.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    return report


def run_transfer_alternative(out: Path, alternative: int):
    """Three bounded contact-transfer experiments; physical state is never reset."""
    env=AllegroRotationEnv(EnvConfig(initial_grasp='assets/initial_grasp.npz',horizon_steps=1000))
    obs,_=env.reset(seed=0)
    kd=mujoco.MjData(env.model);mujoco.mj_copyData(kd,env.model,env.data)
    sites=np.array([env.model.site(f+'_tip_site').id for f in ('ff','mf','rf','th')])
    origin=env.data.xpos[env.palm_id].copy();R=env.data.xmat[env.palm_id].reshape(3,3).copy()
    center=env.object_position().copy();last=env.data.ctrl.copy()
    kd.qpos[env.q_ids]=last;mujoco.mj_forward(env.model,kd)
    offsets=(kd.site_xpos[sites]-origin)@R-center
    phases=[];theta=np.zeros(4)
    def stage(end,steps,label,lift_finger=-1,lift=0.,posture=None):
        nonlocal theta
        for s in np.linspace(0,1,steps+1)[1:]:
            radius=np.ones(4)
            if lift_finger>=0:radius[lift_finger]+=lift*np.sin(np.pi*s)/np.linalg.norm(offsets[lift_finger,[0,2]])
            phases.append((theta*(1-s)+np.asarray(end)*s,radius,label,posture(s) if posture else None))
        theta=np.asarray(end).copy()
    if alternative==1:
        stage(np.ones(4)*.8,80,'initial_roll')
        stage(theta,30,'thumb_posture_reset',posture=lambda s:1.396*(1-s)+.95*s)
        stage(np.ones(4)*1.8,100,'continue_roll',posture=lambda s:.95)
    elif alternative==2:
        stage(np.ones(4)*.8,80,'initial_roll')
        end=theta.copy();end[0]=2.8;stage(end,60,'index_opposition_transfer',lift_finger=0,lift=.004)
        stage(theta,10,'stabilize')
        end=theta.copy();end[3]-=np.pi/3;stage(end,30,'thumb_reset',lift_finger=3,lift=.003)
        end=theta+np.pi/3;stage(end,80,'continue_roll')
    elif alternative==3:
        stage(np.ones(4)*1.4,80,'initial_roll')
        end=theta.copy();end[3]-=np.pi/3;stage(end,50,'late_thumb_reset')
        for finger in (0,1,2):
            end=theta.copy();end[finger]-=np.pi/3;stage(end,30,'finger_reset_'+str(finger))
        stage(theta+np.pi/3,80,'continue_roll')
    else:raise ValueError('alternative must be 1, 2, or 3')
    stage(theta,30,'final_hold',posture=(lambda s:.95) if alternative==1 else None)
    rows={k:[] for k in ('observations','next_observations','rewards','actions','angle','position','contacts','tilt','qpos','qvel','phase','ik_error')}
    forbidden=False;phase_reports={}
    for theta,radius,phase,posture in phases:
        desired=offsets.copy()
        for finger in range(4):
            c,s=np.cos(theta[finger]),np.sin(theta[finger]);rot=np.array([[c,0,s],[0,1,0],[-s,0,c]])
            v=offsets[finger].copy();v[[0,2]]*=radius[finger]
            desired[finger]=center+(env.object_position()-center)*(.8 if alternative==3 else 0.)+rot@v
        def residual(q):
            kd.qpos[env.q_ids]=q;mujoco.mj_kinematics(env.model,kd)
            terms=np.r_[100*((kd.site_xpos[sites]-origin)@R-desired).ravel(),.01*(q-last)]
            if posture is not None:terms=np.r_[terms,4.*(q[12]-posture)]
            return terms
        fit=least_squares(residual,np.clip(last,env.low+1e-5,env.high-1e-5),bounds=(env.low+1e-6,env.high-1e-6),max_nfev=75,ftol=1e-6,xtol=1e-6,gtol=1e-6)
        last=fit.x;action=env.to_action(last)
        next_obs,reward,done,trunc,info=env.step(action)
        for contact in env.data.contact:
            if env.object_geom in (contact.geom1,contact.geom2):
                other=contact.geom2 if contact.geom1==env.object_geom else contact.geom1
                forbidden|=bool(other==env.table_geom or env.model.geom_bodyid[other]==env.palm_id)
        row=dict(observations=obs,next_observations=next_obs,rewards=reward,actions=action,angle=info['angle'],position=info['object_position'],contacts=info['finger_contacts'],tilt=info['axis_tilt_radians'],qpos=env.data.qpos.copy(),qvel=env.data.qvel.copy(),phase=phase,ik_error=np.linalg.norm(fit.fun[:12])/100)
        for key,value in row.items():rows[key].append(value)
        phase_reports[phase]=dict(angle=info['angle'],contacts=info['finger_contacts'],tilt=info['axis_tilt_radians'],ik_error=row['ik_error'])
        obs=next_obs
        if done or trunc:break
    report=dict(alternative=alternative,final_angle=rows['angle'][-1],max_angle=max(rows['angle']),minimum_contacts=min(rows['contacts']),maximum_tilt=max(rows['tilt']),final_contacts=rows['contacts'][-1],final_tilt=rows['tilt'][-1],dropped=info['drop'],forbidden_support=forbidden,elapsed_steps=len(rows['angle']),total_steps=len(phases),phase_reports=phase_reports)
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez(Path(str(out)+'.npz'),**{k:np.array(v) for k,v in rows.items()},report_json=np.array(json.dumps(report)))
    Path(str(out)+'.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    return report


def run_fixed_thumb(out: Path, configuration: int):
    """Attempt three rolling cycles against a thumb held at fixed joint targets."""
    choices={1:dict(stroke=.6,lift=.006,squeeze=0.,drivers=(0,1,2)),2:dict(stroke=.35,lift=.004,squeeze=.0015,drivers=(0,1,2)),3:dict(stroke=.6,lift=.004,squeeze=.0015,drivers=(0,2))}
    cfg=choices[configuration]
    env=AllegroRotationEnv(EnvConfig(initial_grasp='assets/initial_grasp.npz',horizon_steps=1500))
    obs,_=env.reset(seed=0)
    kd=mujoco.MjData(env.model);mujoco.mj_copyData(kd,env.model,env.data)
    sites=np.array([env.model.site(f+'_tip_site').id for f in ('ff','mf','rf','th')])
    origin=env.data.xpos[env.palm_id].copy();R=env.data.xmat[env.palm_id].reshape(3,3).copy()
    center=env.object_position().copy();last=env.data.ctrl.copy();baseline=last.copy()
    kd.qpos[env.q_ids]=last;mujoco.mj_forward(env.model,kd)
    initial_tips=(kd.site_xpos[sites]-origin)@R;offsets=initial_tips-center
    fixed_thumb=baseline[12:].copy()
    if cfg['squeeze']:
        inward=center-initial_tips[3];inward[1]=0;inward/=np.linalg.norm(inward)
        desired_thumb=initial_tips[3]+cfg['squeeze']*inward
        def thumb_residual(q):
            kd.qpos[env.q_ids[12:]]=q;mujoco.mj_kinematics(env.model,kd)
            return np.r_[100*((kd.site_xpos[sites[3]]-origin)@R-desired_thumb),.02*(q-baseline[12:])]
        fit=least_squares(thumb_residual,fixed_thumb,bounds=(env.low[12:]+1e-6,env.high[12:]-1e-6),max_nfev=100)
        fixed_thumb=fit.x
    phases=[];theta=np.zeros(4)
    def stage(end,steps,label,lift_finger=-1):
        nonlocal theta
        for s in np.linspace(0,1,steps+1)[1:]:
            radius=np.ones(4)
            if lift_finger>=0:radius[lift_finger]+=cfg['lift']*np.sin(np.pi*s)/np.linalg.norm(offsets[lift_finger,[0,2]])
            phases.append((theta*(1-s)+np.asarray(end)*s,radius,label))
        theta=np.asarray(end).copy()
    stage(theta,20,'initial_squeeze')
    for cycle in range(3):
        end=theta.copy();end[list(cfg['drivers'])]=cfg['stroke']
        stage(end,60,f'cycle_{cycle+1}_drive')
        for finger in cfg['drivers']:
            end=theta.copy();end[finger]=0.
            stage(end,20,f'cycle_{cycle+1}_reset_{finger}',finger)
        stage(theta,10,f'cycle_{cycle+1}_hold')
    stage(theta,30,'final_hold')
    rows={k:[] for k in ('observations','next_observations','rewards','actions','angle','position','contacts','tilt','qpos','qvel','phase','ik_error')}
    forbidden=False;phase_reports={};max_target_step=0.
    for theta,radius,phase in phases:
        desired=initial_tips.copy()
        for finger in cfg['drivers']:
            c,s=np.cos(theta[finger]),np.sin(theta[finger]);rot=np.array([[c,0,s],[0,1,0],[-s,0,c]])
            v=offsets[finger].copy();v[[0,2]]*=radius[finger]
            desired[finger]=center+.8*(env.object_position()-center)+rot@v
        def residual(q):
            kd.qpos[env.q_ids]=q;mujoco.mj_kinematics(env.model,kd)
            return np.r_[100*((kd.site_xpos[sites[:3]]-origin)@R-desired[:3]).ravel(),.01*(q-last)]
        # Keep thumb targets constant; optimize only the other twelve joints.
        def fingers_residual(q):return residual(np.r_[q,fixed_thumb])
        fit=least_squares(fingers_residual,np.clip(last[:12],env.low[:12]+1e-5,env.high[:12]-1e-5),bounds=(env.low[:12]+1e-6,env.high[:12]-1e-6),max_nfev=75,ftol=1e-6,xtol=1e-6,gtol=1e-6)
        proposed=np.r_[fit.x,fixed_thumb]
        commanded=np.clip(proposed,last-.02,last+.02);commanded=np.clip(commanded,env.low,env.high)
        max_target_step=max(max_target_step,float(np.max(np.abs(commanded-last))));last=commanded
        action=env.to_action(commanded)
        next_obs,reward,done,trunc,info=env.step(action)
        for contact in env.data.contact:
            if env.object_geom in (contact.geom1,contact.geom2):
                other=contact.geom2 if contact.geom1==env.object_geom else contact.geom1
                forbidden|=bool(other==env.table_geom or env.model.geom_bodyid[other]==env.palm_id)
        row=dict(observations=obs,next_observations=next_obs,rewards=reward,actions=action,angle=info['angle'],position=info['object_position'],contacts=info['finger_contacts'],tilt=info['axis_tilt_radians'],qpos=env.data.qpos.copy(),qvel=env.data.qvel.copy(),phase=phase,ik_error=np.linalg.norm(fit.fun[:9])/100)
        for key,value in row.items():rows[key].append(value)
        phase_reports[phase]=dict(angle=info['angle'],contacts=info['finger_contacts'],tilt=info['axis_tilt_radians'],ik_error=row['ik_error'])
        obs=next_obs
        if done or trunc:break
    report=dict(configuration=configuration,parameters=cfg,final_angle=rows['angle'][-1],max_angle=max(rows['angle']),minimum_contacts=min(rows['contacts']),maximum_tilt=max(rows['tilt']),final_contacts=rows['contacts'][-1],final_tilt=rows['tilt'][-1],dropped=info['drop'],forbidden_support=forbidden,elapsed_steps=len(rows['angle']),total_steps=len(phases),max_joint_target_step=max_target_step,phase_reports=phase_reports)
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez(Path(str(out)+'.npz'),**{k:np.array(v) for k,v in rows.items()},report_json=np.array(json.dumps(report)))
    Path(str(out)+'.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    return report


def verify_saved(path: Path):
    """Replay a saved 10 Hz trajectory and audit constraints every physics step."""
    saved=np.load(path,allow_pickle=False)
    env=AllegroRotationEnv(EnvConfig(initial_grasp='assets/initial_grasp.npz',control_seconds=.002,horizon_steps=100000))
    env.reset(seed=0)
    minimum_contacts=4;maximum_tilt=0.;forbidden=False;terminated=False
    samples=[]
    for action in saved['actions']:
        for _ in range(50):
            _,_,done,trunc,info=env.step(action)
            minimum_contacts=min(minimum_contacts,info['finger_contacts']);maximum_tilt=max(maximum_tilt,info['axis_tilt_radians'])
            for contact in env.data.contact:
                if env.object_geom in (contact.geom1,contact.geom2):
                    other=contact.geom2 if contact.geom1==env.object_geom else contact.geom1
                    forbidden|=bool(other==env.table_geom or env.model.geom_bodyid[other]==env.palm_id)
            if done or trunc:terminated=True;break
        samples.append(info['angle'])
        if terminated:break
    report=dict(source=str(path),sample_period_seconds=.002,target_period_seconds=.1,final_angle=info['angle'],minimum_contacts_every_physics_step=minimum_contacts,maximum_tilt_every_physics_step=maximum_tilt,forbidden_support=forbidden,dropped=info['drop'],terminated=terminated,applied_object_force_max=float(np.max(np.abs(env.data.xfrc_applied))),applied_generalized_force_max=float(np.max(np.abs(env.data.qfrc_applied))),replay_angle_error=float(np.max(np.abs(np.array(samples)-saved['angle'][:len(samples)]))),completed_targets=len(samples),total_targets=len(saved['actions']))
    output=path.with_name(path.stem+'_replay.json');output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    return report


def contact_wrench_audit(output: Path):
    """Necessary static gravity support check, not a proof of force closure.

    The friction polygon encloses the circular Coulomb cone; torsional friction
    is independently allowed at its maximum, and joint force limits are omitted.
    Thus this relaxation makes support easier than the physical model. Failure
    rules out gravity balance using the selected instantaneous contacts, while
    success alone does not demonstrate an executable contact transfer.
    """
    from scipy.optimize import linprog
    env=AllegroRotationEnv(EnvConfig(initial_grasp='assets/initial_grasp.npz'))
    env.reset();R=env.data.xmat[env.palm_id].reshape(3,3).copy()
    cases=[(None,0),('orbit_0.8_s8_p1_f0',109),('orbit_1.4_s4_p0.97_f0.8',69),('orbit_1.8_s4_p0.97_f0.8',69)]
    reports=[]
    for filename,index in cases:
        if filename:
            saved=np.load(output.parent/(filename+'.npz'),allow_pickle=False)
            env.data.qpos[:]=saved['qpos'][index];env.data.qvel[:]=saved['qvel'][index]
            env.data.ctrl[:]=env.to_joint_targets(saved['actions'][index]);mujoco.mj_forward(env.model,env.data)
        for include_thumb in (True,False):
            columns=[];normals={}
            for contact in env.data.contact:
                if env.object_geom not in (contact.geom1,contact.geom2):continue
                other=contact.geom2 if contact.geom1==env.object_geom else contact.geom1
                name=env.model.body(env.model.geom_bodyid[other]).name
                if name.startswith('th') and not include_thumb:continue
                basis=contact.frame.reshape(3,3)@R
                normal=basis[0]*(1 if contact.geom2==env.object_geom else -1)
                # Circumscribed polygon encloses the real circular friction cone.
                mu=contact.friction[0]/np.cos(np.pi/16)
                lever=(contact.pos-env.data.xpos[env.object_id])@R
                normals[name]=normal.tolist()
                for angle in np.linspace(0,2*np.pi,16,endpoint=False):
                    force=normal+mu*(np.cos(angle)*basis[1]+np.sin(angle)*basis[2])
                    for sign in (-1,1):
                        torque=np.cross(lever,force)+sign*contact.friction[2]*normal
                        columns.append(np.r_[force,torque/.0225])
            A=np.array(columns).T
            target=np.r_[-env.config.object_mass*(env.model.opt.gravity@R),np.zeros(3)]
            fit=linprog(np.ones(len(columns)),A_eq=A,b_eq=target,bounds=(0,None),method='highs')
            report=dict(state=filename or 'initial',include_thumb=include_thumb,gravity_wrench_feasible=fit.success,total_normal_force=float(sum(fit.x)) if fit.success else None,normals=normals)
            reports.append(report)
            print(json.dumps({k:v for k,v in report.items() if k!='normals'}),flush=True)
    output.write_text(json.dumps(dict(method=contact_wrench_audit.__doc__,reports=reports),indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='artifacts/rotation_probe');p.add_argument('--target',type=float,nargs='+',default=[.2,.4,.6,.8,1.]);p.add_argument('--seconds',type=float,default=4.);p.add_argument('--preload',type=float,default=1.);p.add_argument('--follow',type=float,default=0.)
    p.add_argument('--fixed-thumb',type=int,choices=(1,2,3));p.add_argument('--contact-audit',action='store_true');p.add_argument('--transfer-alternative',type=int,choices=(1,2,3));p.add_argument('--preload-seconds',type=float,default=0.);p.add_argument('--verify',type=Path);p.add_argument('--gait',action='store_true');p.add_argument('--order',default='3012');p.add_argument('--lift',type=float,default=.012);p.add_argument('--gait-seconds',type=float,default=1.);p.add_argument('--cycles',type=int,default=2)
    a=p.parse_args()
    if a.fixed_thumb:
        run_fixed_thumb(Path(a.output)/f'fixed_thumb_{a.fixed_thumb}',a.fixed_thumb);return
    if a.contact_audit:
        contact_wrench_audit(Path(a.output)/'contact_wrench_audit.json');return
    if a.transfer_alternative:
        run_transfer_alternative(Path(a.output)/f'transfer_alternative_{a.transfer_alternative}',a.transfer_alternative);return
    if a.verify:
        verify_saved(a.verify);return
    if a.gait:
        run_gait(Path(a.output)/f'gait_{a.order}_lift{a.lift:g}_s{a.gait_seconds:g}',a.order,a.lift,a.gait_seconds,a.cycles);return
    for target in a.target:run_probe(Path(a.output)/(f'orbit_{target:g}_s{a.seconds:g}_p{a.preload:g}_f{a.follow:g}'+(f'_r{a.preload_seconds:g}' if a.preload_seconds else '')),target,a.seconds,a.preload,a.follow,a.preload_seconds)
if __name__=='__main__':main()
