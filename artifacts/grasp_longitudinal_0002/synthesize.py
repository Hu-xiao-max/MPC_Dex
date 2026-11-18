"""Test exactly three previously specified longitudinal grasp layouts."""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import mujoco
import numpy as np
from scipy.optimize import least_squares

from mpc_dex.env import AllegroRotationEnv,EnvConfig
from scripts.find_grasp import initialize,object_contacts,rollout
from artifacts.grasp_40mm_0001.synthesize import cold_check

OUT=Path(__file__).resolve().parent
LAYOUTS=[
    dict(name='A',diameter=.04,center=[.065,0.,.030],radius=.033,
         angles_degrees=[70,0,-70,180],tip_z=[.055,.050,.040,.005]),
    dict(name='B',diameter=.045,center=[.062,0.,.030],radius=.035,
         angles_degrees=[70,0,-70,175],tip_z=[.050,.045,.035,0.]),
    dict(name='C',diameter=.04,center=[.067,.004,.030],radius=.034,
         angles_degrees=[85,0,-70,160],tip_z=[.050,.045,.035,0.]),
]
SEED=np.array([-.14337,.37803,1.22714,.82672,
               .03958,.97084,.52641,.36228,
               .23320,.64921,1.15055,.73901,
              1.24122,.26302,1.29269,1.02657])


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def solve_ik(env,layout):
    sites=[env.model.site(f+'_tip_site').id for f in ('ff','mf','rf','th')]
    rotation=env.data.xmat[env.palm_id].reshape(3,3).copy()
    origin=env.data.xpos[env.palm_id].copy()
    angles=np.radians(layout['angles_degrees'])
    desired=np.tile(layout['center'],(4,1))
    desired[:,0]+=layout['radius']*np.cos(angles)
    desired[:,1]+=layout['radius']*np.sin(angles)
    desired[:,2]=layout['tip_z']
    def residual(q):
        env.data.qpos[env.q_ids]=q
        mujoco.mj_kinematics(env.model,env.data)
        measured=(env.data.site_xpos[sites]-origin)@rotation
        return np.r_[(measured-desired).ravel()*100,(q-SEED)*.015]
    fit=least_squares(residual,np.clip(SEED,env.low+1e-5,env.high-1e-5),
        bounds=(env.low+1e-6,env.high-1e-6),max_nfev=100,ftol=1e-5,xtol=1e-5,gtol=1e-5)
    measured=(env.data.site_xpos[sites]-origin)@rotation
    return fit.x,desired,measured,float(np.linalg.norm(fit.fun[:12])/100)


def contact_geometry(env):
    result=[]
    rotation=env.data.xmat[env.palm_id].reshape(3,3)
    for index,contact in enumerate(env.data.contact):
        if env.object_geom not in (contact.geom1,contact.geom2):
            continue
        other=contact.geom2 if contact.geom1==env.object_geom else contact.geom1
        body=int(env.model.geom_bodyid[other])
        force=np.zeros(6)
        mujoco.mj_contactForce(env.model,env.data,index,force)
        result.append(dict(body=mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_BODY,body),
            geom_id=int(other),geom_type=int(env.model.geom_type[other]),
            distance_m=float(contact.dist),position_world=contact.pos.tolist(),
            position_in_palm=(rotation.T@(contact.pos-env.data.xpos[env.palm_id])).tolist(),
            normal_force_N=float(force[0])))
    return result


def main():
    report_path=OUT/'search_report.json'
    if report_path.exists():raise RuntimeError('Refusing to repeat or overwrite the three-candidate search')
    started=time.monotonic()
    reports=[]
    accepted=[]
    for layout in LAYOUTS:
        config=EnvConfig(object_diameter=layout['diameter'])
        env=AllegroRotationEnv(config)
        try:
            env.reset(seed=9)
            q,desired,measured,error=solve_ik(env,layout)
            center=np.array(layout['center'])
            initialize(env,q,center)
            relative=np.array([math.sqrt(.5),math.sqrt(.5),0.,0.])
            orientation=np.empty(4)
            mujoco.mju_mulQuat(orientation,env.data.xquat[env.palm_id],relative)
            env.data.qpos[env.object_qadr+3:env.object_qadr+7]=orientation
            mujoco.mj_forward(env.model,env.data)
            env.initial_position=env.object_position().copy()
            forbidden,contacts=object_contacts(env)
            geometry=contact_geometry(env)
            record=dict(layout=layout,joint_targets=q.tolist(),
                desired_tip_positions_palm=desired.tolist(),measured_tip_positions_palm=measured.tolist(),
                ik_error_m=error,initial_relative_quaternion_wxyz=relative.tolist(),
                initial_forbidden=bool(forbidden),initial_contacts=geometry,
                maximum_initial_penetration_m=max((-c[1] for c in contacts),default=0.),
                object_length_m=config.object_length,accepted=False)
            reports.append(record)
            failures=[]
            if forbidden:failures.append('palm/table support contact')
            if error>.02:failures.append('IK residual above 20 mm')
            if any(c[1]<-.009 for c in contacts):failures.append('object-contact penetration above 9 mm')
            if failures:
                record.update(rejection_stage='initial geometry',reasons=failures,physics_steps=0)
            else:
                ok,settling=rollout(env,1.5,center)
                record['settling']=settling
                if not ok:
                    record.update(rejection_stage='settling',reasons=['existing 1.5 second settling gate failed'])
                else:
                    ok,verification=rollout(env,6.,env.object_position().copy(),strict=True)
                    record['verification']=verification
                    if not ok:
                        record.update(rejection_stage='warm hold',reasons=['existing six-second strict hold gate failed'])
                    else:
                        saved={key:getattr(env.data,key).copy() for key in ('qpos','qvel','ctrl')}
                        ok,replay=cold_check(config,saved)
                        record['replay']=replay
                        if not ok:
                            record.update(rejection_stage='cold hold',reasons=['cold six-second four-finger/no-support gate failed'])
                        else:
                            record['accepted']=True
                            output=OUT/f'initial_grasp_layout_{layout["name"]}.npz'
                            metadata=dict(config=asdict(config),synthesis=record,verification=verification,
                                replay=replay,mujoco_version=mujoco.__version__,seed=9,
                                intended_principal_axis_in_palm=[0.,0.,1.],
                                source_relation='Exploratory visual-alignment inference only; not author calibration',
                                generated_by=str(Path(__file__).relative_to(ROOT)),
                                source_model_sha256=sha256(ROOT/'assets/allegro/right_hand.xml'),
                                environment_source_sha256=sha256(ROOT/'mpc_dex/env.py'),
                                synthesis_source_sha256=sha256(__file__))
                            with output.open('xb') as stream:
                                np.savez(stream,**saved,metadata_json=np.array(json.dumps(metadata)))
                            output.chmod(0o444)
                            record['grasp']=str(output.relative_to(ROOT));record['grasp_sha256']=sha256(output)
                            accepted.append(record['grasp'])
            print(json.dumps(record),flush=True)
        finally:
            env.close()
    summary=dict(candidate_limit=3,candidates_attempted=len(reports),accepted_grasps=accepted,
        wall_seconds=time.monotonic()-started,object_length_m=.12,
        criteria=dict(max_ik_error_m=.02,max_initial_penetration_m=.009,
                      cold_hold_seconds=6.,minimum_force_bearing_fingers=4,forbidden_support=['palm','table']),
        source_relation='Exploratory visual-alignment inference only; not author calibration',
        environment_source_sha256=sha256(ROOT/'mpc_dex/env.py'),
        synthesis_source_sha256=sha256(__file__),reports=reports)
    report_path.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({key:value for key,value in summary.items() if key!='reports'}),flush=True)


if __name__=='__main__':main()
