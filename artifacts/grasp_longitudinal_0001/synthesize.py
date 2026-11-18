"""Bounded exploratory longitudinal grasp; visual alignment is not calibration."""
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


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def longitudinal_ik(env,center,radius,outer_angle,axis_offsets,seed):
    sites=[env.model.site(f+'_tip_site').id for f in ('ff','mf','rf','th')]
    rotation=env.data.xmat[env.palm_id].reshape(3,3).copy()
    origin=env.data.xpos[env.palm_id].copy()
    angles=np.array([outer_angle,0.,-outer_angle,math.pi])
    desired=np.tile(center,(4,1))
    desired[:,0]+=radius*np.cos(angles)
    desired[:,1]+=radius*np.sin(angles)
    desired[:,2]+=axis_offsets
    def residual(q):
        env.data.qpos[env.q_ids]=q
        mujoco.mj_kinematics(env.model,env.data)
        measured=(env.data.site_xpos[sites]-origin)@rotation
        return np.r_[(measured-desired).ravel()*100,(q-seed)*.015]
    fit=least_squares(residual,np.clip(seed,env.low+1e-5,env.high-1e-5),
        bounds=(env.low+1e-6,env.high-1e-6),max_nfev=100,ftol=1e-5,xtol=1e-5,gtol=1e-5)
    return fit.x,float(np.linalg.norm(fit.fun[:12])/100),desired


def main():
    if (OUT/'search_report.json').exists():
        raise RuntimeError('Refusing to repeat the bounded search in this artifact directory')
    with np.load(ROOT/'assets/initial_grasp.npz',allow_pickle=False) as archive:
        seed=archive['ctrl'].copy()
    variants=[
        (.065,.070,60,.031,[0.,.025,0.,-.040]),
        (.065,.070,90,.031,[0.,.025,0.,-.040]),
        (.070,.060,60,.031,[.005,.025,.005,-.030]),
        (.070,.060,90,.031,[.005,.025,.005,-.030]),
        (.075,.070,60,.031,[0.,.020,0.,-.040]),
        (.075,.070,90,.031,[0.,.020,0.,-.040]),
        (.060,.075,75,.030,[0.,.020,0.,-.040]),
        (.065,.065,75,.029,[0.,.020,0.,-.035]),
        (.070,.080,75,.030,[0.,.025,0.,-.045]),
        (.075,.065,75,.029,[0.,.025,0.,-.035]),
    ]
    reports=[]
    accepted_paths=[]
    started=time.monotonic()
    for diameter in (.04,.045):
        config=EnvConfig(object_diameter=diameter)
        env=AllegroRotationEnv(config)
        try:
            for variant,(x,z,angle,radius,offsets) in enumerate(variants):
                env.reset(seed=9)
                center=np.array([x,0.,z])
                radius+=.5*(diameter-.04)
                q,error,desired=longitudinal_ik(env,center,radius,math.radians(angle),offsets,seed)
                initialize(env,q,center)
                relative=np.array([math.sqrt(.5),math.sqrt(.5),0.,0.])
                orientation=np.empty(4)
                mujoco.mju_mulQuat(orientation,env.data.xquat[env.palm_id],relative)
                env.data.qpos[env.object_qadr+3:env.object_qadr+7]=orientation
                mujoco.mj_forward(env.model,env.data)
                env.initial_position=env.object_position().copy()
                forbidden,contacts=object_contacts(env)
                record=dict(candidate=len(reports),diameter=diameter,variant=variant,
                    center=center.tolist(),radius=radius,outer_angle_degrees=angle,
                    axis_offsets=offsets,desired_tip_positions_palm=desired.tolist(),
                    initial_relative_quaternion_wxyz=relative.tolist(),ik_error=error,
                    initial_contacts=contacts,initial_forbidden=bool(forbidden))
                reports.append(record)
                if forbidden or error>.02 or any(c[1]<-.009 for c in contacts):
                    record.update(accepted=False,rejection='initial geometry')
                else:
                    accepted,settling=rollout(env,1.5,center)
                    record['settling']=settling
                    if not accepted:
                        record.update(accepted=False,rejection='settling')
                    else:
                        accepted,verification=rollout(env,6.,env.object_position().copy(),strict=True)
                        record['verification']=verification
                        if not accepted:
                            record.update(accepted=False,rejection='warm hold')
                        else:
                            saved={key:getattr(env.data,key).copy() for key in ('qpos','qvel','ctrl')}
                            accepted,replay=cold_check(config,saved)
                            record.update(accepted=accepted,replay=replay)
                            if not accepted:
                                record['rejection']='cold four-finger hold'
                            else:
                                output=OUT/f'initial_grasp_{round(diameter*1000)}mm_longitudinal.npz'
                                metadata=dict(config=asdict(config),synthesis=record,
                                    verification=verification,replay=replay,
                                    intended_principal_axis_in_palm=[0.,0.,1.],
                                    source_relation='Exploratory visual-alignment inference only; not author calibration',
                                    generated_by=str(Path(__file__).relative_to(ROOT)),
                                    mujoco_version=mujoco.__version__,seed=9,
                                    source_model_sha256=sha256(ROOT/'assets/allegro/right_hand.xml'),
                                    environment_source_sha256=sha256(ROOT/'mpc_dex/env.py'),
                                    synthesis_source_sha256=sha256(__file__))
                                with output.open('xb') as stream:
                                    np.savez(stream,**saved,metadata_json=np.array(json.dumps(metadata)))
                                output.chmod(0o444)
                                record['grasp_sha256']=sha256(output)
                                record['grasp']=str(output.relative_to(ROOT))
                                accepted_paths.append(record['grasp'])
                print(json.dumps(record),flush=True)
                if record['accepted']:
                    break
        finally:
            env.close()
        if accepted_paths:
            break
    summary=dict(candidate_limit=20,candidates_attempted=len(reports),accepted_grasps=accepted_paths,
        wall_seconds=time.monotonic()-started,source_relation='Exploratory visual-alignment inference only; not author calibration',
        reports=reports)
    (OUT/'search_report.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({key:value for key,value in summary.items() if key!='reports'}),flush=True)


if __name__=='__main__':
    main()
