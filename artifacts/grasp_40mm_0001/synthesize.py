"""Generate a new 40 mm grasp using the existing IK and settling implementation.

This driver never modifies the default 45 mm grasp or simulator implementation.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np

from mpc_dex.env import AllegroRotationEnv, EnvConfig
from scripts.find_grasp import ik, initialize, object_contacts, rollout


OUT = Path(__file__).resolve().parent
FINGERS = {'ff', 'mf', 'rf', 'th'}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cold_check(config, saved):
    env = AllegroRotationEnv(config)
    env.reset(seed=9)
    mujoco.mj_resetData(env.model, env.data)
    for key, value in saved.items():
        getattr(env.data, key)[:] = value
    mujoco.mj_forward(env.model, env.data)
    assert env.model.neq == 0
    assert env.model.joint('object_joint').type == mujoco.mjtJoint.mjJNT_FREE
    assert env.model.joint('object_joint').id not in env.model.actuator_trnid[:, 0]
    assert np.array_equal(env.model.opt.gravity, [0., 0., -9.81])
    assert not env.data.qfrc_applied.any() and not env.data.xfrc_applied.any()
    reference = env.object_position().copy()
    maxima = dict(displacement=0., max_speed=0., max_angular_speed=0.,
                  settled_max_speed=0., settled_max_angular_speed=0.)
    min_contacts = 4
    min_forces = {finger: float('inf') for finger in FINGERS}
    last_contacts = []
    contact_bodies = set()
    initial_warmstart_norm = float(np.linalg.norm(env.data.qacc_warmstart))
    accepted = True
    failure = None
    try:
        for step in range(3000):
            mujoco.mj_step(env.model, env.data)
            mujoco.mj_forward(env.model, env.data)
            forbidden, contacts = object_contacts(env)
            forces = {finger: 0. for finger in FINGERS}
            for name, distance, normal_force in contacts:
                contact_bodies.add(name)
                if name[:2] in forces:
                    forces[name[:2]] += max(normal_force, 0.)
            finger_count = sum(force > .01 for force in forces.values())
            min_contacts = min(min_contacts, finger_count)
            for finger, force in forces.items():
                min_forces[finger] = min(min_forces[finger], force)
            speed = float(np.linalg.norm(env.data.qvel[env.object_dadr:env.object_dadr+3]))
            angular_speed = float(np.linalg.norm(env.data.qvel[env.object_dadr+3:env.object_dadr+6]))
            displacement = float(np.linalg.norm(env.object_position()-reference))
            maxima['max_speed'] = max(maxima['max_speed'], speed)
            maxima['max_angular_speed'] = max(maxima['max_angular_speed'], angular_speed)
            maxima['displacement'] = max(maxima['displacement'], displacement)
            if step >= 400:
                maxima['settled_max_speed'] = max(maxima['settled_max_speed'], speed)
                maxima['settled_max_angular_speed'] = max(maxima['settled_max_angular_speed'], angular_speed)
            last_contacts = contacts
            if (forbidden or finger_count != 4 or not np.isfinite(env.data.qpos).all()
                    or displacement >= .012 or speed >= .3):
                accepted = False
                failure = dict(step=step+1, forbidden=bool(forbidden), finger_count=finger_count,
                               displacement=displacement, speed=speed)
                break
        accepted &= (maxima['settled_max_speed'] < .025
                     and maxima['settled_max_angular_speed'] < .2)
        geometry = []
        for index, contact in enumerate(env.data.contact):
            if env.object_geom not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == env.object_geom else contact.geom1
            body = int(env.model.geom_bodyid[other])
            force = np.zeros(6)
            mujoco.mj_contactForce(env.model, env.data, index, force)
            palm_R = env.data.xmat[env.palm_id].reshape(3, 3)
            geometry.append(dict(body=mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, body),
                                 geom=int(other), distance_m=float(contact.dist),
                                 normal_force_N=float(force[0]),
                                 position_world=contact.pos.tolist(),
                                 position_in_palm=(palm_R.T @ (contact.pos-env.data.xpos[env.palm_id])).tolist(),
                                 contact_frame=contact.frame.tolist()))
        return bool(accepted), dict(elapsed=(step+1)*config.timestep, **maxima,
            settling_allowance_seconds=.8, forbidden=bool(failure and failure['forbidden']),
            failure=failure, contacts=last_contacts, contact_bodies=4 if accepted else min_contacts,
            contact_body_names=sorted(contact_bodies), min_force_bearing_fingers=min_contacts,
            min_normal_force_by_finger_N=min_forces, final_position=env.object_position().tolist(),
            initial_qacc_warmstart_norm=initial_warmstart_norm, cold_reset=True,
            gravity=env.model.opt.gravity.tolist(), free_object=True,
            no_object_actuator=True, equality_constraints=int(env.model.neq),
            contact_geometry=geometry)
    finally:
        env.close()


def main():
    output = OUT / 'initial_grasp_40mm.npz'
    if output.exists():
        raise RuntimeError(f'Refusing to overwrite immutable grasp: {output}')
    config = EnvConfig(object_diameter=.04)
    env = AllegroRotationEnv(config)
    env.reset(seed=9)
    with np.load(ROOT / 'assets/initial_grasp.npz', allow_pickle=False) as archive:
        prior = json.loads(str(archive['metadata_json']))['synthesis']
    rng = np.random.default_rng(9)
    started = time.monotonic()
    try:
        for index in range(160):
            if time.monotonic()-started > 180:
                break
            if index < 12:
                center = np.array(prior['center'])
                center[0] += .003 * (index//4)
                radius = prior['radius']-.0025+.001*(index % 4 - 1)
                angle = prior['angle']
                seed = np.array(prior['q'])
            else:
                center = np.array([rng.uniform(.06,.1), rng.uniform(-.008,.008), rng.uniform(.005,.045)])
                radius, angle = rng.uniform(.0245,.0335), rng.uniform(-.45,.45)
                seed = np.array(config.initial_hand_q)
            q, error = ik(env, center, radius, seed, angle)
            initialize(env, q, center)
            forbidden, contacts = object_contacts(env)
            if forbidden or error > .02 or any(c[1] < -.009 for c in contacts):
                continue
            accepted, settling = rollout(env, 1.5, center)
            if not accepted:
                print(json.dumps(dict(index=index, stage='settling', accepted=False, report=settling)), flush=True)
                continue
            accepted, verification = rollout(env, 6., env.object_position().copy(), strict=True)
            if not accepted:
                print(json.dumps(dict(index=index, stage='verification', accepted=False, report=verification)), flush=True)
                continue
            saved = {key: getattr(env.data,key).copy() for key in ('qpos','qvel','ctrl')}
            accepted, replay = cold_check(config, saved)
            print(json.dumps(dict(index=index, stage='cold_replay', accepted=accepted, report=replay)), flush=True)
            if not accepted:
                continue
            metadata = dict(config=asdict(config), synthesis=dict(index=index, center=center.tolist(),
                q=q.tolist(), radius=radius, angle=angle, ik_error=error), settling=settling,
                verification=verification, replay=replay, mujoco_version=mujoco.__version__, seed=9,
                generated_by=str(Path(__file__).relative_to(ROOT)),
                helper_source_sha256=sha256(ROOT/'scripts/find_grasp.py'),
                synthesis_source_sha256=sha256(__file__),
                source_model_sha256=sha256(ROOT/'assets/allegro/right_hand.xml'),
                environment_source_sha256=sha256(ROOT/'mpc_dex/env.py'))
            with output.open('xb') as stream:
                np.savez(stream, **saved, metadata_json=np.array(json.dumps(metadata)))
            (OUT/'verification.json').write_text(json.dumps(dict(verified=True, grasp=str(output.relative_to(ROOT)),
                grasp_sha256=sha256(output), wall_seconds=time.monotonic()-started, **metadata), indent=2)+'\n')
            print(json.dumps(dict(verified=True, output=str(output), sha256=sha256(output))), flush=True)
            return
        raise RuntimeError('No candidate passed the strict cold six-second four-finger hold')
    finally:
        env.close()


if __name__ == '__main__':
    main()
