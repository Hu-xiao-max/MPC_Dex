"""Search and physically verify a stationary unsupported Allegro grasp.

This is initialization synthesis, not a rotation controller. The free object is
never constrained or actuated. Candidates use fingertip inverse kinematics,
then run under normal gravity with fixed position targets. Only a state passing
an independent six-second replay is saved.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from scipy.optimize import least_squares

from mpc_dex.env import AllegroRotationEnv, EnvConfig


def object_contacts(env):
    contacts = []
    forbidden = False
    for contact_index, c in enumerate(env.data.contact):
        if env.object_geom not in (c.geom1, c.geom2):
            continue
        other = c.geom2 if c.geom1 == env.object_geom else c.geom1
        body = int(env.model.geom_bodyid[other])
        name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, body)
        forbidden |= other == env.table_geom or body == env.palm_id
        force = np.zeros(6)
        mujoco.mj_contactForce(env.model, env.data, contact_index, force)
        contacts.append((name, float(c.dist), float(force[0])))
    return forbidden, contacts


def initialize(env, q, center, targets=None):
    env.reset()
    d = env.data
    d.qpos[env.q_ids] = q
    d.ctrl[:] = q if targets is None else targets
    R = d.xmat[env.palm_id].reshape(3, 3)
    d.qpos[env.object_qadr:env.object_qadr + 3] = d.xpos[env.palm_id] + R @ center
    mujoco.mj_forward(env.model, d)
    env.initial_position = env.object_position().copy()


def ik(env, center, radius, seed, angular_offset=0.):
    """Put three fingers and the opposing thumb on rod perimeter."""
    site_ids = [env.model.site(f + "_tip_site").id for f in ("ff", "mf", "rf", "th")]
    R = env.data.xmat[env.palm_id].reshape(3, 3).copy()
    origin = env.data.xpos[env.palm_id].copy()
    angles = np.array([np.pi / 2 + angular_offset] * 3 + [-np.pi / 2 + angular_offset])
    desired = np.tile(center, (4, 1))
    desired[:, 0] += radius * np.cos(angles)
    desired[:, 2] += radius * np.sin(angles)
    desired[:, 1] = [.039, 0., -.039, .01]

    def residual(q):
        env.data.qpos[env.q_ids] = q
        mujoco.mj_kinematics(env.model, env.data)
        pos = (env.data.site_xpos[site_ids] - origin) @ R
        return np.r_[(pos - desired).ravel() * 100, (q - seed) * .015]

    fit = least_squares(residual, np.clip(seed, env.low + 1e-5, env.high - 1e-5),
                        bounds=(env.low + 1e-6, env.high - 1e-6), max_nfev=65,
                        ftol=1e-5, xtol=1e-5, gtol=1e-5)
    return fit.x, float(np.linalg.norm(fit.fun[:12]) / 100)


def rollout(env, seconds, reference, *, strict=False):
    max_speed = 0.
    settled_max_speed = 0.
    max_angular_speed = 0.
    settled_max_angular_speed = 0.
    max_displacement = 0.
    forbidden_seen = False
    contact_seen = set()
    sample_start = int(.8 / env.config.timestep)
    for step in range(round(seconds / env.config.timestep)):
        mujoco.mj_step(env.model, env.data)
        forbidden, contacts = object_contacts(env)
        forbidden_seen |= forbidden
        contact_seen.update(x[0] for x in contacts)
        position = env.object_position()
        displacement = float(np.linalg.norm(position - reference))
        max_displacement = max(max_displacement, displacement)
        speed = float(np.linalg.norm(env.data.qvel[env.object_dadr:env.object_dadr+3]))
        angular_speed = float(np.linalg.norm(env.data.qvel[env.object_dadr+3:env.object_dadr+6]))
        max_speed = max(max_speed, speed)
        max_angular_speed = max(max_angular_speed, angular_speed)
        if step >= sample_start:
            settled_max_speed = max(settled_max_speed, speed)
            settled_max_angular_speed = max(settled_max_angular_speed, angular_speed)
        if forbidden or displacement > .08 or not np.isfinite(env.data.qpos).all():
            return False, dict(elapsed=(step + 1)*env.config.timestep,
                max_speed=max_speed, settled_max_speed=settled_max_speed,
                max_angular_speed=max_angular_speed, displacement=max_displacement,
                forbidden=forbidden_seen, contacts=sorted(contact_seen))
    forbidden, contacts = object_contacts(env)
    force_bodies = {x[0] for x in contacts if x[2] > .01}
    accepted = (not forbidden_seen and max_displacement < (.012 if strict else .04)
                and settled_max_speed < (.025 if strict else .10)
                and settled_max_angular_speed < (.2 if strict else 2.)
                and max_speed < .3 and len(force_bodies) >= 3)
    return accepted, dict(elapsed=seconds, max_speed=max_speed,
        settled_max_speed=settled_max_speed, max_angular_speed=max_angular_speed,
        settled_max_angular_speed=settled_max_angular_speed, settling_allowance_seconds=.8,
        displacement=max_displacement, forbidden=forbidden_seen,
        contacts=contacts, contact_bodies=len(force_bodies),
        final_position=env.object_position().tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="assets/initial_grasp.npz")
    parser.add_argument("--seconds", type=float, default=240.)
    parser.add_argument("--candidates", type=int, default=600)
    parser.add_argument("--seed", type=int, default=9)
    args = parser.parse_args()
    start = time.monotonic()
    env = AllegroRotationEnv(EnvConfig())
    env.reset()
    rng = np.random.default_rng(args.seed)
    best_score = -np.inf
    best = None
    seed = np.array(env.config.initial_hand_q)
    for index in range(args.candidates):
        if time.monotonic() - start > args.seconds:
            break
        if index < 12:
            center = np.array([.075 + .005*(index % 4), 0., .015 + .008*(index//4)])
            radius, angle = .031, 0.
        else:
            center = np.array([rng.uniform(.06, .10), rng.uniform(-.008, .008), rng.uniform(.005, .045)])
            radius, angle = rng.uniform(.027, .036), rng.uniform(-.45, .45)
        q, error = ik(env, center, radius, seed, angle)
        initialize(env, q, center)
        forbidden, contacts = object_contacts(env)
        if forbidden or error > .02 or any(c[1] < -.009 for c in contacts):
            continue
        accepted, report = rollout(env, 1.5, center)
        score = report["elapsed"] - 10*report["displacement"] - 2*report["max_speed"]
        if score > best_score:
            best_score = score
            best = dict(index=index, center=center.tolist(), q=q.tolist(), radius=radius,
                        angle=angle, ik_error=error, report=report)
            print(json.dumps({"best": best}), flush=True)
        if not accepted:
            continue
        reference = env.object_position().copy()
        verified, verification = rollout(env, 6., reference, strict=True)
        if not verified:
            continue
        # Copy only the physical state after settling, clear all simulator
        # workspace (including solver warm starts), then independently replay.
        saved = {key: getattr(env.data, key).copy() for key in ("qpos", "qvel", "ctrl")}
        reference = env.object_position().copy()
        mujoco.mj_resetData(env.model, env.data)
        for key, value in saved.items():
            getattr(env.data, key)[:] = value
        env.data.time = 0.
        mujoco.mj_forward(env.model, env.data)
        verified, replay = rollout(env, 6., reference, strict=True)
        if not verified:
            continue
        synthesis = dict(index=index, center=center.tolist(), q=q.tolist(), radius=radius,
                         angle=angle, ik_error=error)
        metadata = dict(config=asdict(env.config), synthesis=synthesis,
                        verification=verification, replay=replay,
                        mujoco_version=mujoco.__version__, seed=args.seed,
                        generated_by="scripts/find_grasp.py")
        metadata["source_model_sha256"] = hashlib.sha256(
            (Path(__file__).resolve().parents[1]/"assets/allegro/right_hand.xml").read_bytes()).hexdigest()
        metadata["environment_source_sha256"] = hashlib.sha256(
            (Path(__file__).resolve().parents[1]/"mpc_dex/env.py").read_bytes()).hexdigest()
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output, **saved, metadata_json=np.array(json.dumps(metadata)))
        print(json.dumps({"verified": True, "output": str(output), "verification": replay}), flush=True)
        return 0
    print(json.dumps({"verified": False, "best": best, "wall_seconds": time.monotonic()-start}), flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
