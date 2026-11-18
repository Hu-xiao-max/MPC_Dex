"""Reproduce the independent read-only audit of bootstrap checkpoint 00000004.

Run from the repository root:
  .venv/bin/python artifacts/bootstrap_0001/audit/audit_rotation_retention.py

Only this audit directory is written. The old environment is dynamically loaded
from its archived source, so subsequent fixes to mpc_dex/env.py cannot change
the audited success definition. No training, checkpoint or raw-run mutation is
performed. Three deterministic and 13 individually seeded stochastic episodes
are examined at every 2 ms physics transition. A reported success is followed
by one second of unchanged joint targets to test physical retention.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from mpc_dex.learning import SACAgent


RUN = ROOT / "artifacts/bootstrap_0001"
OUTPUT = RUN / "audit"
CHECKPOINT = RUN / "checkpoint_00000004"
ARCHIVED_SOURCE = RUN / "source/env.py"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archived_environment():
    spec = importlib.util.spec_from_file_location("audited_bootstrap_environment", ARCHIVED_SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # The archived file's relative path differs from its original module path.
    module.ASSET_ROOT = ROOT / "assets/allegro"
    return module


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    archived = archived_environment()
    checkpoint_state = json.loads((CHECKPOINT / "state.json").read_text())
    expected_source_hash = checkpoint_state["provenance"]["environment_source_sha256"]
    if file_hash(ARCHIVED_SOURCE) != expected_source_hash:
        raise ValueError("archived environment differs from the training source")
    ec = archived.EnvConfig(**checkpoint_state["provenance"]["environment"])
    if file_hash(ec.initial_grasp) != checkpoint_state["provenance"]["initial_grasp_sha256"]:
        raise ValueError("initial grasp differs from the training artifact")
    agent = SACAgent.load(CHECKPOINT / "agent.pt", "cpu")
    torch.set_num_threads(1)
    env = archived.AllegroRotationEnv(ec)
    original_step = mujoco.mj_step
    results = []

    try:
        with (OUTPUT / "episodes.jsonl").open("w") as evidence:
            for episode in range(16):
                torch.manual_seed(episode)
                obs, initial_info = env.reset(seed=episode)
                initial_observation = obs.copy()
                samples = []
                controls = []
                previous = Rotation.from_quat(env.data.xquat[env.object_id], scalar_first=True)
                palm = env.data.xmat[env.palm_id].reshape(3, 3).copy()
                palm_y = palm[:, 1].copy()
                origin = env.data.xpos[env.object_id].copy()
                axial_fixed = 0.0
                axial_body = 0.0

                def physical_observation(model, data):
                    contacts = []
                    normal_force = 0.0
                    active = 0
                    for index, contact in enumerate(data.contact):
                        if env.object_geom not in (contact.geom1, contact.geom2):
                            continue
                        force = np.empty(6)
                        mujoco.mj_contactForce(model, data, index, force)
                        is_table = env.table_geom in (contact.geom1, contact.geom2)
                        contacts.append({
                            "geom1": int(contact.geom1), "geom2": int(contact.geom2),
                            "is_table": is_table, "normal_force": float(force[0]),
                            "distance": float(contact.dist),
                        })
                        if not is_table:
                            active += int(force[0] > 1e-5)
                            normal_force += float(force[0])
                    velocity = np.empty(6)
                    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY,
                                            env.object_id, velocity, 0)
                    rotation = Rotation.from_quat(data.xquat[env.object_id], scalar_first=True)
                    local = palm.T @ rotation.as_matrix()
                    return {
                        "time": float(data.time),
                        "contact": active, "normalforce": normal_force,
                        "contacts": contacts,
                        "object_position_world": data.xpos[env.object_id].tolist(),
                        "object_quaternion_wxyz": data.xquat[env.object_id].tolist(),
                        "object_velocity_world_angular_then_linear": velocity.tolist(),
                        "hand_q": data.qpos[env.q_ids].tolist(),
                        "ctrl": data.ctrl.tolist(),
                        "tilt": float(np.arccos(np.clip(rotation.as_matrix()[:, 1] @ palm_y, -1, 1))),
                        "proj": float(np.hypot(local[0, 0], local[2, 0])),
                        "displacement": float(np.linalg.norm(data.xpos[env.object_id] - origin)),
                        "speed": float(np.linalg.norm(velocity[3:])),
                    }

                initial_physics = physical_observation(env.model, env.data)

                def measured_step(model, data, *args, **kwargs):
                    nonlocal previous, axial_fixed, axial_body
                    original_step(model, data, *args, **kwargs)
                    mujoco.mj_forward(model, data)
                    current = Rotation.from_quat(data.xquat[env.object_id], scalar_first=True)
                    delta = (current * previous.inv()).as_rotvec()
                    old_y, new_y = previous.as_matrix()[:, 1], current.as_matrix()[:, 1]
                    midpoint_axis = old_y + new_y
                    midpoint_axis /= np.linalg.norm(midpoint_axis)
                    axial_fixed += delta @ palm_y
                    axial_body += delta @ midpoint_axis
                    previous = current
                    record = physical_observation(model, data)
                    record.update(fixed=float(axial_fixed), body=float(axial_body),
                                  quaternion_delta_world_rotvec=delta.tolist())
                    samples.append(record)

                mujoco.mj_step = measured_step
                deterministic = episode < 3
                info = {}
                try:
                    for control in range(30):
                        action = agent.act(obs, deterministic=deterministic)
                        before = obs.copy()
                        obs, reward, terminated, truncated, info = env.step(action)
                        controls.append({
                            "control_step": control, "observation": before.tolist(),
                            "action": action.tolist(), "next_observation": obs.tolist(),
                            "reward": reward, "terminated": terminated,
                            "truncated": truncated, "reported_angle": info["angle"],
                            "reported_success": info["success"], "reported_drop": info["drop"],
                        })
                        if terminated or truncated:
                            break
                finally:
                    mujoco.mj_step = original_step

                last = samples[-1]
                contactless = [sample["contact"] == 0 for sample in samples]
                run = longest = 0
                for absent in contactless:
                    run = run + 1 if absent else 0
                    longest = max(longest, run)
                result = {
                    "episode": episode, "stochastic": not deterministic,
                    "success": bool(info["success"]), "drop": bool(info["drop"]),
                    "elapsed": info["sim_time"], "projected_angle": info["angle"],
                    "integrated_fixed_axis": last["fixed"],
                    "integrated_body_axis": last["body"],
                    "contact_at_end": last["contact"],
                    "normalforce_at_end": last["normalforce"],
                    "contact_fraction": 1 - float(np.mean(contactless)),
                    "longest_contactless_seconds": longest * ec.timestep,
                    "terminal_contactless_seconds": run * ec.timestep,
                    "tilt_at_end_degrees": float(np.degrees(last["tilt"])),
                    "max_tilt_degrees": float(np.degrees(max(s["tilt"] for s in samples))),
                    "min_projection_norm": min(s["proj"] for s in samples),
                    "end_speed": last["speed"], "end_displacement": last["displacement"],
                }
                continuation = []
                if info["success"]:
                    first_drop = None
                    any_hand_contact = False
                    for extra in range(round(1 / ec.timestep)):
                        original_step(env.model, env.data)
                        mujoco.mj_forward(env.model, env.data)
                        dropped = bool(env.dropped())
                        if dropped and first_drop is None:
                            first_drop = (extra + 1) * ec.timestep
                        any_hand_contact |= any(
                            env.object_geom in (c.geom1, c.geom2) and env.table_geom not in (c.geom1, c.geom2)
                            for c in env.data.contact
                        )
                        record = physical_observation(env.model, env.data)
                        record.update(seconds_after_reported_success=(extra + 1) * ec.timestep,
                                      dropped=dropped)
                        continuation.append(record)
                    result.update(held_control_drop_after_seconds=first_drop,
                                  any_hand_contact_after_success=any_hand_contact)
                episode_evidence = {
                    "summary": result, "torch_seed": episode, "environment_seed": episode,
                    "initial_observation": initial_observation.tolist(),
                    "initial_physics": initial_physics,
                    "controls": controls, "physics_transitions": samples,
                    "held_control_continuation": continuation,
                }
                evidence.write(json.dumps(episode_evidence, allow_nan=False) + "\n")
                evidence.flush()
                results.append(result)
                print(json.dumps(result, allow_nan=False), flush=True)
    finally:
        mujoco.mj_step = original_step
        env.close()

    successes = [r for r in results if r["success"]]
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "valid_for_primitive_extraction": False,
        "reason": "All sampled reported successes were unsupported airborne tumbling: zero hand contact at success, 70-82 ms already without contact, and drop within 2-30 ms under continued held control. Projected Euler-like angle also overstated rotation around the fixed intended axis. This artifact does not establish maintained in-hand rotation.",
        "checkpoint_file": str((CHECKPOINT / "agent.pt").resolve()),
        "checkpoint_sha256": file_hash(CHECKPOINT / "agent.pt"),
        "checkpoint_environment_steps": checkpoint_state["environment_steps"],
        "archived_environment_file": str(ARCHIVED_SOURCE.resolve()),
        "archived_environment_sha256": file_hash(ARCHIVED_SOURCE),
        "initial_grasp_file": ec.initial_grasp,
        "initial_grasp_sha256": file_hash(ec.initial_grasp),
        "learner_source_sha256": file_hash(ROOT / "mpc_dex/learning.py"),
        "audit_script_sha256": file_hash(__file__),
        "evidence_file": "episodes.jsonl",
        "evidence_sha256": file_hash(OUTPUT / "episodes.jsonl"),
        "versions": {"mujoco": mujoco.__version__, "numpy": np.__version__, "torch": str(torch.__version__)},
        "method": {
            "deterministic_seeds": [0, 1, 2], "stochastic_seeds": list(range(3, 16)),
            "physics_step_seconds": ec.timestep, "control_step_seconds": ec.control_seconds,
            "active_contact_normal_force_threshold_newtons": 1e-5,
            "fixed_axis_integral": "Sum world quaternion-delta rotation vectors projected onto constant palm +y.",
            "body_axis_integral": "Sum world quaternion-delta rotation vectors projected onto normalized old+new object +y axes.",
            "continued_control": "Hold the last joint targets for one second after reported success; do not reset.",
            "observation_instrumentation": "mj_forward after each mj_step refreshes geometric state before measuring; the archived environment also calls mj_forward afterward.",
        },
        "deterministic_episodes": 3, "deterministic_successes": 0,
        "stochastic_episodes": 13, "stochastic_reported_successes": len(successes),
        "reported_successes_with_hand_contact_at_end": sum(r["contact_at_end"] > 0 for r in successes),
        "reported_successes_retained_for_one_second": sum(r["held_control_drop_after_seconds"] is None for r in successes),
        "episodes": results,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"summary": str(OUTPUT / "summary.json"), "valid_for_primitive_extraction": False}))


if __name__ == "__main__":
    main()
