"""Re-observe existing exact/clipped commands for contact identities, no new sweep."""
from pathlib import Path
import json
import sys
D = Path(__file__).resolve().parent
sys.path.insert(0, str(D / 'source'))
import mujoco
import numpy as np
from mpc_dex.env import AllegroRotationEnv, EnvConfig
from mpc_dex.contracts import sha256
from scripts.audit_policy import PhysicsObserver, write_json


class ContactObserver(PhysicsObserver):
    def __init__(self, env):
        super().__init__(env)
        self.states = []; self.geometric = []; self.loaded = []; self.fingers = []

    def capture(self):
        super().capture()
        env = self.env
        geometric = set(); loaded = set(); fingers = set()
        for i, c in enumerate(env.data.contact):
            if env.object_geom not in (c.geom1, c.geom2):
                continue
            other = int(c.geom1 if c.geom2 == env.object_geom else c.geom2)
            geometric.add(other)
            wrench = np.zeros(6); mujoco.mj_contactForce(env.model, env.data, i, wrench)
            if wrench[0] > 1e-6:
                loaded.add(other)
                body = int(env.model.geom_bodyid[other])
                name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, body) or ''
                if name[:2] in {'ff', 'mf', 'rf', 'th'}:
                    fingers.add(name[:2])
        self.states.append(np.r_[env.data.time, env.data.qpos, env.data.qvel])
        self.geometric.append(tuple(sorted(geometric)))
        self.loaded.append(tuple(sorted(loaded)))
        self.fingers.append(tuple(sorted(fingers)))


cfg = json.loads((D / 'config.json').read_text())
env_config = EnvConfig(**cfg['environment'])
with np.load(D / 'teacher_transitions.npz', allow_pickle=False) as a:
    actions = a['actions'].copy()
traces = {}
for name, applied in [('exact', actions), ('clipped_1e6', np.clip(actions, -1 + 1e-6, 1 - 1e-6))]:
    env = AllegroRotationEnv(env_config); env.reset(seed=0)
    observer = ContactObserver(env)
    try:
        for action in applied:
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        states = np.asarray(observer.states)
        traces[name] = dict(states=states, geometric=observer.geometric, loaded=observer.loaded, fingers=observer.fingers)
        np.savez_compressed(D / f'contact_trace_{name}.npz', states=states,
            state_columns=np.asarray(['time'] + [f'qpos_{i}' for i in range(env.model.nq)] +
                                     [f'qvel_{i}' for i in range(env.model.nv)]),
            geometric_partner_geom_ids=np.asarray([json.dumps(x) for x in observer.geometric]),
            loaded_partner_geom_ids=np.asarray([json.dumps(x) for x in observer.loaded]),
            force_bearing_finger_names=np.asarray([json.dumps(x) for x in observer.fingers]))
        if name == 'exact':
            geom_names = {i: dict(geom_name=mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, i),
                                 body_name=mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY,
                                                            int(env.model.geom_bodyid[i])))
                          for i in range(env.model.ngeom)}
    finally:
        observer.close(); env.close()

exact, clipped = traces['exact'], traces['clipped_1e6']
n = min(len(exact['states']), len(clipped['states']))
report = dict(purpose='Contact identity instrumentation of two already evaluated fixed command sequences',
              source_sha256=sha256(__file__), geom_id_map=geom_names,
              first_command_max_error=float(abs(actions[0] - np.clip(actions[0], -1 + 1e-6, 1 - 1e-6)).max()),
              first_step_qpos_max_error=float(abs(exact['states'][49, 1:24] - clipped['states'][49, 1:24]).max()),
              first_step_qvel_max_error=float(abs(exact['states'][49, 24:] - clipped['states'][49, 24:]).max()),
              per_control_state_errors=[])
for key in ('geometric', 'loaded', 'fingers'):
    mismatch = next((i for i in range(n) if exact[key][i] != clipped[key][i]), None)
    report['first_' + key + '_topology_difference'] = None if mismatch is None else dict(
        physics_step=mismatch + 1, time=float(exact['states'][mismatch, 0]),
        exact=exact[key][mismatch], clipped=clipped[key][mismatch])
for i in range(49, n, 50):
    report['per_control_state_errors'].append(dict(step=(i+1)//50,
        time=float(exact['states'][i, 0]),
        qpos_max_error=float(abs(exact['states'][i, 1:24] - clipped['states'][i, 1:24]).max()),
        qvel_max_error=float(abs(exact['states'][i, 24:] - clipped['states'][i, 24:]).max())))
for label, data in [('exact', exact), ('clipped_1e6', clipped)]:
    previous = D / ('teacher_replay' if label == 'exact' else 'epsilon_0000001/clipped_teacher_open_loop')
    with np.load(previous / 'transitions.npz') as old:
        difference = float(abs(data['states'][49::50, 1:24] - old['qpos']).max())
    assert difference == 0., difference
    report[label + '_replay_qpos_max_error'] = difference
write_json(D / 'contact_divergence.json', report)
print(json.dumps(report, indent=2))
