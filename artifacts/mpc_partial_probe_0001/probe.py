"""Diagnostic MPC initialized from partial physical motion, not a paper primitive.

Nothing here supplies the production extraction/collection pipeline. This
bounded probe tests whether the current exact planner can execute useful
contact-preserving commands from a measured partial coordination prior.
"""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from mpc_dex.env import AllegroRotationEnv, EnvConfig
from mpc_dex.mpc import CEMConfig, NoFeasiblePlan, PrimitiveCEM
from mpc_dex.primitives import PrimitiveBasis
from mpc_dex.rollouts import MujocoRolloutEvaluator, RolloutCostConfig
from mpc_dex.contracts import environment_contract

OUT = Path(__file__).resolve().parent
if (OUT / 'report.json').exists():
    raise FileExistsError('Use a fresh diagnostic directory')
source = ROOT / 'artifacts/rotation_probe/orbit_0.8_s8_p1_f0.npz'
ec = EnvConfig(initial_grasp=str(ROOT / 'assets/initial_grasp.npz'))
env = AllegroRotationEnv(ec)
obs, _ = env.reset(seed=0)
with np.load(source, allow_pickle=False) as archive:
    frames = np.vstack((env.data.qpos[env.q_ids], archive['qpos'][:, env.q_ids]))
prior = PrimitiveBasis.from_trajectories(frames, provenance={
    'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'purpose': 'partial-motion diagnostic only; not a successful-policy primitive',
    'full_turns': 0})
planner = PrimitiveCEM(prior, env.low, env.high,
                       CEMConfig(timestep_seconds=.1), seed=42)
planner.reset(env.data.ctrl)
evaluator = MujocoRolloutEvaluator(env, timestep_seconds=.1,
                                   config=RolloutCostConfig(workers=4))
start = time.monotonic()
steps = []
data = {k: [] for k in ['obs', 'actions', 'rewards', 'next_obs', 'qpos', 'qvel']}
report = dict(purpose='partial-prior MPC feasibility diagnostic',
              production_eligible=False, primitive=prior.metadata,
              components=prior.components, environment_contract=environment_contract(ec),
              steps=steps, status='running')
try:
    for step in range(10):
        began = time.monotonic()
        try:
            target, diagnostics = planner.plan(evaluator)
        except NoFeasiblePlan as error:
            report.update(status='no_feasible_plan', error=str(error))
            break
        action = env.to_action(target)
        next_obs, reward, done, truncated, info = env.step(action)
        row = dict(step=step + 1, planning_seconds=time.monotonic() - began,
                   angle=info['angle'], retained_angle=info['retained_angle'],
                   contacts=info['finger_contacts'], tilt=info['axis_tilt_radians'],
                   drop=info['drop'], success=info['success'], reward=reward,
                   planner=diagnostics, evaluator=evaluator.last_diagnostics)
        steps.append(row)
        for key, value in dict(obs=obs, actions=action, rewards=reward, next_obs=next_obs,
                               qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy()).items():
            data[key].append(value)
        obs = next_obs
        print(json.dumps(row), flush=True)
        if done or truncated:
            report['status'] = 'environment_boundary'
            break
    else:
        report['status'] = 'diagnostic_budget_exhausted'
finally:
    report['wall_seconds'] = time.monotonic() - start
    (OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(OUT / 'transitions.npz', **{k: np.asarray(v) for k, v in data.items()})
    evaluator.close()
    env.close()
