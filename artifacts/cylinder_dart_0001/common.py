"""Shared pieces for the cylinder DART/DAgger work: the archived cylinder collector environment and planner (identical to cylinder_demonstrations_0001/collect_used.py), the stage runner's frozen-policy inference, and the strict outcome."""
import importlib.util, json, sys, time
from pathlib import Path
A = Path('/home/autel001/code/mpc_dex/artifacts'); CP = A / 'control_period_0001'; R = A / 'mpc_prefix6_handoff_recovery_0001'
for p in (str(CP), str(R), '/home/autel001/code/mpc_dex'): sys.path.insert(0, p)
import numpy as np, probe
probe.ENV_OVERRIDES.update(object_shape='circular')
from probe import make_env, strict_outcome, protocol, FIELDS
from mpc_dex.mpc import CEMConfig, NoFeasiblePlan
from mpc_dex.primitives import PrimitiveBasis
from mpc_dex.rollouts import RolloutCostConfig
from mpc_dex.native_rollouts import NativeMujocoRolloutEvaluator
from mpc_dex.contracts import sha256
from scripts.probe_mpc import action_search_bounds
from scripts.audit_policy import PhysicsObserver, PHYSICS_COLUMNS
from recovery_recipe import ProjectedReferenceCEM, integration_state
PERIOD = 0.1; PRIOR = R / 'inputs/previous_recovery/prior_transitions.npz'


import os
HORIZON_CONTROLS = int(os.environ.get('DART_HORIZON_CONTROLS', '5'))  # planning horizon in controls (archived recipe: 5 = 0.5 s)


def planner_for(env, seed, cache_dir):
    """Fresh archived planner + native evaluator for the current env state (warm start = current ctrl)."""
    low, high, search = action_search_bounds(env, protocol['action_bound'])
    with np.load(PRIOR) as a: frames = np.vstack((env.data.qpos[env.q_ids].copy(), a['qpos'][:, env.q_ids]))
    prior = PrimitiveBasis.from_trajectories(frames, provenance=dict(source=str(PRIOR), sha256=sha256(PRIOR), purpose='measured 40 mm diagnostic PCA prior; not a SAC primitive', successful_policy=False))
    cem = CEMConfig(**dict(protocol['cem'], control_seconds=PERIOD, timestep_seconds=PERIOD, horizon_seconds=HORIZON_CONTROLS * PERIOD, knots=6))
    planner = ProjectedReferenceCEM(prior, low, high, cem, seed); planner.execution_start = env.data.ctrl.copy(); planner.reset(env.data.ctrl.copy())
    evaluator = NativeMujocoRolloutEvaluator(env, timestep_seconds=PERIOD, config=RolloutCostConfig(**protocol['cost']), require_native=True, cache_directory=cache_dir)
    return planner, evaluator, search, cem


def plan_once(planner, evaluator, env):
    """One planner call at the env's current state; returns the planned action (env action space) or None if no feasible plan."""
    planner.execution_start = env.data.ctrl.copy()
    try: target, diag = planner.plan(evaluator)
    except NoFeasiblePlan: return None, None
    return env.to_action(target), diag


def load_policy(variant, checkpoint):
    """A stage's frozen actor loaded with the repository's SACAgent (the stage runner cannot be imported next to the planner modules);
    inference contract as the stage evaluations and mpc_dagger_labels_0001/label.py: CUDA float32 singleton, mean.tanh()."""
    import torch
    from mpc_dex.learning import SACAgent
    torch.set_num_threads(1); torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    agent = SACAgent.load(A / variant / 'checkpoints' / checkpoint / 'agent.pt', 'cuda')
    @torch.no_grad()
    def act(obs, det=True):
        mean, log_std = agent.actor(torch.as_tensor(np.asarray(obs, np.float32)[None], device=agent.device))
        return (mean.tanh() if det else agent.actor.sample(torch.as_tensor(np.asarray(obs, np.float32)[None], device=agent.device))[0])[0].cpu().numpy().astype(np.float32)
    return agent, act, None
