"""How much of a tracking-planner 'correction' is CEM sampling noise? From the exact spine state s_t (required correction 0) plan
once with S seeds at a given floor scale and report the spread of the returned control around the spine control a_t; then from
one perturbed state per t (r = .01, fresh direction) plan with S seeds and report the mean correction vs its seed spread.
Usage: planner_noise.py <floor_scale> <seeds> <t_from> <t_to>"""
import json, sys, time
from pathlib import Path
A = Path('/home/autel001/code/mpc_dex/artifacts'); R = A / 'mpc_prefix6_handoff_recovery_0001'; T = A / 'mpc_spine_tube_0001'
sys.path.insert(0, str(A / 'mpc_dagger_labels_0001')); sys.path.insert(0, str(T))
import numpy as np
import label as L
from tracking_cem import TrackingCEM
from mpc_dex.mpc import CEMConfig, NoFeasiblePlan
from mpc_dex.primitives import PrimitiveBasis
from mpc_dex.rollouts import RolloutCostConfig
from mpc_dex.native_rollouts import NativeMujocoRolloutEvaluator
from mpc_dex.contracts import sha256
from scripts.probe_mpc import action_search_bounds
floor, S, t0, t1 = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
src, phys = L.load_demo('seed_193'); SA = src['actions']; n = len(SA)
def plan_once(prefix, first, seed):
    rec = L.Recorder()
    try:
        low, high, search = action_search_bounds(rec.env, L.protocol['action_bound']); rec.replay(src, phys, prefix); t = prefix
        if first is not None: assert not rec.step(first, 'perturbed'); t += 1
        reference = np.stack([rec.env.to_joint_targets(a.astype(np.float32)) for a in SA[t:]]) if t < n else np.stack([rec.env.to_joint_targets(SA[-1])])
        with np.load(R / 'inputs/previous_recovery/prior_transitions.npz') as a_: frames = np.vstack((rec.initial['qpos'][rec.q_ids].copy(), a_['qpos'][:, rec.q_ids]))
        prior = PrimitiveBasis.from_trajectories(frames, provenance=dict(source='x', sha256='x', purpose='diag', successful_policy=False))
        planner = TrackingCEM(prior, low, high, CEMConfig(**L.protocol['cem']), seed, floor_scale=floor); planner.execution_start = rec.env.data.ctrl.copy(); planner.reset(rec.env.data.ctrl.copy()); planner.set_reference(reference, 0)
        ev = NativeMujocoRolloutEvaluator(rec.env, timestep_seconds=L.protocol['cem']['timestep_seconds'], config=RolloutCostConfig(**L.protocol['cost']), require_native=True, cache_directory=R / 'native_cache')
        try:
            target, diag = planner.plan(ev); return rec.env.to_action(target), diag['best_cost']
        except NoFeasiblePlan: return None, None
        finally: ev.close()
    finally: rec.close()
out = {}
print(f'floor {floor}, {S} seeds | t | spine state: |a_seed - a_t| median, max (pure noise) | perturbed state r=.01: |mean correction| vs seed spread median | (mean corr) / (spread)')
rng = np.random.default_rng(777)
for t in range(t0, t1 + 1):
    if t >= n: break
    dev0 = []
    for s in range(S):
        a, c = plan_once(t, None, 1000 + s)
        if a is not None: dev0.append(float(np.linalg.norm(a - SA[t])))
    u = rng.standard_normal(16); u /= np.linalg.norm(u); first = np.clip(SA[t] + .01 * u, -1, 1).astype(np.float32); acts = []
    if t + 1 < n:
        for s in range(S):
            a, c = plan_once(t, first, 2000 + s)
            if a is not None: acts.append(a - SA[t + 1])
    if acts:
        acts = np.stack(acts); mean = acts.mean(0); spread = np.median(np.linalg.norm(acts - mean, axis=1)); mc = float(np.linalg.norm(mean))
    else: mean, spread, mc = None, np.nan, np.nan
    out[t] = dict(noise_median=float(np.median(dev0)) if dev0 else None, noise_max=float(max(dev0)) if dev0 else None, mean_corr=mc, spread=float(spread), n0=len(dev0), n1=len(acts) if len(acts) else 0)
    print(f'{t:2d} | {np.median(dev0) if dev0 else float("nan"):.4f}, {max(dev0) if dev0 else float("nan"):.4f} (n={len(dev0)}) | {mc:.4f} vs {spread:.4f} (n={len(acts) if len(acts) else 0}) | {mc / spread if spread == spread and spread > 0 else float("nan"):.2f}', flush=True)
json.dump(out, open(Path(__file__).resolve().parent / f'noise_floor{floor}_S{S}_t{t0}_{t1}.json', 'w'), indent=1)
