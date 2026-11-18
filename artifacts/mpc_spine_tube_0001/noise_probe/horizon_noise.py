"""Does a longer planning horizon make the tracking planner's corrections consistent across seeds (a function of state)?
For each t in the list: plan once from the exact spine state and from one r-perturbed state with S seeds, at each horizon.
Usage: horizon_noise.py <floor> <seeds> <r> <t list comma> <horizons comma (s)> [iterations comma]"""
import json, sys
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
from scripts.probe_mpc import action_search_bounds
floor, S, rad = float(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]); ts = [int(x) for x in sys.argv[4].split(',')]; horizons = [float(x) for x in sys.argv[5].split(',')]; iters = [int(x) for x in sys.argv[6].split(',')] if len(sys.argv) > 6 else [None]
src, phys = L.load_demo('seed_193'); SA = src['actions']; n = len(SA); base = dict(L.protocol['cem'])
def plan_once(prefix, first, seed, horizon, iterations=None):
    rec = L.Recorder()
    try:
        low, high, search = action_search_bounds(rec.env, L.protocol['action_bound']); rec.replay(src, phys, prefix); t = prefix
        if first is not None: assert not rec.step(first, 'perturbed'); t += 1
        reference = np.stack([rec.env.to_joint_targets(a.astype(np.float32)) for a in SA[t:]]) if t < n else np.stack([rec.env.to_joint_targets(SA[-1])])
        with np.load(R / 'inputs/previous_recovery/prior_transitions.npz') as a_: frames = np.vstack((rec.initial['qpos'][rec.q_ids].copy(), a_['qpos'][:, rec.q_ids]))
        prior = PrimitiveBasis.from_trajectories(frames, provenance=dict(source='x', sha256='x', purpose='diag', successful_policy=False))
        cfg = CEMConfig(**dict(base, horizon_seconds=horizon, knots=int(round(horizon / base['control_seconds'])) + 1, **({'iterations': iterations} if iterations else {})))
        planner = TrackingCEM(prior, low, high, cfg, seed, floor_scale=floor); planner.execution_start = rec.env.data.ctrl.copy(); planner.reset(rec.env.data.ctrl.copy()); planner.set_reference(reference, 0)
        ev = NativeMujocoRolloutEvaluator(rec.env, timestep_seconds=base['timestep_seconds'], config=RolloutCostConfig(**L.protocol['cost']), require_native=True, cache_directory=R / 'native_cache')
        try:
            target, diag = planner.plan(ev); return rec.env.to_action(target), diag['best_cost'], diag['valid_candidates']
        except NoFeasiblePlan: return None, None, 0
        finally: ev.close()
    finally: rec.close()
rng = np.random.default_rng(777); out = {}
print('horizon | t | spine state: |a_seed - a_t| median (pure noise) | perturbed r: |mean corr| vs seed spread | ratio | valid candidates/plan (spine, perturbed)')
for t in ts:
    u = rng.standard_normal(16); u /= np.linalg.norm(u); first = np.clip(SA[t] + rad * u, -1, 1).astype(np.float32)
    for H in horizons:
      for it in iters:
        dev0, acts, v0, v1 = [], [], [], []
        for s in range(S):
            a, c, v = plan_once(t, None, 1000 + s, H, it)
            if a is not None: dev0.append(float(np.linalg.norm(a - SA[t]))); v0.append(v)
        for s in range(S):
            a, c, v = plan_once(t, first, 2000 + s, H, it)
            if a is not None: acts.append(a - SA[t + 1]); v1.append(v)
        acts = np.stack(acts) if acts else np.zeros((0, 16)); mean = acts.mean(0) if len(acts) else np.zeros(16); spread = float(np.median(np.linalg.norm(acts - mean, axis=1))) if len(acts) else float('nan'); mc = float(np.linalg.norm(mean))
        out[f'{H}_{t}_{it}'] = dict(noise=dev0, mean_corr=mc, spread=spread, valid0=v0, valid1=v1, iterations=it)
        print(f'{H:4.1f} it{it or 3:2d} | {t:2d} | {np.median(dev0) if dev0 else float("nan"):.4f} (n={len(dev0)}) | {mc:.4f} vs {spread:.4f} (n={len(acts)}) | {mc / spread if spread == spread and spread > 0 else float("nan"):.2f} | {np.mean(v0) if v0 else 0:.0f}, {np.mean(v1) if v1 else 0:.0f}', flush=True)
json.dump(out, open(Path(__file__).resolve().parent / f'horizon_noise_f{floor}_S{S}_r{rad}_it{"-".join(str(i) for i in iters)}.json', 'w'), indent=1)
