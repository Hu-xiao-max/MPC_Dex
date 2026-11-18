"""Deviation at the spine state and seed spread at a perturbed state for the tracking planner WITH a tracking penalty of weight w.
Usage: tracking_cost_noise.py <floor> <seeds> <r> <t list> <horizon> <weights comma> [iterations comma]"""
import json, sys
from pathlib import Path
A = Path('/home/autel001/code/mpc_dex/artifacts'); R = A / 'mpc_prefix6_handoff_recovery_0001'; T = A / 'mpc_spine_tube_0001'
sys.path.insert(0, str(A / 'mpc_dagger_labels_0001')); sys.path.insert(0, str(T))
import numpy as np
import label as L
from tracking_cem import TrackingCEM, TrackingPenaltyEvaluator
from mpc_dex.mpc import CEMConfig, NoFeasiblePlan
from mpc_dex.primitives import PrimitiveBasis
from mpc_dex.rollouts import RolloutCostConfig
from mpc_dex.native_rollouts import NativeMujocoRolloutEvaluator
from scripts.probe_mpc import action_search_bounds
floor, S, rad = float(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]); ts = [int(x) for x in sys.argv[4].split(',')]; H = float(sys.argv[5]); weights = [float(x) for x in sys.argv[6].split(',')]; iters = [int(x) for x in sys.argv[7].split(',')] if len(sys.argv) > 7 else [None]
src, phys = L.load_demo('seed_193'); SA = src['actions']; n = len(SA); base = dict(L.protocol['cem'])
def plan_once(prefix, first, seed, w, it=None):
    rec = L.Recorder()
    try:
        low, high, search = action_search_bounds(rec.env, L.protocol['action_bound']); rec.replay(src, phys, prefix); t = prefix
        if first is not None: assert not rec.step(first, 'perturbed'); t += 1
        reference = np.stack([rec.env.to_joint_targets(a.astype(np.float32)) for a in SA[t:]]) if t < n else np.stack([rec.env.to_joint_targets(SA[-1])])
        with np.load(R / 'inputs/previous_recovery/prior_transitions.npz') as a_: frames = np.vstack((rec.initial['qpos'][rec.q_ids].copy(), a_['qpos'][:, rec.q_ids]))
        prior = PrimitiveBasis.from_trajectories(frames, provenance=dict(source='x', sha256='x', purpose='diag', successful_policy=False))
        cfg = CEMConfig(**dict(base, horizon_seconds=H, knots=int(round(H / base['control_seconds'])) + 1, **({'iterations': it} if it else {})))
        planner = TrackingCEM(prior, low, high, cfg, seed, floor_scale=floor); planner.execution_start = rec.env.data.ctrl.copy(); planner.reset(rec.env.data.ctrl.copy()); planner.set_reference(reference, 0)
        ev = TrackingPenaltyEvaluator(NativeMujocoRolloutEvaluator(rec.env, timestep_seconds=base['timestep_seconds'], config=RolloutCostConfig(**L.protocol['cost']), require_native=True, cache_directory=R / 'native_cache'), planner, w)
        try:
            target, diag = planner.plan(ev); return rec.env.to_action(target), diag['best_cost'], diag['valid_candidates']
        except NoFeasiblePlan: return None, None, 0
        finally: ev.close()
    finally: rec.close()
rng = np.random.default_rng(777); out = {}
print(f'horizon {H}, floor {floor}, {S} seeds | w | t | spine: dev median (best cost) | perturbed r={rad}: |mean corr| vs spread | ratio | valid (spine, pert)')
for t in ts:
    u = rng.standard_normal(16); u /= np.linalg.norm(u); first = np.clip(SA[t] + rad * u, -1, 1).astype(np.float32)
    for w in weights:
      for it in iters:
        dev0, c0, acts, v0, v1 = [], [], [], [], []
        for s in range(S):
            a, c, v = plan_once(t, None, 1000 + s, w, it)
            if a is not None: dev0.append(float(np.linalg.norm(a - SA[t]))); c0.append(c); v0.append(v)
        for s in range(S):
            a, c, v = plan_once(t, first, 2000 + s, w, it)
            if a is not None: acts.append(a - SA[t + 1]); v1.append(v)
        acts = np.stack(acts) if len(acts) else np.zeros((0, 16)); mean = acts.mean(0) if len(acts) else np.zeros(16); spread = float(np.median(np.linalg.norm(acts - mean, axis=1))) if len(acts) else float('nan'); mc = float(np.linalg.norm(mean))
        out[f'{w}_{t}_{it}'] = dict(noise=dev0, best_cost=c0, mean_corr=mc, spread=spread, valid0=v0, valid1=v1, iterations=it)
        print(f'{w:8g} it{it or 3:2d} | {t:2d} | {np.median(dev0) if dev0 else float("nan"):.4f} ({np.median(c0) if c0 else float("nan"):.3g}) | {mc:.4f} vs {spread:.4f} (n={len(acts)}) | {mc / spread if spread == spread and spread > 0 else float("nan"):.2f} | {np.mean(v0) if v0 else 0:.0f}, {np.mean(v1) if v1 else 0:.0f}', flush=True)
json.dump(out, open(Path(__file__).resolve().parent / f'tracking_cost_noise_H{H}_f{floor}_S{S}_r{rad}_t{"-".join(str(x) for x in ts)}_it{"-".join(str(i) for i in iters)}.json', 'w'), indent=1)
