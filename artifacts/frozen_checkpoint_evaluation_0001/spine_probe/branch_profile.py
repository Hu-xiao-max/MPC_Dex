"""Per-step deviation profile of tube branches: for each branch, |a_t - spine_a_t| and |obs_t - spine_obs_t| along the correction segment.
Usage: profile.py <episodes_dir> [prefix]"""
import json, sys, numpy as np
from pathlib import Path
E = Path(sys.argv[1]); prefix = sys.argv[2] if len(sys.argv) > 2 else ''
sp = np.load('/home/autel001/code/mpc_dex/artifacts/mpc_cold_demonstrations_0001/episodes/seed_193/transitions.npz'); SA, SO = sp['actions'], sp['obs']; n = len(SA)
rows = []
for d in sorted(E.glob(prefix + '*')):
    if not (d / 'report.json').exists(): continue
    rep = json.loads((d / 'report.json').read_text()); z = np.load(d / 'transitions.npz'); j = rep['perturbed_control_index']; acts, obs = z['actions'], z['obs']
    da = [np.linalg.norm(acts[t] - SA[t]) for t in range(j + 1, min(len(acts), n))]; do = [np.linalg.norm(obs[t] - SO[t]) for t in range(j + 1, min(len(obs), n))]
    rows.append((d.name.replace('track_seed_193_', ''), j, rep['perturbation_radius'], len(acts) - j - 1, da, do))
print('branch | j | r | mpc steps | per-step |a - spine_a| (first 8) | per-step |obs - spine_obs| (first 8)')
for name, j, r, m, da, do in rows:
    print(f'{name:22s} | {j:2d} | {r:.3f} | {m:2d} | ' + ' '.join(f'{x:.3f}' for x in da[:8]) + ' | ' + ' '.join(f'{x:.2f}' for x in do[:8]))
# at which step after the perturbation does the correction exceed 0.01?
late = [next((i for i, x in enumerate(da) if x > 0.01), None) for *_, da, _ in rows]
print('first correction step > 0.01 after the perturbation (None = never):', late)
