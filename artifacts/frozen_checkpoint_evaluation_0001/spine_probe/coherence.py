"""Label coherence of a collection of branches around a spine: python coherence.py <episodes_dir> [prefix]"""
import json, sys, numpy as np
from pathlib import Path
E = Path(sys.argv[1]); prefix = sys.argv[2] if len(sys.argv) > 2 else ''
spine = np.load('/home/autel001/code/mpc_dex/artifacts/mpc_cold_demonstrations_0001/episodes/seed_193/transitions.npz'); A = spine['actions']; O = spine['obs']; n = len(A)
by = {}
for e in sorted(E.glob(prefix + '*')):
    rep = json.loads((e / 'report.json').read_text()); z = np.load(e / 'transitions.npz'); by.setdefault(rep['perturbed_control_index'], []).append((rep['perturbation_radius'], z['actions'], z['obs'], rep['mpc_controls']))
print('j | n | perturbed-state obs L2 from spine (median) | first MPC control L2 to spine a_{j+1} (median, max) | pairwise L2 among first MPC controls (median) | all MPC rows: median L2 to the time-aligned spine control | MPC controls to finish vs spine remaining')
for j in sorted(by):
    rows = by[j]
    if j + 1 >= n: continue
    first = np.stack([acts[j + 1] for _, acts, _, _ in rows]); dobs = [np.linalg.norm(obs[j + 1] - O[j + 1]) for _, _, obs, _ in rows]
    d_spine = np.linalg.norm(first - A[j + 1], axis=1); iu = np.triu_indices(len(rows), 1); pw = np.linalg.norm(first[:, None] - first[None], axis=-1)[iu] if len(rows) > 1 else np.array([np.nan])
    aligned = [np.linalg.norm(acts[i] - A[i]) for _, acts, _, _ in rows for i in range(j + 1, min(len(acts), n))]
    print(f'{j:2d} | {len(rows):2d} | {np.median(dobs):.3f} | {np.median(d_spine):.3f}, {d_spine.max():.3f} | {np.nanmedian(pw):.3f} | {np.median(aligned):.3f} | {int(np.median([m for *_, m in rows]))} vs {n - j - 1}')
