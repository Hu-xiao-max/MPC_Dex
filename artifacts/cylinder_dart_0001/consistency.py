"""Label consistency across DART rollouts of the same planner seed: at each step t, pairwise L2 between the labels of the rollouts and
between their observations (and the hand-joint part of the observation), plus per-rollout outcome. Usage: consistency.py <glob of rollouts/*.npz>"""
import glob, json, sys
import numpy as np
files = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else 'rollouts/p302_sigma0.01_n*.npz'))
R = [(f, dict(np.load(f)), json.load(open(f[:-4] + '.json'))) for f in files]
for f, z, rep in R: print(f.split('/')[-1], rep['status'], rep['controls'], 'strict', rep['strict_force_success'], 'angle', round(rep['final_force_qualified_angle'], 2))
T = min(len(z['labels']) for _, z, _ in R)
print('t | pairwise label L2 (median, max) | pairwise obs L2 (median, max) | pairwise last-action L2 | label-vs-previous-step-label L2 (median over rollouts)')
for t in range(T):
    L = [z['labels'][t] for _, z, _ in R]; O = [z['obs'][t] for _, z, _ in R]
    dl = [np.linalg.norm(L[i] - L[j]) for i in range(len(L)) for j in range(i + 1, len(L))]
    do = [np.linalg.norm(O[i] - O[j]) for i in range(len(O)) for j in range(i + 1, len(O))]
    da = [np.linalg.norm(O[i][-16:] - O[j][-16:]) for i in range(len(O)) for j in range(i + 1, len(O))]
    dprev = [np.linalg.norm(z['labels'][t] - z['labels'][t - 1]) for _, z, _ in R] if t > 0 else [0.]
    print(f'{t:2d} | {np.median(dl):.3f} {np.max(dl):.3f} | {np.median(do):.3f} {np.max(do):.3f} | {np.median(da):.3f} | {np.median(dprev):.3f}')
