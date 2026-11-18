"""Open-loop tolerance table of a collection demonstration recorded in an overridden environment.
Usage: tolerance_collection.py <episodes_dir> <name> <period> [--shape circular] [--dirs 3]"""
import argparse, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, probe
ap = argparse.ArgumentParser(); ap.add_argument('episodes'); ap.add_argument('name'); ap.add_argument('period', type=float); ap.add_argument('--shape', default=None); ap.add_argument('--dirs', type=int, default=3); ap.add_argument('--radii', default='0,0.0003,0.001,0.003,0.01,0.03')
a = ap.parse_args(); radii = [float(x) for x in a.radii.split(',')]
if a.shape: probe.ENV_OVERRIDES['object_shape'] = a.shape
actions = np.load(Path(a.episodes) / a.name / 'transitions.npz')['actions']; n = len(actions)
print(f'{a.name}: {n} controls, shape {a.shape or "archived"}, {a.dirs} rays, hold 2 s; successes of {a.dirs} per (j, r):', flush=True)
tol = {}
for j in range(n):
    tol[j] = {str(r_): sum(probe.replay_tolerance(a.period, actions, j, r_, d) for d in range(a.dirs)) for r_ in radii}
    print(f'  j{j:2d}: ' + ' '.join(f"{r_}:{tol[j][str(r_)]}" for r_ in radii), flush=True)
json.dump(dict(name=a.name, controls=n, shape=a.shape, radii=radii, tolerance=tol), open(HERE / f'tolerance_{a.name}_{a.shape or "archived"}.json', 'w'), indent=1)
