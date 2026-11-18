"""Open-loop tolerance table of a stored probe episode. Usage: tolerance.py <episode.npz> <period> [--dirs 3] [--radii ...]"""
import argparse, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np
from probe import replay_tolerance
ap = argparse.ArgumentParser(); ap.add_argument('episode'); ap.add_argument('period', type=float); ap.add_argument('--dirs', type=int, default=3); ap.add_argument('--radii', default='0,0.0003,0.001,0.003,0.01,0.03,0.1')
a = ap.parse_args(); radii = [float(x) for x in a.radii.split(',')]; actions = np.load(a.episode)['actions']; n = len(actions)
print(f'{Path(a.episode).name}: period {a.period}, {n} controls, {a.dirs} rays per control, hold 2 s; successes of {a.dirs} per (j, r):', flush=True)
tol = {}
for j in range(n):
    tol[j] = {str(r_): sum(replay_tolerance(a.period, actions, j, r_, d) for d in range(a.dirs)) for r_ in radii}
    print(f'  j{j:2d}: ' + ' '.join(f"{r_}:{tol[j][str(r_)]}" for r_ in radii), flush=True)
json.dump(dict(episode=str(a.episode), period=a.period, controls=n, dirs=a.dirs, radii=radii, tolerance=tol), open(HERE / f'tolerance_{Path(a.episode).stem}.json', 'w'), indent=1)
