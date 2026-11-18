"""Open-loop divergence rate: perturb control j of a stored episode by r along 3 fixed rays, replay the rest, and record the
observation distance from the unperturbed replay after each control (first control at which it exceeds 0.1 and 1.0).
Usage: divergence.py <episode.npz> <period> <j> <r>"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np
from probe import make_env
import probe
ep, P, j, r = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]); actions = np.load(ep)['actions']; n = len(actions)
if len(sys.argv) > 5: probe.ENV_OVERRIDES['object_shape'] = sys.argv[5]
if len(sys.argv) > 6: probe.ENV_OVERRIDES['object_diameter'] = float(sys.argv[6])
def rollout(acts):
    env = make_env(P); obs, _ = env.reset(seed=42); out = [obs.copy()]
    try:
        for a in acts:
            obs, _, term, trunc, _ = env.step(a); out.append(obs.copy())
            if term or trunc: break
    finally: env.close()
    return np.stack(out)
ref = rollout(actions)
print(f'{Path(ep).name}: period {P}, {n} controls; perturb control {j} by {r}; obs distance from the unperturbed replay after each control:')
for d in range(3):
    rng = np.random.default_rng(np.random.SeedSequence([90000, int(round(P * 1000)), j, d])); u = rng.standard_normal(16); u /= np.linalg.norm(u)
    acts = actions.copy(); acts[j] = np.clip(actions[j] + r * u, -1, 1).astype(np.float32); pert = rollout(acts); m = min(len(ref), len(pert))
    dist = np.linalg.norm(pert[:m] - ref[:m], axis=1); first01 = next((i for i, x in enumerate(dist) if x > 0.1), None); first1 = next((i for i, x in enumerate(dist) if x > 1.0), None)
    print(f'  ray {d}: ' + ' '.join(f'{x:.3f}' for x in dist[j:j + 12]) + f'  | >0.1 after {None if first01 is None else first01 - j} controls ({None if first01 is None else round((first01 - j) * P, 2)} s), >1.0 after {None if first1 is None else first1 - j} controls', flush=True)
