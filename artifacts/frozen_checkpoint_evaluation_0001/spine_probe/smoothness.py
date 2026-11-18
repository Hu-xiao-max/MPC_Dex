"""Held-out regression error vs distance to the nearest training row. Usage: smoothness.py <run_tag>"""
import sys; sys.path.insert(0, '/home/autel001/code/mpc_dex/artifacts/mpc_prefix6_handoff_recovery_0001/source')
import json, numpy as np, torch
from pathlib import Path
A = Path('/home/autel001/code/mpc_dex/artifacts'); F = A / 'feedback_bc_0001'; tag = sys.argv[1]
from mpc_dex.learning import SACAgent
torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32 = False
man = json.loads((F / 'runs' / tag / 'result.json').read_text())['manifest']; agent = SACAgent.load(F / 'runs' / tag / 'agent.pt', 'cuda')
spine = np.load(A / 'mpc_cold_demonstrations_0001/episodes/seed_193/transitions.npz'); SA = spine['actions']
def rows(items):
    O, Aa, T, first = [], [], [], []
    for x in items:
        with np.load(A / 'mpc_spine_tube_0001' / x['collection'] / 'episodes' / x['name'] / 'transitions.npz') as z:
            origin = z['control_origin']; s0 = int(np.flatnonzero((origin == 'perturbed') | (origin == 'policy'))[-1]) + 1; O.append(z['obs'][s0:]); Aa.append(z['actions'][s0:]); T += list(range(s0, len(origin))); first += [True] + [False] * (len(origin) - s0 - 1)
    return np.concatenate(O).astype(np.float32), np.concatenate(Aa).astype(np.float32), np.minimum(np.asarray(T), len(SA) - 1), np.asarray(first)
Otr, Atr, Ttr, _ = rows(man['train']); Oho, Aho, Tho, Fho = rows(man['heldout'])
Otr = np.vstack([Otr, spine['obs']]); Atr = np.vstack([Atr, SA])
with torch.no_grad(): pred = agent.actor(torch.as_tensor(Oho, device='cuda'))[0].tanh().cpu().numpy()
err = np.linalg.norm(pred - Aho, axis=1); d = np.linalg.norm(Oho[:, None, :] - Otr[None, :, :], axis=2); nn = d.argmin(1); dist = d.min(1)
nn_err = np.linalg.norm(Atr[nn] - Aho, axis=1); ref_err = np.linalg.norm(Aho - SA[Tho], axis=1)
print(f'{tag}: held-out rows {len(Aho)} | learner error median {np.median(err):.4f} | nearest-training-row label difference {np.median(nn_err):.4f} | reference-action error {np.median(ref_err):.4f} | nearest-neighbour obs distance median {np.median(dist):.3f}')
q = np.quantile(dist, [0, .25, .5, .75, 1.])
for lo, hi in zip(q[:-1], q[1:]):
    m = (dist >= lo) & (dist <= hi); print(f'  neighbour obs distance {lo:.3f}-{hi:.3f}: n={m.sum():3d} | learner error {np.median(err[m]):.4f} | nearest-row label difference {np.median(nn_err[m]):.4f} | reference-action error {np.median(ref_err[m]):.4f}')
m = Fho; print(f'  first corrected states only: n={m.sum()} | learner error {np.median(err[m]):.4f} | nearest-row label difference {np.median(nn_err[m]):.4f} | reference-action error {np.median(ref_err[m]):.4f} | neighbour distance {np.median(dist[m]):.3f}')
