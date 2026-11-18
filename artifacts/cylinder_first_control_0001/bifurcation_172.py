"""Open-loop tolerance of the first control at tiny radii: perturb control 0 by a random direction of L2 radius r, replay the rest bit-for-bit (hold mode), count successes; report the observation deviation one control later and its largest components."""
import importlib.util, json, sys, numpy as np, torch
from pathlib import Path
import os
A = Path('/home/autel001/code/mpc_dex/artifacts'); D = A / os.environ.get('PROBE_VARIANT', 'bootstrap_variant_0099_cyl_spine172_from96'); OUT = Path(__file__).resolve().parent  # PROBE_VARIANT selects the stage whose environment, runner and sources are used (default: the cylinder spine stage)
spec = importlib.util.spec_from_file_location('r99e', D / 'run.py'); r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
torch.set_num_threads(1)
p = json.loads((D / 'protocol.json').read_text()); ec = r.EnvConfig(**json.loads((D / 'config.json').read_text())['environment'])
sources = r.load_sources(p)
def rec(action): return dict(action=np.asarray(action, np.float32), log_prob=np.nan, raw_mean=np.full(16, np.nan, np.float32), latent_std=np.full(16, np.nan, np.float32), density_is_stochastic_sample=False, learner_bc_updates=-1, learner_pretrain_updates=-1, learner_online_updates=-1)
def replay(name, j, pert):
    src, phys, n = sources[name]['source'], sources[name]['physics'], sources[name]['length']
    r.seed_all(42); env = r.environment_for_policy(ec, p['policy_interface']); ep = None
    try:
        ep = r.Episode(env, 0, src, phys, p['physical_reset_seed'], f'bif_{name}', 0, p['raw_reward_divisor']); t = 0; dev = None
        while not (ep.term or ep.trunc):
            a = pert if t == j else src['actions'][min(t, n - 1)]
            ep.advance(rec(a), policy=False); t += 1
            if t == j + 1 and t < n: dev = np.asarray(ep.obs, np.float32) - src['obs'][t]
            if t >= n + 20: break
        rep = ep.report(); return bool(rep['finger_supported_rotation_success']), t, rep['environment_failure_reason'], dev
    finally:
        if ep is not None: ep.close()
        env.close()
out = {}; tag = ("hex_" if "cyl" not in D.name else "") + "_".join(a.replace("seed_", "") for a in sys.argv[1:])
for name in sys.argv[1:]:
    src = sources[name]['source']; j = 0; out[name] = {}
    ok, t, why, dev = replay(name, j, src['actions'][j]); print(f'{name}: exact replay success {ok} ({t} controls, {why})', flush=True)
    for rad in (1e-6, 1e-5, 1e-4, 3e-4, 1e-3):
        rng = np.random.default_rng(int(rad * 1e7) + 7); res = []; devs = []
        for k in range(10):
            d = rng.standard_normal(16).astype(np.float64); d *= rad / np.linalg.norm(d)
            pert = np.clip(src['actions'][j].astype(np.float64) + d, -1, 1).astype(np.float32)
            ok, t, why, dev = replay(name, j, pert); res.append(ok); devs.append(float(np.linalg.norm(dev)) if dev is not None else float('nan'))
        out[name][str(rad)] = dict(successes=sum(res), trials=10, obs_dev_next=[round(x, 6) for x in devs])
        print(f'  r={rad:.0e}: success {sum(res)}/10; |obs deviation| after control {j}: ' + ' '.join(f'{x:.1e}' for x in devs), flush=True)
    d = rng.standard_normal(16); d *= 1e-5 / np.linalg.norm(d); _, _, _, dev = replay(name, j, np.clip(src['actions'][j] + d, -1, 1).astype(np.float32))
    idx = np.argsort(-np.abs(dev))[:6]; print('  largest obs deviations at r=1e-5 (index: value):', [(int(i), f'{dev[i]:.2e}') for i in idx])
json.dump(out, open(OUT / f"bifurcation_{tag}.json", 'w'), indent=1)
