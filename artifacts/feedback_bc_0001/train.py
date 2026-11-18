"""Supervised feedback learner around the spine: a (possibly larger) squashed-Gaussian actor in the archived SACAgent format,
trained by mean-only regression (atanh space, the archived bc_step) on the spine rows and the MPC correction rows of the tube
collections, then evaluated closed-loop from the cold reset, every spine prefix, and corrected states of training and held-out
branches. No critic, no RL, no primitive eligibility; the question is only whether the MPC's feedback law is representable.

Usage: train.py --tag T --hidden 1024,1024,1024 --steps 100000 [--collections tracking,tracking2] [--holdout-direction 3] [--lr 3e-4] [--spine-weight 10] [--std 0.01]
"""
import argparse, importlib.util, json, math, re, time
from pathlib import Path
A = Path('/home/autel001/code/mpc_dex/artifacts'); HERE = Path(__file__).resolve().parent; STAGE = A / 'bootstrap_variant_0087_ms_spine193_tube_from84'
spec = importlib.util.spec_from_file_location('fb_runner', STAGE / 'run.py'); r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
np, torch = r.np, r.torch
torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
from mpc_dex.learning import SACAgent, LearnerConfig
from mpc_dex.contracts import sha256
protocol = json.loads((STAGE / 'protocol.json').read_text()); ec = r.EnvConfig(**json.loads((STAGE / 'config.json').read_text())['environment'])
SPINE = A / 'mpc_cold_demonstrations_0001/episodes/seed_193'


def load_episode(d):
    with np.load(d / 'transitions.npz', allow_pickle=False) as z: src = {k: z[k].copy() for k in z.files}
    with np.load(d / 'physics.npz', allow_pickle=False) as z: phys = z['samples'].copy()
    rep = json.loads((d / 'report.json').read_text()); assert rep['strict_force_success'] and rep['stored'] and sha256(d / 'transitions.npz') == rep['transitions_sha256']
    return src, phys, rep


def root_direction(rep, episodes_dir):
    """Perturbation direction of the first-order ancestor (second-order children inherit their parent's); DAgger branches (-1) are never held out."""
    d = rep.get('perturbation_direction', -1)
    if rep.get('order', 1) == 1 or not rep.get('parent'): return d
    m = re.search(r'_d(\d+)$', rep['parent'])
    if m: return int(m.group(1))
    parent_report = Path(episodes_dir).parent.parent / 'tracking' / 'episodes' / rep['parent'] / 'report.json'
    return root_direction(json.loads(parent_report.read_text()), parent_report.parent.parent) if parent_report.exists() else d


def correction_rows(src):
    """Rows after the last non-target control (perturbed/policy): the MPC correction segment."""
    origin = src['control_origin']; last = int(np.flatnonzero((origin == 'perturbed') | (origin == 'policy'))[-1])
    assert (origin[last + 1:] == 'mpc').all() and last + 1 < len(origin)
    return last + 1


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--hidden', default='1024,1024,1024'); ap.add_argument('--steps', type=int, default=100000)
    ap.add_argument('--collections', default='tracking'); ap.add_argument('--holdout-direction', type=int, default=3); ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--spine-weight', type=int, default=10)
    ap.add_argument('--std', type=float, default=.01); ap.add_argument('--batch', type=int, default=512); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--eval-branches', type=int, default=24); ap.add_argument('--originals', action='store_true', help='also regress the 62 other original demonstrations (their BC rows as the 0087 stage defines them, shared-prefix rows included)')
    ap.add_argument('--manifest', default=None, help='reuse the episode split of an earlier run (its result.json) instead of the live directory listing')
    ap.add_argument('--noise', type=float, default=0., help='Gaussian input noise (obs units) during training, for smoothness'); ap.add_argument('--weight-decay', type=float, default=0.); ap.add_argument('--early-stop', type=int, default=0, help='evaluate held-out regression every this many steps and keep the best weights')
    a = ap.parse_args(); out = HERE / 'runs' / a.tag; out.mkdir(parents=True, exist_ok=True)
    # ---- data (frozen split: names and hashes are recorded so runs started at different times remain comparable)
    spine_src, spine_phys, _ = load_episode(SPINE); n_spine = len(spine_src['actions'])
    train_eps, hold_eps = [], []
    if a.manifest:
        man = json.loads(Path(a.manifest).read_text())['manifest']
        train_eps = [A / 'mpc_spine_tube_0001' / x['collection'] / 'episodes' / x['name'] for x in man['train']]; hold_eps = [A / 'mpc_spine_tube_0001' / x['collection'] / 'episodes' / x['name'] for x in man['heldout']]
    else:
        for coll in a.collections.split(','):
            E = A / 'mpc_spine_tube_0001' / coll / 'episodes'
            for d in sorted(E.iterdir()):
                if not (d / 'report.json').exists(): continue
                rep = json.loads((d / 'report.json').read_text())
                if not rep.get('mpc_controls', 0): continue
                (hold_eps if root_direction(rep, E) == a.holdout_direction else train_eps).append(d)
    manifest = dict(train=[dict(collection=d.parent.parent.name, name=d.name, sha256=json.loads((d / 'report.json').read_text())['transitions_sha256']) for d in train_eps],
                    heldout=[dict(collection=d.parent.parent.name, name=d.name, sha256=json.loads((d / 'report.json').read_text())['transitions_sha256']) for d in hold_eps])
    (out / 'manifest.json').write_text(json.dumps(dict(manifest=manifest), indent=1) + '\n')  # written before training so parallel runs can share the split
    obs, act, tag_rows = [spine_src['obs']] * a.spine_weight, [spine_src['actions']] * a.spine_weight, ['spine'] * (a.spine_weight * n_spine)
    for d in train_eps:
        src, _, _ = load_episode(d); s0 = correction_rows(src); obs.append(src['obs'][s0:]); act.append(src['actions'][s0:]); tag_rows += [d.name] * (len(src['actions']) - s0)
    if a.originals:
        stage_sources = r.load_sources(protocol); demos = r.multisource_demonstrations(protocol, stage_sources)
        for item in demos:
            if item['name'].startswith('track_') or item['name'] == 'seed_193': continue
            obs.append(item['arrays']['obs'][item['bc_rows']]); act.append(item['arrays']['actions'][item['bc_rows']]); tag_rows += [item['name']] * len(item['bc_rows'])
    obs = np.concatenate(obs).astype(np.float32); act = np.concatenate(act).astype(np.float32); tag_rows = np.asarray(tag_rows)
    distinct = np.sort(np.unique(obs, axis=0, return_index=True)[1]); uniq = len(distinct)
    hobs, hact, htag = [], [], []
    for d in hold_eps:
        src, _, _ = load_episode(d); s0 = correction_rows(src); hobs.append(src['obs'][s0:]); hact.append(src['actions'][s0:]); htag += [d.name] * (len(src['actions']) - s0)
    hobs = np.concatenate(hobs).astype(np.float32) if hobs else np.zeros((0, 60), np.float32); hact = np.concatenate(hact).astype(np.float32) if hact else np.zeros((0, 16), np.float32)
    print(f'rows {len(obs)} (distinct observations {uniq}; spine x{a.spine_weight}; training branches {len(train_eps)}, held-out branches {len(hold_eps)} with {len(hobs)} correction rows)', flush=True)
    # ---- model
    torch.manual_seed(a.seed); hidden = tuple(int(x) for x in a.hidden.split(','))
    agent = SACAgent(60, 16, LearnerConfig(hidden_sizes=hidden), 'cuda')
    for g in agent.bc_optimizer.param_groups: g['lr'] = a.lr; g['weight_decay'] = a.weight_decay
    rng = np.random.default_rng(a.seed); t0 = time.monotonic(); log = []; best = (float('inf'), None, -1)
    def heldout_l2():
        if not len(hobs): return float('nan')
        with torch.no_grad(): m, _ = agent.actor(torch.as_tensor(hobs, device='cuda')); return float(np.median(np.linalg.norm(m.tanh().cpu().numpy() - hact, axis=1)))
    for i in range(a.steps):
        if i == int(.6 * a.steps):
            for g in agent.bc_optimizer.param_groups: g['lr'] = a.lr / 3
        idx = rng.integers(len(obs), size=a.batch); o = obs[idx] + (rng.standard_normal((len(idx), 60)) * a.noise).astype(np.float32) if a.noise > 0 else obs[idx]
        res = agent.bc_step(dict(obs=o, actions=act[idx]))
        if a.early_stop and (i % a.early_stop == 0 or i == a.steps - 1):
            h = heldout_l2()
            if h < best[0]: best = (h, {k: v.detach().clone() for k, v in agent.actor.state_dict().items()}, i)
        if i % 5000 == 0 or i == a.steps - 1: log.append((i, res['bc_loss'])); print(f'step {i} bc_loss {res["bc_loss"]:.5f}' + (f' heldout {heldout_l2():.4f}' if a.early_stop else ''), flush=True)
    if a.early_stop and best[1] is not None:
        agent.actor.load_state_dict(best[1]); print(f'early stop: best held-out L2 {best[0]:.4f} at step {best[2]}', flush=True)
    agent.initialize_std()
    with torch.no_grad(): agent.actor.log_std_head.bias.fill_(math.log(a.std))  # constant spread for the stochastic trials
    eps_ = agent.config.bc_action_epsilon
    def errors(o, ac):
        with torch.no_grad():
            m, _ = agent.actor(torch.as_tensor(o, device='cuda')); pred = m.tanh().cpu().numpy(); tgt = np.arctanh(np.clip(ac, -1 + eps_, 1 - eps_))
            return np.linalg.norm(pred - ac, axis=1), ((m.cpu().numpy() - tgt) ** 2).mean(axis=1)
    fit, pre = errors(obs, act); fit_d, pre_d = fit[distinct], pre[distinct]
    spine_fit = fit[:n_spine].round(3).tolist(); worst = int(np.argmax(fit_d))
    print(f'training fit over {uniq} distinct rows: action-space L2 median {np.median(fit_d):.4f} p90 {np.quantile(fit_d, .9):.4f} max {fit_d.max():.3f} (row {tag_rows[distinct][worst]}); atanh-space MSE {pre_d.mean():.5f} | spine rows {spine_fit}', flush=True)
    hfit, hpre = errors(hobs, hact) if len(hobs) else (np.zeros(0), np.zeros(0))
    if len(hobs): print(f'HELD-OUT regression (chaos-free generalisation): {len(hobs)} rows, action-space L2 median {np.median(hfit):.4f} p90 {np.quantile(hfit, .9):.4f} max {hfit.max():.3f}; atanh-space MSE {hpre.mean():.5f}', flush=True)
    agent.save(out / 'agent.pt')
    # ---- closed loop
    def run(prefix, src, phys, det, seed):
        r.seed_all(seed); env = r.environment_for_policy(ec, protocol['policy_interface']); ep = None
        try:
            ep = r.Episode(env, prefix, src, phys, protocol['physical_reset_seed'], 'feedback_bc', seed, protocol['raw_reward_divisor'])
            while not (ep.term or ep.trunc): ep.advance(r.command(agent, ep.obs, det))
            rep = ep.report(); return dict(ok=bool(rep['finger_supported_rotation_success']), controls=rep['policy_suffix_controls'], angle=round(rep['final_force_qualified_angle'], 2), fail=rep['environment_failure_reason'])
        finally:
            if ep is not None: ep.close()
            env.close()
    res = dict(tag=a.tag, hidden=hidden, steps=a.steps, collections=a.collections, holdout_direction=a.holdout_direction, rows=int(len(obs)), distinct_obs=int(uniq), training_branches=len(train_eps), heldout_branches=len(hold_eps),
               fit_distinct=dict(l2_median=float(np.median(fit_d)), l2_p90=float(np.quantile(fit_d, .9)), l2_max=float(fit_d.max()), worst_row=str(tag_rows[distinct][worst]), atanh_mse=float(pre_d.mean())),
               heldout_regression=(dict(rows=int(len(hobs)), l2_median=float(np.median(hfit)), l2_p90=float(np.quantile(hfit, .9)), l2_max=float(hfit.max()), atanh_mse=float(hpre.mean())) if len(hobs) else None),
               spine_fit=spine_fit, loss=log, manifest=manifest)
    res['cold'] = dict(det=run(0, spine_src, spine_phys, True, 42), stoch=[run(0, spine_src, spine_phys, False, 500 + i)['ok'] for i in range(5)])
    res['spine_det'] = [run(k, spine_src, spine_phys, True, 42)['ok'] for k in range(1, n_spine)]
    res['spine_stoch3'] = [sum(run(k, spine_src, spine_phys, False, 600 + 10 * k + i)['ok'] for i in range(3)) for k in range(1, n_spine)]
    print('cold', res['cold'], '\nspine det k1..', res['spine_det'], '\nspine stoch3', res['spine_stoch3'], flush=True)
    def branch_test(eps, label):
        pick = [eps[i] for i in np.random.default_rng(1).choice(len(eps), size=min(a.eval_branches, len(eps)), replace=False)] if eps else []
        rows = []
        for d in pick:
            src, phys, rep = load_episode(d); s0 = correction_rows(src); o = run(s0, src, phys, True, 42); rows.append(dict(name=d.name, j=rep['perturbed_control_index'], order=rep.get('order', 1), ok=o['ok'], controls=o['controls'], remaining=len(src['actions']) - s0, fail=o['fail']))
        early = [x for x in rows if x['j'] <= 9]
        print(f'{label}: corrected-state handoffs (deterministic) {sum(x["ok"] for x in rows)}/{len(rows)}; informative cells j<=9: {sum(x["ok"] for x in early)}/{len(early)}; by (order, j)', {f"o{o_}j{j}": f"{sum(x['ok'] for x in rows if x['j'] == j and x['order'] == o_)}/{sum(1 for x in rows if x['j'] == j and x['order'] == o_)}" for o_ in sorted({x['order'] for x in rows}) for j in sorted({x['j'] for x in rows if x['order'] == o_})}, flush=True)
        return rows
    res['train_branches'] = branch_test(train_eps, 'training branches'); res['heldout_branches'] = branch_test(hold_eps, 'held-out branches')
    res['wall'] = time.monotonic() - t0; (out / 'result.json').write_text(json.dumps(res, indent=1) + '\n'); print('wall', round(res['wall'], 1))


if __name__ == '__main__':
    main()
