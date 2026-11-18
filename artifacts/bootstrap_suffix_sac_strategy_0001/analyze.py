"""Read-only frozen-checkpoint inference and recorded-trace analysis; no physics."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

D = Path(__file__).resolve().parent
S = D.parent / 'bootstrap_suffix_sac_0001'
assert json.loads((S / 'summary.json').read_text())['status'] == 'complete'
assert not (D / 'analysis.json').exists()

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')

bindings = {}
def snapshot(relative):
    src, dst = S / relative, D / 'inputs' / relative
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    bindings[relative] = sha(src)
    assert sha(dst) == bindings[relative]
    return dst

shutil.copytree(S / 'source', D / 'source', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
for relative in ('run.py', 'protocol.json', 'summary.json', 'execution_manifest.json', 'resolved_learner.json', 'runtime.json', 'metrics.jsonl', 'heldout_calibration_before.json', 'heldout_calibration_after.json', 'warmup_calibration_before.json', 'warmup_calibration_after.json'):
    snapshot(relative)
stages = [('baseline', 'warmup_complete', 'pretraining_baseline'), ('pretrained', 'pretrained', 'pretraining_baseline'),
          ('online_500', 'online_500', 'online_500'), ('online_1000', 'online_1000', 'online_1000'), ('online_2000', 'online_2000', 'online_2000')]
for label, checkpoint, evaluation in stages:
    for name in ('agent.pt', 'agent.pt.metadata.json', 'manifest.json', 'state.json'):
        snapshot(f'checkpoints/{checkpoint}/{name}')
    snapshot(f'evaluations/{evaluation}/summary.json')
    summary = json.loads((S / f'evaluations/{evaluation}/summary.json').read_text())
    for row in summary['reports']:
        snapshot(f"evaluations/{evaluation}/{row['phase']}/transitions.npz")
write(D / 'input_bindings.json', {'root': str(S), 'files': bindings,
                                 'source_files': {str(p.relative_to(D / 'source')): sha(p) for p in sorted((D / 'source').rglob('*')) if p.is_file()}})
sys.path.insert(0, str(D / 'source'))
import numpy as np
import torch
from mpc_dex.learning import SACAgent
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.set_float32_matmul_precision('highest')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
protocol = json.loads((D / 'inputs/protocol.json').read_text())
divisor = protocol['raw_reward_divisor']

def stats(prediction, target):
    difference = prediction - target[:, None]
    return {'rows': len(target), 'mean_target_normalized': float(target.mean()),
            'critic_mean_normalized': prediction.mean(axis=0).tolist(),
            'bias_normalized': difference.mean(axis=0).tolist(),
            'rmse_normalized': np.sqrt(np.square(difference).mean(axis=0)).tolist(),
            'rmse_raw_equivalent': (divisor * np.sqrt(np.square(difference).mean(axis=0))).tolist()}

calibrations = []
for label, checkpoint, evaluation in stages:
    agent = SACAgent.load(D / f'inputs/checkpoints/{checkpoint}/agent.pt', 'cuda')
    alpha = float(agent.alpha.detach())
    evaluation_summary = json.loads((D / f'inputs/evaluations/{evaluation}/summary.json').read_text())
    by_group = {}
    for report in evaluation_summary['reports']:
        if report['deterministic']:
            continue  # Q_pi calibration requires actual sampled actions/continuation.
        with np.load(D / f"inputs/evaluations/{evaluation}/{report['phase']}/transitions.npz") as z:
            data = {k: z[k].copy() for k in z.files}
        if not report['empirical_terminal_return_available']:
            continue  # Do not impute zero time-limit tails.
        start = report['prefix_controls']
        rewards, logp = data['rewards'][start:] / divisor, data['log_prob'][start:]
        targets = rewards.copy()
        for i in range(len(targets) - 2, -1, -1):
            targets[i] += agent.config.discount * (targets[i + 1] - alpha * logp[i + 1])
        values = []
        with torch.no_grad():
            for ob, action in zip(data['obs'][start:], data['actions'][start:]):
                q = agent.critics(torch.as_tensor(ob[None], device='cuda'), torch.as_tensor(action[None], device='cuda'))[:, 0]
                values.append(q.cpu().numpy())
        q = np.asarray(values, dtype=np.float64)
        group = 'phase7' if report['phase'].startswith('phase_') else 'retention8' if report['phase'].startswith('retention8') else 'cold'
        by_group.setdefault(group, []).append((report, targets, q))
        dest = D / 'calibration_arrays' / label
        dest.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest / (report['phase'] + '.npz'), soft_return_normalized=targets,
                            q_normalized=q, actual_obs=data['obs'][start:], actual_actions=data['actions'][start:],
                            actual_log_prob=logp, actual_raw_rewards=data['rewards'][start:], alpha_normalized=alpha)
    for group, episodes in by_group.items():
        rows = np.concatenate([x[2] for x in episodes])
        targets = np.concatenate([x[1] for x in episodes])
        entry = {'checkpoint': label, 'group': group, 'alpha_normalized': alpha, 'alpha_raw_equivalent': alpha * divisor,
                 'episodes': len(episodes), 'successes': sum(x[0]['finger_supported_rotation_success'] for x in episodes),
                 'all_rows': stats(rows, targets),
                 'episode_initial': stats(np.asarray([x[2][0] for x in episodes]), np.asarray([x[1][0] for x in episodes])),
                 'per_episode': [{'name': x[0]['phase'], 'success': x[0]['finger_supported_rotation_success'],
                                  'suffix_controls':len(x[1]), 'initial_target_normalized':float(x[1][0]),
                                  'initial_q_normalized':x[2][0].tolist()} for x in episodes]}
        for success, name in ((True, 'successful_rows'), (False, 'failed_rows')):
            selected = [x for x in episodes if x[0]['finger_supported_rotation_success'] == success]
            if selected:
                entry[name] = stats(np.concatenate([x[2] for x in selected]), np.concatenate([x[1] for x in selected]))
        calibrations.append(entry)

drift = []
for name, start in (('phase_deterministic', 7), ('cold_deterministic', 0), ('retention8_deterministic', 8)):
    with np.load(D / f'inputs/evaluations/pretraining_baseline/{name}/transitions.npz') as z:
        ref = {k: z[k].copy() for k in z.files}
    m0, s0 = ref['raw_mean'][start].astype(float), ref['latent_std'][start].astype(float)
    for point in ('online_500', 'online_1000', 'online_2000'):
        path = D / f'inputs/evaluations/{point}/{name}/transitions.npz'
        if not path.exists():
            continue
        with np.load(path) as z:
            data = {k: z[k].copy() for k in z.files}
        np.testing.assert_array_equal(data['obs'][start], ref['obs'][start])
        m1, s1 = data['raw_mean'][start].astype(float), data['latent_std'][start].astype(float)
        # Exact analytic KL for the latent independent Gaussians. In exact arithmetic,
        # the same bijective tanh preserves KL; no float32-density equivalence is claimed.
        kl = .5 * np.sum((s1*s1 + (m1-m0)**2)/(s0*s0) - 1 + 2*np.log(s0/s1))
        drift.append({'checkpoint': point, 'state': name, 'observation_bitexact': True,
                      'action_max_shift':float(np.max(np.abs(data['actions'][start] - ref['actions'][start]))),
                      'mean_latent_max_shift':float(np.max(np.abs(m1-m0))),
                      'mean_shift_in_initial_std_max':float(np.max(np.abs((m1-m0)/s0))),
                      'latent_kl_new_to_initial':float(kl), 'std_min':float(s1.min()), 'std_mean':float(s1.mean()), 'std_max':float(s1.max())})

metrics = [json.loads(line) for line in (D / 'inputs/metrics.jsonl').read_text().splitlines()]
episodes = [r for r in metrics if r['event'] == 'episode_complete' and r['report']['phase'] == 'online']
bins = []
for start in range(0, len(episodes), 100):
    batch = episodes[start:start+100]
    bins.append({'episode_begin':start+1, 'episodes':len(batch),
                 'successes':sum(x['report']['finger_supported_rotation_success'] for x in batch),
                 'last_online_interaction':batch[-1]['counts']['online_policy_interactions']})
write(D / 'analysis.json', {'calibration':calibrations, 'fixed_state_actor_drift':drift, 'online_episode_bins':bins,
                           'new_physics_controls':0,'new_learning_updates':0,
                           'calibration_semantics':'Sampled frozen-policy terminal rollouts only; actual per-checkpoint alpha, first-action entropy excluded, later entropy included; all-row and episode-initial statistics separately. Single returns are noisy Q samples; five episodes do not establish calibration generally.'})
assert all(sha(S / path) == digest for path,digest in bindings.items())
print(json.dumps({'status':'complete','calibration_groups':len(calibrations),'drift_comparisons':len(drift),'online_episode_bins':bins}))
