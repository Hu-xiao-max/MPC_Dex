"""Read-only checkpoint/error analysis and diagnostic artifact annotation."""
from pathlib import Path
import json
import sys
D = Path(__file__).resolve().parent
sys.path.insert(0, str(D / 'source'))
import numpy as np
import torch
from mpc_dex.learning import SACAgent
from mpc_dex.contracts import sha256
from scripts.audit_policy import write_json

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
with np.load(D / 'teacher_transitions.npz', allow_pickle=False) as a:
    obs, actions, qpos, qvel = [a[k].copy() for k in ('obs', 'actions', 'qpos', 'qvel')]
report = json.loads((D / 'report.json').read_text())
analysis = dict(purpose='Quantify recorded-target distortion and closed-loop error growth, with no new training or physics trials',
               original_training_script_sha256=sha256(D / 'run.py'), conditions=[])
assert report['run_source_sha256'] == analysis['original_training_script_sha256']
assert report['conditions'][0]['initial_actor_sha256'] == report['conditions'][1]['initial_actor_sha256']
for condition in report['conditions']:
    folder = D / condition['label']
    out = dict(epsilon=condition['epsilon'], evaluations=[])
    for result in condition['results']:
        update = result['update']
        agent = SACAgent.load(folder / f'checkpoint_{update:05d}.pt', 'cuda')
        batched = agent.act(obs, deterministic=True)
        singleton = np.asarray([agent.act(o, deterministic=True) for o in obs])
        with np.load(folder / f'audit_{update:05d}' / 'transitions.npz') as a:
            n = min(len(a['actions']), len(actions))
            rows = []
            for k in range(n):
                difference = a['obs'][k] - obs[k]
                rows.append(dict(step=k + 1,
                    current_observation_max_error=float(abs(difference).max()),
                    current_joint_position_max_error=float(abs(difference[:16]).max()),
                    current_object_position_error=float(np.linalg.norm(difference[16:19])),
                    current_angle_error=float(difference[23]),
                    current_joint_velocity_max_error=float(abs(difference[25:41]).max()),
                    actual_action_max_teacher_error=float(abs(a['actions'][k] - actions[k]).max()),
                    next_joint_position_max_error=float(abs(a['qpos'][k, :16] - qpos[k, :16]).max()),
                    next_object_position_error=float(np.linalg.norm(a['qpos'][k, 16:19] - qpos[k, 16:19])),
                    next_generalized_velocity_max_error=float(abs(a['qvel'][k] - qvel[k]).max())))
        item = dict(update=update,
                    teacher_singleton_action_mse=float(np.mean((singleton - actions)**2)),
                    teacher_singleton_action_max_error=float(abs(singleton - actions).max()),
                    batch_vs_singleton_action_max_error=float(abs(batched - singleton).max()),
                    steps=rows)
        out['evaluations'].append(item)
    analysis['conditions'].append(out)
write_json(D / 'error_growth.json', analysis)

# The generic recorder used one controller label; explicitly classify the two
# teacher controls as open-loop replay, without touching numerical evidence.
for name in ['teacher_replay'] + [c['label'] + '/clipped_teacher_open_loop' for c in report['conditions']]:
    p = D / name / 'report.json'; value = json.loads(p.read_text())
    value['policy_kind'] = 'recorded-control-replay'
    value['metadata_annotation'] = 'Controller label corrected by analyze_results.py; physics and numerical results unchanged'
    write_json(p, value)
report['teacher_revalidation']['policy_kind'] = 'recorded-control-replay'
for c in report['conditions']:
    c['clipped_teacher_open_loop']['policy_kind'] = 'recorded-control-replay'
report.update(analysis_source_sha256=sha256(__file__),
              source_and_input_hashes_verified=True,
              aftermath_performed=False,
              aftermath_reason='No BC actor achieved a validated full turn at any prespecified evaluation checkpoint')
original = json.loads((D / 'inputs_manifest.json').read_text())
assert all(sha256(D / key) == expected for key, expected in original.items())
write_json(D / 'report.json', report)
manifest = {str(p.relative_to(D)): sha256(p) for p in sorted(D.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.name not in ('manifest.json', 'run.log')}
write_json(D / 'manifest.json', manifest)
print(json.dumps(analysis, indent=2))
