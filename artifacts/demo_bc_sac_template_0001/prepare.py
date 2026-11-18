"""Stamp a frozen demonstration-initialized SAC stage from this template.

Usage: prepare.py --name <stage_name> --demos <collection_dir> [--min-demos 20] [--bc 12000] [--pretrain 2000]
       [--online 4000] [--checkpoints 500,1000,...] [--offline-fraction .3] [--eval-trials 10] [--seed 30001]
Creates artifacts/<stage_name>/ with the frozen numerical source (from the recovery stage snapshot), config,
the SAC0001 learner config, hash-bound demonstration episodes, protocol.json and execution_manifest.json.
"""
from pathlib import Path
import argparse, hashlib, json, shutil
T = Path(__file__).resolve().parent; A = T.parent
R = A / 'mpc_prefix6_handoff_recovery_0001'; L = A / 'bootstrap_suffix_sac_0001'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, v): Path(p).write_text(json.dumps(v, indent=2, allow_nan=False) + '\n')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True); ap.add_argument('--period', type=float, default=None, help='control period in seconds for a route (b) stage (default: the archived 0.1 s); every demonstration must carry report["period"] equal to it'); ap.add_argument('--demos', required=True); ap.add_argument('--min-demos', type=int, default=20)
    ap.add_argument('--bc', type=int, default=12000); ap.add_argument('--pretrain', type=int, default=2000); ap.add_argument('--online', type=int, default=4000)
    ap.add_argument('--checkpoints', default='500,1000,1500,2000,2500,3000,3500,4000'); ap.add_argument('--offline-fraction', type=float, default=.3)
    ap.add_argument('--eval-trials', type=int, default=10); ap.add_argument('--seed', type=int, default=30001); ap.add_argument('--learning-starts', type=int, default=256)
    ap.add_argument('--max-demos', type=int, default=10**6)
    a = ap.parse_args()
    D = A / a.name
    if D.exists(): raise SystemExit(f'{D} exists; stages are never overwritten')
    C = A / a.demos
    episodes = sorted(p.name for p in (C / 'episodes').iterdir() if (p / 'report.json').exists())[:a.max_demos]
    assert len(episodes) >= a.min_demos, f'only {len(episodes)} verified demonstrations'
    D.mkdir(); bindings = {}
    rec_manifest = json.loads((R / 'execution_manifest.json').read_text())
    sources = {k: v for k, v in rec_manifest.items() if k.startswith('source/')}
    assert sources and all(sha(R / k) == v for k, v in sources.items())
    shutil.copytree(R / 'source', D / 'source', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    assert {str(p.relative_to(D)) for p in (D / 'source').rglob('*') if p.is_file()} == set(sources)
    bindings['source/'] = dict(source=str(R / 'source'), files=len(sources), execution_manifest_sha256=sha(R / 'execution_manifest.json'))
    for name in ('run.py', 'test_protocol.py'):
        if (T / name).exists(): shutil.copyfile(T / name, D / name); bindings[name] = dict(source=str(T / name), sha256=sha(T / name))
    shutil.copyfile(T / 'prepare.py', D / 'prepare_used.py'); bindings['prepare_used.py'] = dict(source=str(T / 'prepare.py'), sha256=sha(T / 'prepare.py'))
    shutil.copyfile(R / 'initial_grasp.npz' if (R / 'initial_grasp.npz').exists() else R / 'inputs/prefix_stage/initial_grasp.npz', D / 'initial_grasp.npz')
    config = json.loads((R / 'config.json').read_text()); config['environment']['initial_grasp'] = str(D / 'initial_grasp.npz')
    if a.period is not None:
        config['environment']['control_seconds'] = float(a.period)  # route (b): the one deviation from the archived environment; horizon_steps stays a count of control decisions
    write(D / 'config.json', config)
    bindings['config.json'] = dict(source=str(R / 'config.json'), sha256=sha(R / 'config.json'), initial_grasp_rebound=True, control_seconds_override=a.period)
    meta = json.loads((L / 'initial.pt.metadata.json').read_text()); write(D / 'learner_config.json', meta['learner_config'])
    bindings['learner_config.json'] = dict(source=str(L / 'initial.pt.metadata.json'), sha256=sha(L / 'initial.pt.metadata.json'), note='SAC0001 LearnerConfig (architecture, learning rates, initial std .05, discount, tau); networks are fresh')
    (D / 'demonstrations').mkdir()
    for name in episodes:
        rep_ = json.loads((C / 'episodes' / name / 'report.json').read_text())
        assert abs(float(rep_.get('period', 0.1)) - float(a.period if a.period is not None else 0.1)) < 1e-9, f'{name}: demonstration period {rep_.get("period", 0.1)} differs from the stage period'
        shutil.copytree(C / 'episodes' / name, D / 'demonstrations' / name)
        for f in ('transitions.npz', 'physics.npz', 'report.json'):
            bindings[f'demonstrations/{name}/{f}'] = dict(source=str(C / 'episodes' / name / f), sha256=sha(C / 'episodes' / name / f))
    for f in sorted(C.glob('attempts_*.jsonl')) + [C / 'collect.py', C / 'README.md', C.parent / 'collect.py', C.parent / 'probe.py', C.parent / 'findings.md']:
        if f.exists(): shutil.copyfile(f, D / ('inputs_collection_' + f.name)); bindings['inputs_collection_' + f.name] = dict(source=str(f), sha256=sha(f))
    write(D / 'input_bindings.json', bindings)
    checkpoints = [int(x) for x in a.checkpoints.split(',')]; assert checkpoints[-1] == a.online
    write(D / 'protocol.json', dict(schema_version=1, kind='demonstration-initialized-cold-sac', status='frozen-ready-for-separate-explicit-execution',
        purpose='Artifact-level replica of the paper stages BC -> Soft-SARSA critic -> online SAC from cold reset, initialized from verified cold-reset MPC demonstrations produced with the measured diagnostic prior (not a SAC-derived primitive; a disclosed deviation). No MPC scaffolding chunks in the online phase.',
        demonstration_collection=str(C), demonstration_episodes=episodes, demonstration_count=len(episodes), control_period_seconds=float(a.period if a.period is not None else 0.1),
        control_period_note=('archived 0.1 s' if a.period is None else f'route (b): {a.period} s instead of the archived 0.1 s (paper reference 10 Hz), a disclosed deviation; the per-control reward time penalty (-0.3) and the observation finite-difference velocities are therefore defined over this period'),
        train_seed=a.seed, demonstration_seed=a.seed + 1, physical_reset_seed=42,
        bc_updates=a.bc, bc_sampling='episode-weighted (softmax return / 1) as in the archived learner', critic_pretrain_updates=a.pretrain,
        online_policy_interactions=a.online, checkpoints=checkpoints, learning_starts=a.learning_starts, batch_size=256, replay_capacity=100000,
        offline_fraction=a.offline_fraction, raw_reward_divisor=200., utd=1,
        eval_deterministic_seed=42, eval_stochastic_seeds=[a.seed + 100 + i for i in range(1, a.eval_trials + 1)], eval_stochastic_trials=a.eval_trials,
        evaluation='cold deterministic plus stochastic trials at bc_complete, pretrained and every online checkpoint; separate environments, training RNG restored, nothing enters replay',
        learner='fresh SACAgent with the SAC0001 LearnerConfig; BC mean-only then initialize_std; archived pretrain_critic_step and update',
        eligible_for_primitive=False, main_mpc_dataset_eligible=False, no_trajectory_export=True, source_bindings=bindings))
    write(D / 'execution_manifest.json', {str(p.relative_to(D)): sha(p) for p in sorted(D.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name not in ('execution_manifest.json', 'run.log')})
    print(json.dumps(dict(stage=str(D), demonstrations=len(episodes), frozen=True, executed=False)))

if __name__ == '__main__':
    main()
