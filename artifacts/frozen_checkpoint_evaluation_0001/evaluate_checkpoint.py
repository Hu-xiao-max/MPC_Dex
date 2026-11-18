"""Large-sample frozen evaluation of one saved variant checkpoint; no learning, nothing enters replay.

Usage: evaluate_checkpoint.py <variant_dir_name> <checkpoint_label> <trials_per_prefix>
Imports the variant's own archived runner (Episode, command, environment) so physics, prefix
replay checks and success criteria are exactly those of the stage. Per-trial physics and
transitions are not saved (disclosed); each trial's report.json fields are kept.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import time

A = Path('/home/autel001/code/mpc_dex/artifacts')
OUT = Path(__file__).resolve().parent / 'results'
PREFIXES = (5, 6, 7, 8)  # always evaluated; a stage's own eval_prefixes below 5 are added
SEED_BASE = 23000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    variant, label, trials = sys.argv[1], sys.argv[2], int(sys.argv[3])
    D = A / variant
    spec = importlib.util.spec_from_file_location('variant_runner_' + variant, D / 'run.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    np, torch = runner.np, runner.torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    ck = D / 'checkpoints' / label
    manifest = json.loads((ck / 'manifest.json').read_text())
    assert sha(ck / 'agent.pt') == manifest['agent.pt'], 'checkpoint agent hash changed'
    protocol = json.loads((D / 'protocol.json').read_text())
    ec = runner.EnvConfig(**json.loads((D / 'config.json').read_text())['environment'])
    agent = runner.SACAgent.load(ck / 'agent.pt', 'cuda')
    with np.load(D / 'prefix_transitions.npz', allow_pickle=False) as z:
        source = {k: z[k].copy() for k in z.files}
    with np.load(D / 'prefix_physics.npz', allow_pickle=False) as z:
        physics = z['samples'].copy()
    before = {k: runner.tensor_hashes(getattr(agent, k)) for k in ('actor', 'critics', 'target_critics')}
    out = dict(variant=variant, checkpoint=label, agent_sha256=manifest['agent.pt'], trials_per_prefix=trials,
               learner_counters=dict(bc_updates=agent.bc_updates, pretrain_updates=agent.pretrain_updates, online_updates=agent.online_updates),
               prefixes={}, physics_and_transitions_saved=False)
    started = time.monotonic()
    prefixes = sorted(set(PREFIXES) | {int(q) for q in protocol['eval_prefixes'] if 0 < int(q) < 5})
    out['evaluated_prefixes'] = prefixes
    for prefix in (*prefixes, 0):
        rows = []
        tasks = [('deterministic', protocol['eval_deterministic_seed'], True)] + [(f'stochastic_{s}', s, False) for s in range(SEED_BASE + 100 * prefix + 1, SEED_BASE + 100 * prefix + 1 + trials)]
        for name, seed, deterministic in tasks:
            runner.seed_all(seed)
            env = runner.environment_for_policy(ec, protocol['policy_interface'])
            episode = None
            try:
                episode = runner.Episode(env, prefix, source, physics, protocol['physical_reset_seed'], name, seed, protocol['raw_reward_divisor'])
                while not (episode.term or episode.trunc):
                    episode.advance(runner.command(agent, episode.obs, deterministic))
                r = episode.report()
                rows.append(dict(trial=name, seed=seed, deterministic=deterministic, success=bool(r['finger_supported_rotation_success']),
                                 policy_suffix_controls=r['policy_suffix_controls'], failure=r['environment_failure_reason'],
                                 truncated=bool(r['truncated']), final_force_qualified_angle=r['final_force_qualified_angle'],
                                 max_force_qualified_angle=r['max_force_qualified_angle'], palm=bool(r['independent_palm_support_violation'])))
            finally:
                if episode is not None:
                    episode.close()
                env.close()
        stoch = [x for x in rows if not x['deterministic']]
        out['prefixes'][str(prefix)] = dict(deterministic_success=rows[0]['success'], stochastic_successes=sum(x['success'] for x in stoch),
                                            stochastic_trials=len(stoch), trials=rows)
    assert before == {k: runner.tensor_hashes(getattr(agent, k)) for k in before}
    out['wall_seconds'] = time.monotonic() - started
    (OUT / variant).mkdir(parents=True, exist_ok=True)
    (OUT / variant / f'{label}.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(dict(variant=variant, checkpoint=label, **{f'p{p}': f"{out['prefixes'][str(p)]['stochastic_successes']}/{trials}{'D' if out['prefixes'][str(p)]['deterministic_success'] else ''}" for p in (*prefixes, 0)}, wall=round(out['wall_seconds'], 1))), flush=True)


if __name__ == '__main__':
    main()
