"""Prespecified descending prefix support screen; no optimization or state injection."""
from pathlib import Path
import hashlib
import json
import platform
import sys
import time

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D / 'source'))
import mujoco
import numpy as np
import torch
from mpc_dex.env import EnvConfig
from mpc_dex.bootstrap_env import environment_for_policy
from mpc_dex.contracts import validate_checkpoint
from mpc_dex.learning import SACAgent
from scripts import audit_policy
from scripts.train_bootstrap import save_environment


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def rng():
    return {'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def same_rng(a, b):
    return torch.equal(a['cpu'], b['cpu']) and all(torch.equal(x, y) for x, y in zip(a['cuda'], b['cuda'], strict=True))


def tensors(agent):
    return {name: {key: hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest() for key, value in getattr(agent, name).state_dict().items()} for name in ('actor', 'critics', 'target_critics')}


def strict_success(result):
    return bool(result['environment_success'] and result['finger_supported_rotation_success']
                and not result['environment_drop'] and not result['independent_drop']
                and not result['independent_palm_support_violation'] and result['no_external_support']
                and result['terminated'] and not result['truncated'])


def run():
    assert not (D / 'summary.json').exists()
    inputs = json.loads((D / 'inputs_manifest.json').read_text())
    assert all(sha(D / path) == digest for path, digest in inputs.items())
    protocol = json.loads((D / 'protocol.json').read_text())
    assert protocol['prefix_order'] == list(range(6, 0, -1))
    assert protocol['stop_after_first_stage_with_fewer_than_stochastic_successes'] == 4
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ec = EnvConfig(**json.loads((D / 'config.json').read_text())['environment'])
    assert ec.task == 'rotation' and ec.horizon_steps == 100 and ec.terminate_on_success
    metadata = validate_checkpoint(D / 'checkpoint.pt', ec)
    assert metadata['policy_interface'] == {'kind': 'absolute_joint_target'}
    agent = SACAgent.load(D / 'checkpoint.pt', 'cuda')
    assert (agent.obs_dim, agent.action_dim, agent.bc_updates, agent.pretrain_updates, agent.online_updates) == (60, 16, 12000, 500, 2000)
    before = tensors(agent)
    alpha_before = float(agent.alpha.detach())
    with np.load(D / 'prefix_transitions.npz') as z:
        source = {key: z[key].copy() for key in z.files}
    with np.load(D / 'prefix_physics.npz') as z:
        source_physics = {key: z[key].copy() for key in z.files}
    assert len(source['actions']) == 9 and source['terminated'][-1]
    assert tuple(source_physics['columns']) == audit_policy.PHYSICS_COLUMNS
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    runtime = dict(python=platform.python_version(), platform=platform.platform(), torch=str(torch.__version__), cuda=torch.version.cuda,
                   cudnn=torch.backends.cudnn.version(), mujoco=mujoco.__version__, numpy=np.__version__, device=torch.cuda.get_device_name(),
                   capability=list(torch.cuda.get_device_capability()), torch_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                   tf32=False, matmul_precision=torch.get_float32_matmul_precision(), deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                   cudnn_deterministic=torch.backends.cudnn.deterministic, cudnn_benchmark=torch.backends.cudnn.benchmark)
    write(D / 'runtime.json', runtime)
    all_stages = []
    start = time.monotonic()
    for prefix_controls in protocol['prefix_order']:
        stage_dir = D / f'prefix_{prefix_controls:02d}'
        stage_dir.mkdir()
        results = []
        for experiment in protocol['experiments_per_stage']:
            seed, deterministic = experiment['seed'], experiment['deterministic']
            dest = stage_dir / ('deterministic' if deterministic else f'stochastic_{seed}')
            dest.mkdir()
            env = environment_for_policy(ec, metadata['policy_interface'])
            assert env.substeps == 50
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            initial_rng = rng()
            torch.save(initial_rng, dest / 'rng_initial.pt')
            records = {key: [] for key in ('obs', 'actions', 'policy_controlled', 'raw_mean', 'latent_std', 'log_prob', 'next_obs', 'rewards', 'terminated', 'truncated', 'qpos', 'qvel', 'targets', 'integration_state', 'decision_times', 'control_times')}
            initial = {}
            original_reset, original_step = env.reset, env.step

            def state():
                out = np.empty(mujoco.mj_stateSize(env.model, state_spec))
                mujoco.mj_getState(env.model, env.data, out, state_spec)
                return out

            def reset(*args, **kwargs):
                kwargs['seed'] = protocol['physical_reset_seed']
                ob, info = original_reset(*args, **kwargs)
                initial.update(obs=ob.copy(), integration_state=state(), qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(), targets=env.data.ctrl.copy())
                for key, value in initial.items():
                    np.testing.assert_array_equal(value, source['initial_' + ('ctrl' if key == 'targets' else key)])
                save_environment(dest / 'initial_environment.npz', env, ob, 0., [env.data.qpos[env.q_ids].copy()])
                return ob, info

            def step(action):
                result = original_step(action)
                ob, reward, term, trunc, info = result
                values = dict(next_obs=ob.copy(), rewards=reward, terminated=term, truncated=trunc, qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(), targets=env.data.ctrl.copy(), integration_state=state(), control_times=float(env.data.time))
                for key, value in values.items():
                    records[key].append(value)
                return result

            class Policy:
                @torch.no_grad()
                def act(self, ob, deterministic=True):
                    index = len(records['actions'])
                    tensor = torch.as_tensor(ob[None], dtype=torch.float32, device='cuda')
                    mean, log_std = agent.actor(tensor)
                    if index == prefix_controls:
                        handoff_rng = rng()
                        assert same_rng(initial_rng, handoff_rng)
                        torch.save(handoff_rng, dest / 'rng_handoff.pt')
                        trace = [initial['qpos'][env.q_ids]] + [q[env.q_ids] for q in records['qpos']]
                        save_environment(dest / 'handoff_environment.npz', env, ob, float(sum(records['rewards'])), trace)
                    if index < prefix_controls:
                        action = source['actions'][index].copy()
                        log_prob = np.nan  # No sampled density is claimed for teacher controls.
                        np.testing.assert_array_equal(ob, source['obs'][index])
                    elif deterministic:
                        action = mean.tanh()[0].cpu().numpy()
                        log_prob = float(agent.actor._log_prob(mean, log_std, mean)[0])
                    else:
                        sampled, density = agent.actor.sample(tensor)
                        action = sampled[0].cpu().numpy()
                        log_prob = float(density[0])
                    values = dict(obs=ob.copy(), actions=action.copy(), policy_controlled=index >= prefix_controls, raw_mean=mean[0].cpu().numpy(), latent_std=log_std[0].exp().cpu().numpy(), log_prob=log_prob, decision_times=float(env.data.time))
                    for key, value in values.items():
                        records[key].append(value)
                    return action

            env.reset, env.step = reset, step
            try:
                result, physics, commands = audit_policy.audit_episode(env, Policy(), seed, 100, deterministic)
                trace = [initial['qpos'][env.q_ids]] + [q[env.q_ids] for q in records['qpos']]
                save_environment(dest / 'terminal_environment.npz', env, records['next_obs'][-1], float(sum(records['rewards'])), trace)
            finally:
                env.close()
            arrays = {key: np.asarray(value) for key, value in records.items()}
            checked_fields = ('obs', 'actions', 'next_obs', 'rewards', 'terminated', 'truncated', 'qpos', 'qvel', 'targets', 'integration_state')
            for key in checked_fields:
                np.testing.assert_array_equal(arrays[key][:prefix_controls], source[key][:prefix_controls])
            np.testing.assert_array_equal(physics[:prefix_controls * 50], source_physics['samples'][:prefix_controls * 50])
            if deterministic:
                assert same_rng(initial_rng, rng())
            suffix_std = arrays['latent_std'][prefix_controls:]
            assert np.isfinite(suffix_std).all() and (suffix_std >= agent.config.min_std * .99999).all() and (suffix_std <= agent.config.max_std * 1.00001).all()
            suffix_rewards = arrays['rewards'][prefix_controls:].astype(np.float64)
            suffix_logp = arrays['log_prob'][prefix_controls:]
            discounts = agent.config.discount ** np.arange(len(suffix_rewards))
            soft_return = float(discounts @ suffix_rewards - alpha_before * protocol['reward_divisor'] * (discounts[1:] @ suffix_logp[1:]))
            handoff = {key: float(source_physics['samples'][prefix_controls * 50 - 1, index]) for index, key in enumerate(audit_policy.PHYSICS_COLUMNS)}
            terminal_complete = bool(arrays['terminated'][-1])
            result.update(deterministic=deterministic, prefix_controls=prefix_controls, policy_suffix_controls=len(suffix_rewards),
                          initial_and_prefix_fields_bitexact=list(checked_fields), prefix_physics_bitexact=True,
                          deterministic_suffix_may_differ_from_original=True, physical_reset_seed=protocol['physical_reset_seed'], no_policy_rng_draws_before_handoff=True,
                          fixed_prefix_assisted=True, cold_start_policy_success=False, sac_policy_success=False,
                          learner_online_updates=agent.online_updates, learner_pretrain_updates=agent.pretrain_updates, source_is_sac_trained=True, assisted_sac_completion=strict_success(result), strict_force_qualified_success=strict_success(result),
                          handoff_physics=handoff, suffix_discounted_return=float(discounts @ suffix_rewards),
                          observed_suffix_soft_return_raw_equivalent=soft_return, observed_suffix_soft_return_normalized=soft_return / protocol['reward_divisor'], alpha_normalized=alpha_before, alpha_raw_equivalent=alpha_before * protocol['reward_divisor'], suffix_std_min=float(suffix_std.min()), suffix_std_max=float(suffix_std.max()), terminal_complete=terminal_complete,
                          soft_return_has_unmeasured_tail=not terminal_complete,
                          entropy_convention='Q-like suffix return excludes the first suffix action entropy and includes later sampled-action entropy')
            np.savez_compressed(dest / 'transitions.npz', **arrays, **{'initial_' + key: value for key, value in initial.items()})
            np.savez_compressed(dest / 'physics.npz', samples=physics, columns=np.asarray(audit_policy.PHYSICS_COLUMNS))
            torch.save(rng(), dest / 'rng_final.pt')
            write(dest / 'report.json', result)
            write(dest / 'commands.json', commands)
            results.append(result)
            write(stage_dir / 'progress.json', results)
            print(json.dumps({key: result[key] for key in ('prefix_controls', 'seed', 'deterministic', 'steps', 'policy_suffix_controls', 'strict_force_qualified_success', 'final_force_qualified_angle', 'environment_failure_reason')}), flush=True)
        successes = sum(result['strict_force_qualified_success'] for result in results[1:])
        stage = dict(prefix_controls=prefix_controls, stochastic_successes=successes, stochastic_episodes=5,
                     passes_prespecified_gate=successes >= 4, results=results)
        write(stage_dir / 'summary.json', stage)
        all_stages.append(stage)
        write(D / 'progress.json', all_stages)
        if successes < 4:
            break
    assert before == tensors(agent) and alpha_before == float(agent.alpha.detach())
    assert (agent.bc_updates, agent.pretrain_updates, agent.online_updates) == (12000, 500, 2000)
    assert all(sha(D / path) == digest for path, digest in inputs.items())
    summary = dict(status='complete', protocol=protocol, runtime=runtime, stages=all_stages,
                   stop_reason='first_stage_below_four_of_five' if all_stages[-1]['stochastic_successes'] < 4 else 'all_six_stages_completed',
                   trained_deterministic_suffix_not_required_to_reproduce_original=True, all_prefixes_bitexact=True,
                   actor_critics_targets_alpha_and_counters_unchanged=True, new_training_updates=0,
                   all_results_are_prefix_assisted=True, eligible_primitive=False, wall_seconds=time.monotonic() - start)
    write(D / 'summary.json', summary)
    write(D / 'manifest.json', {str(path.relative_to(D)): sha(path) for path in sorted(D.rglob('*')) if path.is_file() and '__pycache__' not in path.parts and path.name not in ('manifest.json', 'run.log')})


if __name__ == '__main__':
    run()
