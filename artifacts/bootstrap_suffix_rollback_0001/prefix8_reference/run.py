"""One fixed suffix support probe; no training, state injection or physics edits."""
from pathlib import Path
import hashlib
import json
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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def run():
    assert not (D / 'summary.json').exists()
    inputs = json.loads((D / 'inputs_manifest.json').read_text())
    assert all(sha(D / path) == digest for path, digest in inputs.items())
    protocol = json.loads((D / 'protocol.json').read_text())
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
    assert (agent.obs_dim, agent.action_dim, agent.bc_updates, agent.pretrain_updates, agent.online_updates) == (60, 16, 12000, 0, 0)
    before = {n: {k: sha_bytes(v.detach().cpu().numpy().tobytes()) for k, v in getattr(agent, n).state_dict().items()} for n in ('actor', 'critics', 'target_critics')}
    with np.load(D / 'prefix_transitions.npz') as z:
        source = {k: z[k].copy() for k in z.files}
    with np.load(D / 'prefix_physics.npz') as z:
        source_physics = {k: z[k].copy() for k in z.files}
    assert len(source['actions']) == 9 and source['terminated'][-1]
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    runtime = dict(torch=str(torch.__version__), mujoco=mujoco.__version__, numpy=np.__version__, device=torch.cuda.get_device_name(), torch_threads=1, interop_threads=1, tf32=False, matmul_precision=torch.get_float32_matmul_precision())
    write(D / 'runtime.json', runtime)
    results = []
    start = time.monotonic()
    for experiment in protocol['experiments']:
        seed, deterministic = experiment['seed'], experiment['deterministic']
        dest = D / ('deterministic' if deterministic else f'stochastic_{seed}')
        dest.mkdir()
        env = environment_for_policy(ec, metadata['policy_interface'])
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.save({'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}, dest / 'rng_initial.pt')
        records = {k: [] for k in ('obs', 'actions', 'policy_controlled', 'raw_mean', 'latent_std', 'log_prob', 'next_obs', 'rewards', 'terminated', 'truncated', 'qpos', 'qvel', 'targets', 'integration_state')}
        initial = {}
        original_reset, original_step = env.reset, env.step

        def state():
            out = np.empty(mujoco.mj_stateSize(env.model, state_spec))
            mujoco.mj_getState(env.model, env.data, out, state_spec)
            return out

        def reset(*args, **kwargs):
            ob, info = original_reset(*args, **kwargs)
            initial.update(obs=ob.copy(), integration_state=state(), qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(), targets=env.data.ctrl.copy())
            return ob, info

        def step(action):
            result = original_step(action)
            ob, reward, term, trunc, info = result
            values = dict(next_obs=ob.copy(), rewards=reward, terminated=term, truncated=trunc, qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(), targets=env.data.ctrl.copy(), integration_state=state())
            for k, v in values.items():
                records[k].append(v)
            return result

        class Policy:
            @torch.no_grad()
            def act(self, ob, deterministic=True):
                index = len(records['actions'])
                tensor = torch.as_tensor(ob[None], dtype=torch.float32, device='cuda')
                mean, log_std = agent.actor(tensor)
                if index < 8:
                    action = source['actions'][index].copy()
                    log_prob = np.nan  # Recorded prefix: no sampled policy density claimed.
                    np.testing.assert_array_equal(ob, source['obs'][index])
                elif deterministic:
                    action = mean.tanh()[0].cpu().numpy()
                    log_prob = float(agent.actor._log_prob(mean, log_std, mean)[0])
                else:
                    sampled, density = agent.actor.sample(tensor)
                    action = sampled[0].cpu().numpy()
                    log_prob = float(density[0])
                for k, v in dict(obs=ob.copy(), actions=action.copy(), policy_controlled=index >= 8, raw_mean=mean[0].cpu().numpy(), latent_std=log_std[0].exp().cpu().numpy(), log_prob=log_prob).items():
                    records[k].append(v)
                return action

        env.reset, env.step = reset, step
        try:
            result, physics, commands = audit_policy.audit_episode(env, Policy(), seed, 100, deterministic)
        finally:
            env.close()
        arrays = {k: np.asarray(v) for k, v in records.items()}
        for key in ('obs', 'actions', 'next_obs', 'rewards', 'terminated', 'truncated', 'qpos', 'qvel', 'targets', 'integration_state'):
            np.testing.assert_array_equal(arrays[key][:8], source[key][:8])
        np.testing.assert_array_equal(physics[:400], source_physics['samples'][:400])
        if deterministic:
            for key in ('obs', 'actions', 'next_obs', 'qpos', 'qvel', 'targets', 'integration_state'):
                np.testing.assert_array_equal(arrays[key], source[key])
            np.testing.assert_array_equal(physics, source_physics['samples'])
            assert result['finger_supported_rotation_success']
        assert np.allclose(arrays['latent_std'][8:], .05, atol=1e-8, rtol=0)
        suffix_rewards = arrays['rewards'][8:].astype(np.float64)
        suffix_logp = arrays['log_prob'][8:]
        discounts = agent.config.discount ** np.arange(len(suffix_rewards))
        suffix_soft_q = float(discounts @ suffix_rewards - float(agent.alpha.detach()) * (discounts[1:] @ suffix_logp[1:]))
        result.update(deterministic=deterministic, prefix_controls=8, policy_suffix_controls=len(suffix_rewards), prefix_physics_bitexact=True, fixed_prefix_assisted=True, cold_start_policy_success=False, sac_policy_success=False, learner_online_updates=0, suffix_discounted_return=float(discounts @ suffix_rewards), suffix_soft_q_return=suffix_soft_q, terminal_complete=bool(arrays['terminated'][-1]))
        np.savez_compressed(dest / 'transitions.npz', **arrays, **{'initial_' + k: v for k, v in initial.items()})
        np.savez_compressed(dest / 'physics.npz', samples=physics, columns=np.asarray(audit_policy.PHYSICS_COLUMNS))
        torch.save({'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}, dest / 'rng_final.pt')
        write(dest / 'report.json', result)
        write(dest / 'commands.json', commands)
        results.append(result)
        write(D / 'progress.json', results)
        print(json.dumps({k: result[k] for k in ('seed', 'deterministic', 'steps', 'policy_suffix_controls', 'finger_supported_rotation_success', 'final_force_qualified_angle', 'environment_failure_reason', 'suffix_soft_q_return')}), flush=True)
    after = {n: {k: sha_bytes(v.detach().cpu().numpy().tobytes()) for k, v in getattr(agent, n).state_dict().items()} for n in before}
    assert before == after
    assert all(sha(D / path) == digest for path, digest in inputs.items())
    stochastic = results[1:]
    summary = dict(status='complete', protocol=protocol, runtime=runtime, results=results, stochastic_prefix_assisted_successes=sum(r['finger_supported_rotation_success'] for r in stochastic), stochastic_episodes=5, deterministic_control_reproduces_source_bitexact=True, all_prefixes_bitexact=True, actor_critics_unchanged=True, new_training_updates=0, eligible_primitive=False, wall_seconds=time.monotonic() - start)
    write(D / 'summary.json', summary)
    write(D / 'manifest.json', {str(p.relative_to(D)): sha(p) for p in sorted(D.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name not in ('manifest.json', 'run.log')})


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


if __name__ == '__main__':
    run()
