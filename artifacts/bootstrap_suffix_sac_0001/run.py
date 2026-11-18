"""Bounded artifact-only suffix SAC diagnostic. Never imports live core code.

Execution is separate from preparation and requires --execute plus a finalized
execution_manifest.json. No resume is attempted after an interrupted run.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
import random
import sys
import time

D = Path(__file__).resolve().parent
for required in ('mpc_dex/env.py', 'mpc_dex/learning.py', 'mpc_dex/replay.py', 'scripts/audit_policy.py'):
    if not (D / 'source' / required).is_file():
        raise RuntimeError('Missing archived numerical source; prepare without execution first: ' + required)
sys.path.insert(0, str(D / 'source'))

import mujoco
import numpy as np
import torch

from mpc_dex.bootstrap_env import environment_for_policy
from mpc_dex.contracts import validate_checkpoint, write_checkpoint_metadata
from mpc_dex.env import EnvConfig
from mpc_dex.learning import SACAgent
from mpc_dex.replay import ReplayBuffer
from scripts.audit_policy import PhysicsObserver, PHYSICS_COLUMNS

for name in ('mpc_dex.env', 'mpc_dex.learning', 'mpc_dex.replay', 'mpc_dex.bootstrap_env', 'scripts.audit_policy'):
    if not Path(sys.modules[name].__file__).resolve().is_relative_to((D / 'source').resolve()):
        raise RuntimeError('Runtime imported source outside the frozen artifact: ' + name)

ENV_ARRAYS = ('initial_position', 'initial_orientation', 'last_action', 'last_q', 'last_p', '_last_object_quaternion')
ENV_SCALARS = ('start_angle', '_last_wrapped', 'angle', 'previous_angle', 'steps', 'rotations', 'dwell', 'goal',
               'heading_error', '_episode_over', '_contactless_seconds', '_goal_rewarded', 'retained_angle',
               'previous_retained_angle', 'retained_positive_step', 'retained_negative_step',
               '_palm_support_violation', '_peak_palm_support_force')
SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
REPLAY_KEYS = ('obs', 'actions', 'rewards', 'next_obs', 'terminated', 'truncated')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def verify_manifest():
    manifest = json.loads((D / 'execution_manifest.json').read_text())
    if not all((D / name).is_file() and sha(D / name) == digest for name, digest in manifest.items()):
        raise RuntimeError('Frozen execution source/input hash changed')
    return manifest


def rng_state():
    return dict(torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state_all(),
                numpy=np.random.get_state(), python=random.getstate())


def restore_rng(state):
    torch.set_rng_state(state['torch_cpu'])
    torch.cuda.set_rng_state_all(state['torch_cuda'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])


def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def tensor_hashes(module):
    return {k: hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()
            for k, v in module.state_dict().items()}


def integration_state(env):
    state = np.empty(mujoco.mj_stateSize(env.model, SPEC))
    mujoco.mj_getState(env.model, env.data, state, SPEC)
    return state


def boundary(env):
    return dict(integration_state=integration_state(env), qpos=env.data.qpos.copy(),
                qvel=env.data.qvel.copy(), targets=env.data.ctrl.copy(), time=float(env.data.time),
                **{'env_' + k: np.asarray(getattr(env, k)).copy() for k in ENV_ARRAYS},
                env_scalars_json=json.dumps(jsonable({k: getattr(env, k) for k in ENV_SCALARS}), allow_nan=False))


@torch.no_grad()
def command(agent, obs, deterministic=False):
    value = torch.as_tensor(obs[None], dtype=torch.float32, device=agent.device)
    mean, log_std = agent.actor(value)
    if deterministic:
        action = mean.tanh()
        density = agent.actor._log_prob(mean, log_std, mean)
    else:
        action, density = agent.actor.sample(value)
    return dict(action=action[0].cpu().numpy(), log_prob=float(density[0]),
                raw_mean=mean[0].cpu().numpy(), latent_std=log_std[0].exp().cpu().numpy(),
                density_is_stochastic_sample=not deterministic,
                learner_bc_updates=agent.bc_updates, learner_pretrain_updates=agent.pretrain_updates,
                learner_online_updates=agent.online_updates)


class Episode:
    """Actual cold prefix and suffix with one continuous independent observer."""
    def __init__(self, env, prefix, source, source_physics, seed, phase, number, divisor):
        self.env, self.prefix, self.phase, self.number, self.divisor = env, prefix, phase, number, divisor
        self.obs, self.info = env.reset(seed=seed)
        self.seed = seed
        self.initial = dict(obs=self.obs.copy(), **boundary(env))
        self.initial_rng = rng_state()
        self.observer = PhysicsObserver(env)
        self.rows, self.closed = [], False
        self.term = self.trunc = False
        for index in range(prefix):
            np.testing.assert_array_equal(self.obs, source['obs'][index])
            self.advance(dict(action=source['actions'][index].copy(), log_prob=np.nan,
                              raw_mean=np.full(16, np.nan, np.float32),
                              latent_std=np.full(16, np.nan, np.float32),
                              density_is_stochastic_sample=False, learner_bc_updates=-1,
                              learner_pretrain_updates=-1, learner_online_updates=-1), policy=False)
            row = self.rows[-1]
            for key in ('obs', 'actions', 'next_obs', 'rewards', 'terminated', 'truncated',
                        'qpos', 'qvel', 'targets', 'integration_state'):
                np.testing.assert_array_equal(row[key], source[key][index])
            if self.term or self.trunc:
                raise RuntimeError('Recorded prefix terminated before handoff')
        if prefix:
            np.testing.assert_array_equal(np.asarray(self.observer.rows), source_physics[:prefix * env.substeps])
        self.handoff = dict(obs=self.obs.copy(), **boundary(env))

    def advance(self, proposal, policy=True):
        assert not self.term and not self.trunc
        before = self.obs.copy()
        action = proposal['action']
        assert action.dtype == np.float32 and action.shape == (16,) and np.isfinite(action).all()
        self.obs, reward, self.term, self.trunc, self.info = self.env.step(action)
        row = dict(obs=before, actions=action.copy(), next_obs=self.obs.copy(), rewards=float(reward),
                   normalized_rewards=float(reward) / self.divisor,
                   terminated=bool(self.term), truncated=bool(self.trunc),
                   policy_controlled=bool(policy), log_prob=proposal['log_prob'],
                   log_prob_valid=bool(policy), density_is_stochastic_sample=proposal['density_is_stochastic_sample'],
                   learner_bc_updates=proposal['learner_bc_updates'],
                   learner_pretrain_updates=proposal['learner_pretrain_updates'],
                   learner_online_updates=proposal['learner_online_updates'],
                   raw_mean=proposal['raw_mean'].copy(), latent_std=proposal['latent_std'].copy(),
                   **boundary(self.env))
        self.rows.append(row)
        if len(self.observer.rows) != len(self.rows) * self.env.substeps:
            raise RuntimeError('Missing per-physics observer samples')
        if abs(self.observer.raw_angle - self.info['angle']) > 1e-8 or abs(self.observer.qualified_angle - self.info['retained_angle']) > 1e-8:
            raise RuntimeError('Independent rotation observer disagrees with task progress')
        return row

    def arrays(self):
        out = {k: np.asarray([r[k] for r in self.rows]) for k in self.rows[0]} if self.rows else {}
        # These are measured successors within this single physical episode.
        # Prefix rows remain evidence only and never enter learner replay.
        if self.rows:
            out['next_actions'] = np.zeros_like(out['actions'])
            out['next_action_valid'] = np.zeros(len(self.rows), dtype=bool)
            out['next_actions'][:-1] = out['actions'][1:]
            out['next_action_valid'][:-1] = ~(out['terminated'][:-1] | out['truncated'][:-1])
        return out

    def report(self):
        observer, env = self.observer, self.env
        physics = np.asarray(observer.rows, dtype=np.float64)
        col = {k: physics[:, i] for i, k in enumerate(PHYSICS_COLUMNS)}
        external_keys = ('palm_force_norm', 'table_force_norm', 'other_force_norm',
                         'table_normal_force', 'other_normal_force',
                         'object_applied_wrench_norm', 'object_applied_generalized_force_norm')
        no_external = not observer.palm_support_violation and all(np.max(col[k], initial=0) <= 1e-6 for k in external_keys)
        force_success = bool(self.term and not self.trunc and self.info.get('success') and not self.info.get('drop') and
                             col['force_bearing_fingers'][-1] >= 2 and observer.raw_angle >= 2 * np.pi and
                             observer.qualified_angle >= 2 * np.pi and observer.force_qualified_angle >= 2 * np.pi and
                             col['force_retained'][-1] and not observer.independent_drop and no_external and env.model.neq == 0)
        rewards = np.asarray([r['rewards'] for r in self.rows])
        suffix = rewards[self.prefix:]
        return dict(phase=self.phase, episode_number=self.number, seed=self.seed,
                    prefix_controls=self.prefix, policy_suffix_controls=len(suffix),
                    steps=len(self.rows), physics_steps=len(physics),
                    terminated=bool(self.term), truncated=bool(self.trunc),
                    environment_success=bool(self.info.get('success')), environment_drop=bool(self.info.get('drop')),
                    environment_failure_reason=self.info.get('failure_reason'),
                    finger_supported_rotation_success=force_success,
                    independent_drop=observer.independent_drop,
                    independent_palm_support_violation=observer.palm_support_violation,
                    no_external_support=bool(no_external), model_equality_constraints=env.model.neq,
                    final_raw_angle=observer.raw_angle, final_qualified_angle=observer.qualified_angle,
                    final_force_qualified_angle=observer.force_qualified_angle,
                    max_force_qualified_angle=float(np.max(col['force_qualified_angle'], initial=0)),
                    final_force_bearing_fingers=int(col['force_bearing_fingers'][-1]) if len(physics) else 0,
                    peak_palm_normal_force=float(np.max(col['palm_normal_force'], initial=0)),
                    raw_full_return=float(rewards.sum()), raw_suffix_return=float(suffix.sum()),
                    normalized_suffix_return=float(suffix.sum() / self.divisor),
                    prefix_physics_bitexact=True, fixed_prefix_assisted=bool(self.prefix),
                    cold_start_policy_success=bool(self.prefix == 0 and force_success),
                    environment_episode_complete=bool(self.term or self.trunc),
                    empirical_terminal_return_available=bool(self.term and not self.trunc),
                    truncation_tail_unknown=bool(self.trunc),
                    collector_stopped_without_terminal=not bool(self.term or self.trunc),
                    episode_policy_may_change_between_decisions=self.phase == 'online',
                    frozen_policy_success_claim=False if self.phase == 'online' else None,
                    eligible_for_primitive=False, trajectory_exported=False)

    def save(self, dest, extra=None):
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest / 'transitions.npz', **self.arrays(),
                            **{'initial_' + k: v for k, v in self.initial.items()},
                            **{'handoff_' + k: v for k, v in self.handoff.items()})
        np.savez_compressed(dest / 'physics.npz', samples=np.asarray(self.observer.rows, np.float64),
                            columns=np.asarray(PHYSICS_COLUMNS))
        write(dest / 'report.json', dict(self.report(), **(extra or {})))
        torch.save(self.initial_rng, dest / 'rng_initial.pt')

    def close(self):
        if not self.closed:
            self.observer.close()
            self.closed = True


def normalized_batch(replay, protocol):
    batch = replay.sample(protocol['batch_size'])
    # sample() returns copies; the raw replay and its episode returns stay intact.
    raw = batch['rewards'].copy()
    batch['rewards'] = raw / np.float32(protocol['raw_reward_divisor'])
    return batch, dict(batch_raw_reward_mean=float(raw.mean()),
                       batch_normalized_reward_mean=float(batch['rewards'].mean()))


def terminal_returns(episode, gamma, alpha_normalized, divisor, pretrain_alpha_normalized):
    rows = episode.arrays()
    start = episode.prefix
    if not episode.term or episode.trunc:
        return None  # Horizon truncation has an unobserved continuation value.
    rewards = rows['rewards'][start:].astype(np.float64) / divisor
    logp = rows['log_prob'][start:].astype(np.float64)
    result = rewards.copy()
    pretrain_result = rewards.copy()
    for i in range(len(result) - 2, -1, -1):
        result[i] += gamma * (result[i + 1] - alpha_normalized * logp[i + 1])
        pretrain_result[i] += gamma * (pretrain_result[i + 1] - pretrain_alpha_normalized * logp[i + 1])
    return dict(obs=rows['obs'][start:], actions=rows['actions'][start:],
                soft_q_normalized=result, soft_q_raw_equivalent=result * divisor,
                soft_q_pretrain_normalized=pretrain_result, soft_q_pretrain_raw_equivalent=pretrain_result * divisor,
                log_prob=logp, raw_rewards=rows['rewards'][start:],
                episode_number=np.full(len(result), episode.number, np.int64))


def concatenate_returns(episodes):
    episodes = [e for e in episodes if e is not None and len(e['obs'])]
    return {k: np.concatenate([e[k] for e in episodes]) for k in episodes[0]} if episodes else {}


@torch.no_grad()
def calibrate(agent, samples, path, divisor, role):
    if not samples:
        write(path.with_suffix('.json'), dict(role=role, terminal_rows=0, reason='No complete terminal targets'))
        return
    predictions = []
    for obs, action in zip(samples['obs'], samples['actions']):
        predictions.append(agent.critics(torch.as_tensor(obs[None], device=agent.device),
                                        torch.as_tensor(action[None], device=agent.device))[:, 0].cpu().numpy())
    values = np.asarray(predictions, np.float64)
    target = samples['soft_q_normalized']
    np.savez_compressed(path, **samples, q_normalized=values, q_raw_equivalent=values * divisor)
    errors = values - target[:, None]
    pretrain_errors = values - samples['soft_q_pretrain_normalized'][:, None]
    write(path.with_suffix('.json'), dict(role=role, terminal_rows=len(target),
          target_normalized_mean=float(target.mean()), target_raw_mean=float(target.mean() * divisor),
          critic_normalized_bias=errors.mean(axis=0), critic_normalized_rmse=np.sqrt((errors ** 2).mean(axis=0)),
          critic_raw_equivalent_rmse=np.sqrt((errors ** 2).mean(axis=0)) * divisor,
          initial_online_alpha_normalized=.05 / 200,
          pretrain_alpha_normalized=.02 / 200,
          pretrain_target_normalized_mean=float(samples['soft_q_pretrain_normalized'].mean()),
          pretrain_alpha_critic_normalized_bias=pretrain_errors.mean(axis=0),
          pretrain_alpha_critic_normalized_rmse=np.sqrt((pretrain_errors ** 2).mean(axis=0)),
          pretrain_alpha_critic_raw_equivalent_rmse=np.sqrt((pretrain_errors ** 2).mean(axis=0)) * divisor,
          alpha_convention='Both empirical targets exclude current-action entropy and terminal bootstrap; pretrain_alpha fields match .02/200 SoftSARSA, other fields compare initial online .05/200. Cross-alpha differences are not solely fitting error.',
          evaluation_rows_used_for_training=False if role.startswith('heldout') else None))


def evaluate(agent, ec, source, physics, protocol, label, final=False):
    saved = rng_state()
    before = {k: tensor_hashes(getattr(agent, k)) for k in ('actor', 'critics', 'target_critics')}
    counters = (agent.bc_updates, agent.pretrain_updates, agent.online_updates)
    dest = D / 'evaluations' / label
    dest.mkdir(parents=True, exist_ok=False)
    reports, heldout = [], []
    tasks = [('phase_deterministic', protocol['prefix_controls'], protocol['eval_deterministic_seed'], True)]
    tasks += [(f'phase_stochastic_{s}', protocol['prefix_controls'], s, False) for s in protocol['eval_phase_stochastic_seeds']]
    tasks += [('cold_deterministic', 0, protocol['eval_deterministic_seed'], True)]
    if final:
        tasks += [(f'cold_stochastic_{s}', 0, s, False) for s in protocol['eval_final_cold_stochastic_seeds']]
    if label in protocol['retention_evaluation_points']:
        tasks += [('retention8_deterministic', protocol['retention_prefix_controls'], protocol['eval_deterministic_seed'], True)]
        tasks += [(f'retention8_stochastic_{s}', protocol['retention_prefix_controls'], s, False)
                  for s in protocol['retention_stochastic_seeds']]
    try:
        for name, prefix, seed, deterministic in tasks:
            seed_all(seed)
            env = environment_for_policy(ec, protocol['policy_interface'])
            episode = None
            try:
                episode = Episode(env, prefix, source, physics, protocol['physical_reset_seed'], name, seed, protocol['raw_reward_divisor'])
                initial_rng = rng_state()
                while not (episode.term or episode.trunc):
                    episode.advance(command(agent, episode.obs, deterministic))
                extra = dict(policy_sampling_seed=seed, deterministic=deterministic,
                             learner_counters=dict(bc_updates=agent.bc_updates, pretrain_updates=agent.pretrain_updates, online_updates=agent.online_updates),
                             heldout_never_inserted_in_replay=True)
                episode.save(dest / name, extra)
                torch.save(initial_rng, dest / name / 'rng_initial.pt')
                torch.save(rng_state(), dest / name / 'rng_final.pt')
                reports.append(dict(episode.report(), **extra))
                if name.startswith('phase_stochastic'):
                    heldout.append(terminal_returns(episode, agent.config.discount,
                                   protocol['warmup_alpha_for_calibration'], protocol['raw_reward_divisor'], protocol['pretrain_alpha']))
            finally:
                if episode is not None:
                    episode.close()
                env.close()
    finally:
        restore_rng(saved)
    assert before == {k: tensor_hashes(getattr(agent, k)) for k in before}
    assert counters == (agent.bc_updates, agent.pretrain_updates, agent.online_updates)
    write(dest / 'summary.json', dict(reports=reports, training_rng_restored=True,
                                    learner_tensors_and_counters_unchanged=True, eligible_for_primitive=False))
    return concatenate_returns(heldout)


def save_checkpoint(label, agent, replay, episode, counts, protocol, ec):
    """Save after a complete physical transition and its complete update set."""
    parent = D / 'checkpoints'
    parent.mkdir(exist_ok=True)
    dest = parent / label
    temp = parent / (label + '.pending')
    temp.mkdir(exist_ok=False)
    agent.save(temp / 'agent.pt')
    metadata = write_checkpoint_metadata(temp / 'agent.pt', ec, agent.config,
                  stage='diagnostic-prefix-assisted-normalized-reward-sac', policy_interface=protocol['policy_interface'])
    metadata.update(learner_counters=dict(bc_updates=agent.bc_updates, pretrain_updates=agent.pretrain_updates, online_updates=agent.online_updates),
                    fixed_prefix_controls=protocol['prefix_controls'], reward_divisor=protocol['raw_reward_divisor'],
                    eligible_for_primitive=False, main_mpc_dataset_eligible=False,
                    frozen_checkpoint_collection=False, source_checkpoint_sha256=sha(D / 'initial.pt'))
    write(temp / 'agent.pt.metadata.json', metadata)
    replay.save(temp / 'replay.npz')
    torch.save(rng_state(), temp / 'rng.pt')
    state = dict(counts=counts, label=label, learner_counters=metadata['learner_counters'],
                 normalized_alpha=float(agent.alpha.detach()), raw_equivalent_alpha=float(agent.alpha.detach()) * protocol['raw_reward_divisor'],
                 replay_rows=len(replay), replay_pending_slot=replay._pending,
                 unfinished_episode=bool(episode is not None and not (episode.term or episode.trunc)),
                 automatic_resume_supported=False, complete_update_boundary=True)
    if episode is not None:
        env = episode.env
        (temp / 'environment.bin').write_bytes(env.data.__getstate__())
        np.savez_compressed(temp / 'environment.npz', observation=episode.obs.copy(), **boundary(env))
        state['environment_rng'] = env.np_random.bit_generator.state
        state['environment_scalars'] = {k: getattr(env, k) for k in ENV_SCALARS}
        episode.save(temp / 'current_episode', dict(checkpoint_label=label))
        observer = episode.observer
        state['observer_state'] = {k: v for k, v in vars(observer).items() if k not in ('env', 'original', 'rows')}
    metrics = D / 'metrics.jsonl'
    state['metrics_bytes'] = metrics.stat().st_size if metrics.exists() else 0
    state['metrics_sha256'] = sha(metrics) if metrics.exists() else hashlib.sha256(b'').hexdigest()
    write(temp / 'state.json', state)
    hashes = {str(p.relative_to(temp)): sha(p) for p in sorted(temp.rglob('*')) if p.is_file()}
    write(temp / 'manifest.json', hashes)
    # No learner/replay mutation during publication; leave incomplete generations
    # visibly pending on unexpected failure. Never fabricate a final transition.
    temp.rename(dest)
    write(D / 'latest.json', dict(checkpoint=str(dest.relative_to(D)), files=hashes, counts=counts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit('Prepared runner only. Execution requires separate --execute after root review.')
    if (D / 'runtime.json').exists():
        raise SystemExit('This generation already started; no overwrite or implicit restart')
    verify_manifest()
    protocol = json.loads((D / 'protocol.json').read_text())
    if protocol['status'] != 'frozen-ready-for-separate-explicit-execution' or type(protocol['prefix_controls']) is not int:
        raise RuntimeError('Protocol has not been finalized')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    runtime = dict(torch=str(torch.__version__), mujoco=mujoco.__version__, numpy=np.__version__,
                   python=sys.version, device=torch.cuda.get_device_name(), threads=1, interop_threads=1,
                   tf32=False, matmul_precision=torch.get_float32_matmul_precision(),
                   frozen_execution_manifest_sha256=sha(D / 'execution_manifest.json'))
    write(D / 'runtime.json', runtime)  # Saved before learner/model construction or training.
    ec = EnvConfig(**json.loads((D / 'config.json').read_text())['environment'])
    assert ec.task == 'rotation' and ec.object_diameter == .04 and ec.horizon_steps == 100 and ec.terminate_on_success
    metadata = validate_checkpoint(D / 'initial.pt', ec)
    assert metadata['policy_interface'] == protocol['policy_interface']
    agent = SACAgent.load(D / 'initial.pt', 'cuda')
    assert (agent.obs_dim, agent.action_dim, agent.bc_updates, agent.pretrain_updates, agent.online_updates) == (60, 16, 12000, 0, 0)
    assert agent.config.ensemble_size == 2 and agent.config.online_target_critics == 2
    assert agent.config.actor_warmup_updates == 5000 and agent.config.critic_lr == agent.config.actor_lr == 3e-4
    assert agent.config.initial_std == .05 and agent.std_initialized
    assert not agent.actor_optimizer.state and not agent.critic_optimizer.state and not agent.alpha_optimizer.state
    source_hashes = {k: tensor_hashes(getattr(agent, k)) for k in ('actor', 'critics', 'target_critics')}
    agent.config = replace(agent.config, initial_alpha=protocol['initial_alpha'], pretrain_alpha=protocol['pretrain_alpha'])
    with torch.no_grad():
        agent.log_alpha.fill_(math.log(protocol['initial_alpha']))
    agent.alpha_optimizer = torch.optim.Adam([agent.log_alpha], lr=agent.config.alpha_lr)
    assert source_hashes == {k: tensor_hashes(getattr(agent, k)) for k in source_hashes}
    write(D / 'resolved_learner.json', asdict(agent.config))
    with np.load(D / 'prefix_transitions.npz', allow_pickle=False) as z:
        source = {k: z[k].copy() for k in z.files}
    with np.load(D / 'prefix_physics.npz', allow_pickle=False) as z:
        assert tuple(z['columns']) == PHYSICS_COLUMNS
        physics = z['samples'].copy()
    assert len(source['actions']) == 9 and source['actions'].dtype == np.float32 and source['terminated'][-1]
    assert protocol['prefix_controls'] < len(source['actions'])
    seed_all(protocol['train_seed'])
    replay = ReplayBuffer(protocol['replay_capacity'], protocol['train_seed'])
    counts = dict(warmup_complete_episodes=0, warmup_policy_interactions=0,
                  online_policy_interactions=0, physical_prefix_controls=0,
                  warmup_assisted_completed_episodes=0, warmup_assisted_force_successes=0,
                  online_assisted_completed_episodes=0, online_assisted_force_successes=0,
                  warmup_cold_completed_episodes=0, warmup_cold_force_successes=0,
                  online_cold_completed_episodes=0, online_cold_force_successes=0,
                  new_pretrain_updates=0, new_online_updates=0,
                  warmup_raw_reward_sum=0., online_raw_reward_sum=0.)
    env = environment_for_policy(ec, protocol['policy_interface'])
    current, train_targets = None, []
    start = time.monotonic()
    metrics_stream = (D / 'metrics.jsonl').open('x')

    def log(event, **values):
        record = dict(event=event, elapsed_seconds=time.monotonic() - start, counts=copy.deepcopy(counts), **values)
        metrics_stream.write(json.dumps(jsonable(record), allow_nan=False) + '\n')
        metrics_stream.flush()

    def complete_episode(episode, phase):
        report = episode.report()
        category = 'assisted' if episode.prefix else 'cold'
        counts[phase + '_' + category + '_completed_episodes'] += 1
        counts[phase + '_' + category + '_force_successes'] += int(report['finger_supported_rotation_success'])
        episode.save(D / 'episodes' / phase / f'{episode.number:06d}',
                     dict(learner_online_updates=agent.online_updates, learner_pretrain_updates=agent.pretrain_updates))
        log('episode_complete', report=report)
        episode.close()

    try:
        save_checkpoint('initialized', agent, replay, None, counts, protocol, ec)
        for number in range(protocol['warmup_complete_episodes']):
            current = Episode(env, protocol['prefix_controls'], source, physics, protocol['physical_reset_seed'], 'warmup', number, protocol['raw_reward_divisor'])
            counts['physical_prefix_controls'] += current.prefix
            while not (current.term or current.trunc):
                row = current.advance(command(agent, current.obs))
                replay.add_transition(*(row[k] for k in REPLAY_KEYS))
                counts['warmup_policy_interactions'] += 1
                counts['warmup_raw_reward_sum'] += row['rewards']
            train_targets.append(terminal_returns(current, agent.config.discount, protocol['warmup_alpha_for_calibration'], protocol['raw_reward_divisor'], protocol['pretrain_alpha']))
            counts['warmup_complete_episodes'] += 1
            complete_episode(current, 'warmup')
            assert replay._pending is None
        assert tensor_hashes(agent.actor) == source_hashes['actor'] and agent.online_updates == agent.pretrain_updates == 0
        warmup = concatenate_returns(train_targets)
        np.savez_compressed(D / 'warmup_empirical_terminal_returns.npz', **warmup)
        save_checkpoint('warmup_complete', agent, replay, current, counts, protocol, ec)
        heldout = evaluate(agent, ec, source, physics, protocol, 'pretraining_baseline')
        calibrate(agent, warmup, D / 'warmup_calibration_before.npz', protocol['raw_reward_divisor'], 'training-warmup')
        calibrate(agent, heldout, D / 'heldout_calibration_before.npz', protocol['raw_reward_divisor'], 'heldout-phase-baseline')
        for _ in range(protocol['critic_pretrain_updates']):
            batch, stats = normalized_batch(replay, protocol)
            result = agent.pretrain_critic_step(batch)
            if result['pretrain_samples'] == 0:
                raise RuntimeError('Prescribed SoftSARSA update received no eligible actual successor or terminal rows')
            counts['new_pretrain_updates'] += 1
            log('critic_pretrain_update', **result, **stats)
        assert agent.pretrain_updates == 500 and agent.online_updates == 0
        assert tensor_hashes(agent.actor) == source_hashes['actor']
        assert not agent.actor_optimizer.state and not agent.alpha_optimizer.state
        assert abs(float(agent.alpha.detach()) - protocol['initial_alpha']) < 1e-9
        calibrate(agent, warmup, D / 'warmup_calibration_after.npz', protocol['raw_reward_divisor'], 'training-warmup')
        calibrate(agent, heldout, D / 'heldout_calibration_after.npz', protocol['raw_reward_divisor'], 'heldout-phase-baseline')
        save_checkpoint('pretrained', agent, replay, current, counts, protocol, ec)
        current, online_episode = None, 0
        for step in range(1, protocol['online_policy_interactions'] + 1):
            if current is None or current.term or current.trunc:
                current = Episode(env, protocol['prefix_controls'], source, physics, protocol['physical_reset_seed'], 'online', online_episode, protocol['raw_reward_divisor'])
                online_episode += 1
                counts['physical_prefix_controls'] += current.prefix
            row = current.advance(command(agent, current.obs))
            replay.add_transition(*(row[k] for k in REPLAY_KEYS))
            counts['online_policy_interactions'] += 1
            counts['online_raw_reward_sum'] += row['rewards']
            batch, stats = normalized_batch(replay, protocol)
            result = agent.update(batch)
            counts['new_online_updates'] += 1
            log('online_update', **result, **stats,
                raw_transition_reward=row['rewards'], normalized_transition_reward=row['normalized_rewards'],
                alpha_raw_equivalent=float(agent.alpha.detach()) * protocol['raw_reward_divisor'])
            assert agent.online_updates == counts['online_policy_interactions'] == step
            if current.term or current.trunc:
                complete_episode(current, 'online')
            if step in protocol['checkpoints']:
                save_checkpoint(f'online_{step}', agent, replay, current, counts, protocol, ec)
                evaluate(agent, ec, source, physics, protocol, f'online_{step}', final=step == protocol['online_policy_interactions'])
        if not (current.term or current.trunc):
            current.save(D / 'unfinished_suffix', dict(collection_budget_reached=True,
                         terminal_not_fabricated=True, replay_pending_preserved=True))
        assert agent.bc_updates == 12000 and agent.pretrain_updates == 500 and agent.online_updates == 2000
        verify_manifest()
        write(D / 'summary.json', dict(status='complete', counts=counts, runtime=runtime,
                    learner_counters=dict(bc_updates=agent.bc_updates, pretrain_updates=agent.pretrain_updates, online_updates=agent.online_updates),
                    unfinished_last_suffix=not bool(current.term or current.trunc),
                    fixed_prefix_controls=protocol['prefix_controls'], eligible_for_primitive=False,
                    main_mpc_dataset_eligible=False, trajectory_exported=False,
                    reward_divisor=protocol['raw_reward_divisor'], wall_seconds=time.monotonic() - start,
                    caveat=protocol['normalization_caveat']))
    except BaseException as error:
        write(D / 'failure.json', dict(type=type(error).__name__, message=str(error), counts=counts,
                                      current_state_not_published_as_committed=True))
        if current is not None and current.rows:
            current.save(D / 'failure_episode_evidence', dict(partially_updated_learner_not_checkpointed=True))
        raise
    finally:
        metrics_stream.close()
        if current is not None:
            current.close()
        env.close()
    write(D / 'manifest.json', {str(p.relative_to(D)): sha(p) for p in sorted(D.rglob('*'))
          if p.is_file() and '__pycache__' not in p.parts and p.name not in ('manifest.json', 'run.log')})


if __name__ == '__main__':
    main()
