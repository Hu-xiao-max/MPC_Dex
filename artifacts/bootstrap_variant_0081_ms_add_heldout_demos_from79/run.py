"""Bounded artifact-only prefix-assisted suffix SAC, parameterized by protocol.

Derived from the SAC0004 runner; never imports live core code. Protocol flags select
the collection prefix set (sampled per episode), the demonstration composition, the
demonstration mixing fraction, an optional actor BC term toward demonstration actions,
and whether the SAC actor loss covers demonstration states. Critic pretraining keeps
SAC0004's entropy-free demonstration target. Evaluations cover every prefix in
`eval_prefixes` plus cold reset at each checkpoint.

Execution is separate from preparation and requires --execute plus a finalized
execution_manifest.json. No resume is attempted after an interrupted run.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import copy
from dataclasses import asdict
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
import torch.nn.functional as F

from mpc_dex.bootstrap_env import environment_for_policy
from mpc_dex.contracts import validate_checkpoint, write_checkpoint_metadata
from mpc_dex.env import EnvConfig
from mpc_dex.learning import SACAgent
from mpc_dex.replay import ReplayBuffer, sample_mixed, _BATCH_KEYS
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
PREFIX_RNG = HANDOFF_RNG = None
DEMO_BUFFER = None


def replay_demo_rng_state():
    return None if DEMO_BUFFER is None else DEMO_BUFFER.rng.bit_generator.state
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


def assert_tree_equal(first, second):
    """Exact inherited optimizer/network comparison without an update."""
    if isinstance(first, torch.Tensor):
        assert isinstance(second, torch.Tensor) and torch.equal(first.detach().cpu(), second.detach().cpu())
    elif isinstance(first, np.ndarray):
        assert isinstance(second, np.ndarray) and first.dtype == second.dtype and np.array_equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_tree_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert type(first) is type(second) and len(first) == len(second)
        for left, right in zip(first, second):
            assert_tree_equal(left, right)
    else:
        assert first == second


def optimizer_steps(optimizer):
    return sorted({int(state['step']) for state in optimizer.state.values()})


def demonstration_gate(demo, protocol):
    """Learning is permitted by verified off-policy positives, never by warm-up successes."""
    rows = demo.transitions()
    assert len(demo) == protocol['demonstration_rows'] and demo.episode_count == protocol['demonstration_episodes']
    assert int(rows['terminated'].sum()) == protocol['demonstration_episodes'] and not rows['truncated'].any()
    assert demo._pending is None and rows['next_action_valid'].sum() == len(demo) - protocol['demonstration_episodes']
    return True


MPC_HANDOFF_PREFIX = 6  # The MPC recovery branched after exactly six recorded controls.


def load_demonstrations(protocol):
    """Verified positive episodes: optional MPC suffix, plus recorded source rows [start, 9)."""
    keys = ('obs', 'actions', 'rewards', 'next_obs', 'terminated', 'truncated')
    with np.load(D / 'prefix_transitions.npz', allow_pickle=False) as z:
        source = {k: z[k].copy() for k in z.files}
    with np.load(D / 'mpc_correction_pairs.npz', allow_pickle=False) as z:
        pairs = {k: z[k].copy() for k in z.files}
    with np.load(D / 'mpc_recovery_transitions.npz', allow_pickle=False) as z:
        recovery = {k: z[k].copy() for k in z.files}
    p = MPC_HANDOFF_PREFIX
    assert all(np.array_equal(pairs[k], recovery[k][p:]) for k in keys)
    assert np.array_equal(pairs['source_control_1based'], np.arange(p + 1, len(recovery['actions']) + 1))
    assert np.array_equal(recovery['actions'][:p], source['actions'][:p]) and np.array_equal(recovery['obs'][:p], source['obs'][:p])
    start, end = protocol['demonstration_recorded_rows']
    assert 0 <= start < end == len(source['actions']) == 9
    mpc = {k: pairs[k].copy() for k in keys}
    recorded = {k: source[k][start:end].copy() for k in keys}
    for arrays in (mpc, recorded):
        assert arrays['terminated'][-1] and not arrays['terminated'][:-1].any() and not arrays['truncated'].any()
        assert np.array_equal(arrays['next_obs'][:-1], arrays['obs'][1:]) and np.abs(arrays['actions']).max() <= 1
        assert arrays['actions'].dtype == np.float32 and arrays['obs'].shape[1] == 60
    assert np.array_equal(mpc['obs'][0], source['obs'][p])
    demonstrations = []
    if protocol['demonstration_include_mpc']:
        assert len(mpc['obs']) == protocol['demonstration_rows_mpc']
        demonstrations.append(dict(name='mpc', arrays=mpc, provenance=dict(origin='actual_corrected_native_mpc_controls_7_to_11', file='mpc_correction_pairs.npz',
                              sha256=sha(D / 'mpc_correction_pairs.npz'), rows=len(mpc['obs']), raw_return=float(mpc['rewards'].sum()))))
    assert len(recorded['obs']) == protocol['demonstration_rows_recorded']
    demonstrations.append(dict(name='recorded', arrays=recorded, provenance=dict(origin=f'recorded_source_controls_{start + 1}_to_{end}', file='prefix_transitions.npz',
                          sha256=sha(D / 'prefix_transitions.npz'), rows=len(recorded['obs']), raw_return=float(recorded['rewards'].sum()))))
    assert sum(len(item['arrays']['obs']) for item in demonstrations) == protocol['demonstration_rows']
    assert len(demonstrations) == protocol['demonstration_episodes']
    return demonstrations


def sample_prefix(prefix_rng, protocol):
    """Per-episode physical prefix length, drawn from the protocol set with its own RNG."""
    prefix_set = protocol['prefix_set']
    assert all(isinstance(p, int) and 0 <= p <= 8 for p in prefix_set) and prefix_set
    weights = protocol.get('prefix_weights')
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        assert weights.shape == (len(prefix_set),) and (weights > 0).all()
        return int(prefix_rng.choice(np.asarray(prefix_set, dtype=np.int64), p=weights / weights.sum()))
    return int(prefix_rng.choice(np.asarray(prefix_set, dtype=np.int64)))


def handoff_perturbation(prefix, protocol, rng, source):
    """Recorded control `prefix` plus N(0, sigma^2) in action space, or None when not perturbed."""
    sigma = protocol.get('handoff_noise_sigma', 0.)
    if sigma <= 0 or prefix < 1 or prefix not in protocol.get('handoff_noise_prefixes', []):
        return None
    noise = rng.normal(0., sigma, size=16)
    return np.clip(source['actions'][prefix - 1].astype(np.float64) + noise, -1., 1.).astype(np.float32)


def prefix_key(prefix, noisy):
    return f'{prefix}+noise' if noisy else str(prefix)


def sample_suffix(rng, protocol):
    """Suffix length for a multi-source episode: uniform over suffix_set, or weighted by suffix_weights."""
    suffixes = np.asarray(protocol['suffix_set'], dtype=np.int64)
    weights = protocol.get('suffix_weights')
    if weights is None:
        return int(rng.choice(suffixes))
    weights = np.asarray(weights, dtype=np.float64)
    assert weights.shape == suffixes.shape and (weights > 0).all()
    return int(rng.choice(suffixes, p=weights / weights.sum()))


def load_sources(protocol):
    """Verified demonstration trajectories usable as physically replayed prefixes (multi-source mode)."""
    sources = {}
    for item in protocol.get('prefix_sources', []):
        with np.load(D / item['transitions'], allow_pickle=False) as z:
            src = {k: z[k].copy() for k in z.files}
        with np.load(D / item['physics'], allow_pickle=False) as z:
            assert tuple(z['columns']) == PHYSICS_COLUMNS
            phys = z['samples'].copy()
        assert src['terminated'][-1] and not src['terminated'][:-1].any() and not src['truncated'].any()
        assert src['actions'].dtype == np.float32 and src['obs'].shape[1] == 60 and len(phys) == len(src['actions']) * 50
        assert sha(D / item['transitions']) == item['sha256']
        origin = src.get('control_origin')  # DAgger-labelled episodes: 'demo_prefix' | 'policy' | 'mpc' per control
        if origin is not None:
            assert len(origin) == len(src['actions']) and set(origin.tolist()) <= {'demo_prefix', 'policy', 'mpc'}
            assert item.get('policy_rows') == int((origin == 'policy').sum()) and origin[-1] == 'mpc'
        else:
            assert not item.get('policy_rows')
        sources[item['name']] = dict(source=src, physics=phys, length=int(len(src['actions'])), origin=origin)
    return sources


def multisource_demonstrations(protocol, sources):
    """The last `demonstration_suffix_rows` rows of every source, each a terminal-success episode."""
    keys = ('obs', 'actions', 'rewards', 'next_obs', 'terminated', 'truncated')
    m = int(protocol['demonstration_suffix_rows'])
    out = []
    for name in protocol['prefix_source_names']:
        s = sources[name]['source']; n = sources[name]['length']
        rows = np.arange(max(0, n - m), n)  # whole demonstration when it is shorter than m
        origin = sources[name].get('origin'); dropped = 0
        if origin is not None and (origin == 'policy').any():
            # DAgger episode: the learner's own controls are states to label, never targets. Keep the contiguous
            # MPC segment after the last policy control (its first row is the label at the policy-visited state).
            start = int(np.flatnonzero(origin == 'policy')[-1]) + 1; dropped = int((rows < start).sum()); rows = rows[rows >= start]
            assert (origin[rows] == 'mpc').all() and len(rows) >= 1
        arrays = {k: s[k][rows].copy() for k in keys}
        assert arrays['terminated'][-1] and np.array_equal(arrays['next_obs'][:-1], arrays['obs'][1:])
        out.append(dict(name=name, arrays=arrays, provenance=dict(origin='verified_cold_reset_mpc_demonstration_suffix' if origin is None else 'dagger_mpc_labels_at_policy_visited_states', source=name, rows=int(len(rows)),
                        rows_before_mpc_segment_excluded=dropped, raw_return=float(arrays['rewards'].sum()), first_row_is_handoff_for_suffix=int(len(rows)))))
    return out


def episode_key(episode):
    if getattr(episode, 'suffix', None) is not None:
        return f's{episode.suffix}' + ('+noise' if episode.noisy_last is not None else '')
    return prefix_key(episode.prefix, episode.noisy_last is not None)


def multisource_keys(protocol):
    noisy = set(protocol.get('handoff_noise_suffixes', [])) if protocol.get('handoff_noise_sigma', 0.) > 0 else set()
    return [f's{s}' + ('+noise' if s in noisy else '') for s in protocol['suffix_set']]


def prefix_keys(protocol):
    noisy = set(protocol.get('handoff_noise_prefixes', [])) if protocol.get('handoff_noise_sigma', 0.) > 0 else set()
    return [prefix_key(p, p in noisy) for p in protocol['prefix_set']]


def new_episode(env, protocol, source, physics, phase, number, prefix_rng, handoff_rng, sources=None):
    if sources:
        names = protocol['prefix_source_names']
        name = names[int(prefix_rng.integers(len(names)))]
        suffix = sample_suffix(prefix_rng, protocol)
        s = sources[name]; prefix = max(0, s['length'] - suffix)  # a suffix longer than the demonstration is a cold-start episode
        noisy_last = None
        if prefix >= 1 and protocol.get('handoff_noise_sigma', 0.) > 0 and suffix in protocol.get('handoff_noise_suffixes', []):
            noisy_last = handoff_perturbation(prefix, dict(protocol, handoff_noise_prefixes=[prefix]), handoff_rng, s['source'])
        episode = Episode(env, prefix, s['source'], s['physics'], protocol['physical_reset_seed'], phase, number, protocol['raw_reward_divisor'], noisy_last)
        episode.source_name, episode.suffix = name, suffix
        return episode
    prefix = sample_prefix(prefix_rng, protocol)
    noisy_last = handoff_perturbation(prefix, protocol, handoff_rng, source)
    return Episode(env, prefix, source, physics, protocol['physical_reset_seed'], phase, number, protocol['raw_reward_divisor'], noisy_last)


def multisource_evaluation_tasks(protocol, sources, final):
    """(name, prefix, seed, deterministic, noisy, source_name) over eval sources and suffix lengths."""
    tasks = []
    for name in protocol['eval_source_names']:
        n = sources[name]['length']
        for s in protocol['suffix_set']:
            prefix = max(0, n - s)
            tasks.append((f'src_{name}_s{s}_deterministic', prefix, protocol['eval_deterministic_seed'], True, False, name))
            tasks += [(f'src_{name}_s{s}_stochastic_{seed}', prefix, seed, False, False, name) for seed in protocol['eval_stochastic_seeds_by_suffix'][str(s)]]
            if prefix >= 1 and protocol.get('handoff_noise_sigma', 0.) > 0 and s in protocol.get('handoff_noise_suffixes', []):
                tasks += [(f'src_{name}_s{s}_noisy_{seed}', prefix, seed, False, True, name) for seed in protocol['eval_noisy_seeds_by_suffix'][str(s)]]
    return tasks


def evaluation_tasks(protocol, label, final):
    """(name, prefix, seed, deterministic, noisy) for every evaluated prefix plus cold reset."""
    tasks = []
    for prefix in protocol['eval_prefixes']:
        assert prefix >= 1, 'cold reset is evaluated separately'
        tasks.append((f'prefix{prefix}_deterministic', prefix, protocol['eval_deterministic_seed'], True, False))
        tasks += [(f'prefix{prefix}_stochastic_{s}', prefix, s, False, False) for s in protocol['eval_stochastic_seeds_by_prefix'][str(prefix)]]
    for prefix in protocol.get('eval_noisy_prefixes', []):
        assert protocol.get('handoff_noise_sigma', 0.) > 0 and prefix in protocol['handoff_noise_prefixes']
        tasks += [(f'prefix{prefix}_noisy_{s}', prefix, s, False, True) for s in protocol['eval_noisy_seeds_by_prefix'][str(prefix)]]
    tasks.append(('cold_deterministic', 0, protocol['eval_deterministic_seed'], True, False))
    if final or protocol.get('eval_cold_stochastic_every_point', False):
        tasks += [(f'cold_stochastic_{s}', 0, s, False, False) for s in protocol['eval_final_cold_stochastic_seeds']]
    return tasks


def runner_update(agent, batch, protocol, demonstrations):
    """SACAgent.update replicated call-for-call, with two protocol-controlled additions.

    actor_on_demonstration_states=False restricts the SAC actor and temperature losses to
    on-policy rows of the batch. actor_bc_lambda>0 adds lambda * MSE(actor mean, atanh of the
    clipped demonstration action) over all demonstration rows to the same single actor step.
    With actor_on_demonstration_states=True and actor_bc_lambda=0 the parameter updates and
    RNG consumption are identical to the archived method.
    """
    values = agent._batch(batch, ('obs', 'actions', 'rewards', 'next_obs', 'terminated'))
    demonstration = torch.as_tensor(np.asarray(batch['demonstration'], dtype=bool), device=agent.device)
    with torch.no_grad():
        targets = values['rewards'].clone()
        bootstrap = ~values['terminated'].bool()
        if bootstrap.any():
            next_actions, next_log_prob = agent.actor.sample(values['next_obs'][bootstrap])
            selected = torch.randperm(agent.config.ensemble_size, device=agent.device)[:agent.config.online_target_critics]
            next_q = agent.target_critics(values['next_obs'][bootstrap], next_actions)[selected].min(dim=0).values
            targets[bootstrap] += agent.config.discount * (next_q - agent.alpha * next_log_prob)
    critic_loss = agent._critic_step(values, targets)
    agent.critics.requires_grad_(False)
    try:
        actions, log_prob = agent.actor.sample(values['obs'])
        estimates = agent.critics(values['obs'], actions)
        q = estimates.mean(dim=0) if agent.config.actor_critic_reduction == 'mean' else estimates.min(dim=0).values
        per_row = agent.alpha.detach() * log_prob - q
        if protocol['actor_on_demonstration_states']:
            actor_loss = per_row.mean()
            entropy_rows = log_prob
        else:
            online = ~demonstration
            assert online.any(), 'batch without on-policy rows'
            actor_loss = per_row[online].mean()
            entropy_rows = log_prob[online]
        bc_loss = None
        if protocol['actor_bc_lambda'] > 0:
            mode = protocol.get('actor_bc_rows', 'all')
            rows = slice(0, 1) if mode == 'first' else slice(0, int(protocol['actor_bc_first_rows'])) if mode == 'first_k' else slice(None)
            demo_obs = torch.as_tensor(np.concatenate([item['arrays']['obs'][rows] for item in demonstrations]), device=agent.device)
            demo_actions = torch.as_tensor(np.concatenate([item['arrays']['actions'][rows] for item in demonstrations]), device=agent.device)
            epsilon = agent.config.bc_action_epsilon
            bc_target = torch.atanh(demo_actions.clamp(-1 + epsilon, 1 - epsilon))
            demo_mean, demo_log_std = agent.actor(demo_obs)
            bc_loss = F.mse_loss(demo_mean, bc_target)
            actor_loss = actor_loss + protocol['actor_bc_lambda'] * bc_loss
            if protocol.get('actor_bc_std_target') is not None:
                # Spread anchor at the supervised rows: pulls the latent log-std toward log(target) so the
                # sampled handoff control stays inside the narrow success basin around the label.
                std_loss = F.mse_loss(demo_log_std, torch.full_like(demo_log_std, math.log(float(protocol['actor_bc_std_target']))))
                actor_loss = actor_loss + float(protocol.get('actor_bc_std_lambda', protocol['actor_bc_lambda'])) * std_loss
        warmup = agent.config.actor_warmup_updates
        fraction = min((agent.online_updates + 1) / warmup, 1.) if warmup else 1.
        actor_lr = agent.config.actor_lr * fraction
        for group in agent.actor_optimizer.param_groups:
            group['lr'] = actor_lr
        agent.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        agent.actor_optimizer.step()
    finally:
        agent.critics.requires_grad_(True)
    alpha_loss = -(agent.log_alpha * (entropy_rows.detach() + agent.config.target_entropy)).mean()
    if protocol.get('alpha_freeze', False):
        # Temperature held at its stage value: the loss is computed for logging only; no optimizer step, no state change.
        alpha_loss = alpha_loss.detach()
    else:
        agent.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        agent.alpha_optimizer.step()
    agent._polyak()
    agent.online_updates += 1
    return {'critic_loss': float(critic_loss), 'actor_loss': float(actor_loss.detach()),
            'alpha_loss': float(alpha_loss.detach()), 'alpha': float(agent.alpha.detach()),
            'actor_bc_std_loss': None if bc_loss is None or protocol.get('actor_bc_std_target') is None else float(std_loss.detach()),
            'actor_lr': actor_lr, 'entropy': float(-entropy_rows.detach().mean()),
            'entropy_all_rows': float(-log_prob.detach().mean()),
            'target_mean': float(targets.mean()), 'online_updates': agent.online_updates,
            'actor_bc_loss': None if bc_loss is None else float(bc_loss.detach()),
            'actor_bc_rows': None if bc_loss is None else int(len(demo_obs)),
            'actor_rows_used': int(len(entropy_rows))}


@torch.no_grad()
def calibrate_demonstrations(agent, demonstrations, path, protocol, source_handoff_obs):
    """Critic values on the fixed rows versus entropy-free discounted returns; actor distance to each episode's first action."""
    out = dict(alpha_convention='targets are discounted raw returns / divisor without entropy terms; not the SAC backup',
               learner_counters=dict(bc_updates=agent.bc_updates, pretrain_updates=agent.pretrain_updates, online_updates=agent.online_updates),
               episodes={})
    for item in demonstrations:
        a = item['arrays']
        target = a['rewards'].astype(np.float64) / protocol['raw_reward_divisor']
        for i in range(len(target) - 2, -1, -1):
            target[i] += agent.config.discount * target[i + 1]
        obs = torch.as_tensor(a['obs'], device=agent.device)
        actions = torch.as_tensor(a['actions'], device=agent.device)
        q = agent.critics(obs, actions).cpu().numpy().reshape(agent.config.ensemble_size, len(target))
        mean, log_std = agent.actor(obs[:1])
        deterministic = mean.tanh()[0].cpu().numpy()
        out['episodes'][item['name']] = dict(rows=len(target), target_normalized=target.tolist(),
            q_normalized=q.tolist(), critic_normalized_bias=(q - target[None]).mean(axis=1).tolist(),
            critic_normalized_rmse=np.sqrt(((q - target[None]) ** 2).mean(axis=1)).tolist(),
            first_row_deterministic_action_max_abs_difference=float(np.abs(deterministic - a['actions'][0]).max()),
            first_row_deterministic_action_l2_difference=float(np.linalg.norm(deterministic - a['actions'][0])),
            first_row_latent_std_mean=float(log_std.exp().mean()),
            first_row_is_prefix6_handoff=bool(np.array_equal(a['obs'][0], source_handoff_obs)),
            actor_log_prob_of_demonstration_actions=agent.actor.log_prob(obs, actions).cpu().numpy().tolist())
    write(path, out)


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
    def __init__(self, env, prefix, source, source_physics, seed, phase, number, divisor, noisy_last=None):
        self.env, self.prefix, self.phase, self.number, self.divisor = env, prefix, phase, number, divisor
        self.obs, self.info = env.reset(seed=seed)
        self.seed = seed
        self.initial = dict(obs=self.obs.copy(), **boundary(env))
        self.initial_rng = rng_state()
        self.observer = PhysicsObserver(env)
        self.rows, self.closed = [], False
        self.term = self.trunc = False
        self.noisy_last = None if noisy_last is None else np.asarray(noisy_last, dtype=np.float32).copy()
        self.handoff_noise_l2 = 0.
        self.handoff_terminated_before_policy = False
        exact = prefix - 1 if self.noisy_last is not None else prefix
        assert exact >= 0
        for index in range(exact):
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
        if exact:
            np.testing.assert_array_equal(np.asarray(self.observer.rows), source_physics[:exact * env.substeps])
        if self.noisy_last is not None:
            # Perturbed final prefix control: a non-policy row executed from the bit-exact recorded
            # state before it. Never enters replay; the resulting handoff state is off-source by design.
            assert prefix >= 1 and self.noisy_last.shape == (16,) and np.isfinite(self.noisy_last).all() and np.abs(self.noisy_last).max() <= 1
            np.testing.assert_array_equal(self.obs, source['obs'][prefix - 1])
            self.handoff_noise_l2 = float(np.linalg.norm(self.noisy_last - source['actions'][prefix - 1]))
            self.advance(dict(action=self.noisy_last.copy(), log_prob=np.nan,
                              raw_mean=np.full(16, np.nan, np.float32), latent_std=np.full(16, np.nan, np.float32),
                              density_is_stochastic_sample=False, learner_bc_updates=-1,
                              learner_pretrain_updates=-1, learner_online_updates=-1), policy=False)
            self.handoff_terminated_before_policy = bool(self.term or self.trunc)
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
                    prefix_physics_bitexact=True, exact_prefix_controls=self.prefix - int(self.noisy_last is not None),
                    perturbed_last_prefix_control=self.noisy_last is not None, handoff_noise_l2=self.handoff_noise_l2,
                    prefix_source=getattr(self, 'source_name', 'recorded'), policy_suffix_target=getattr(self, 'suffix', None),
                    handoff_terminated_before_policy=self.handoff_terminated_before_policy,
                    fixed_prefix_assisted=bool(self.prefix),
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


def mixed_batch_with_origin(demo, replay, batch_size, offline_fraction):
    """Exactly sample_mixed's draws and permutation, plus a per-row demonstration mask."""
    if batch_size <= 0 or not 0 <= offline_fraction <= 1:
        raise ValueError('invalid batch size or offline fraction')
    offline_count = int(np.floor(batch_size * offline_fraction + .5))
    online_count = batch_size - offline_count
    batches, origin = [], []
    if offline_count:
        batches.append(demo.sample(offline_count))
        origin.append(np.ones(offline_count, dtype=bool))
    if online_count:
        batches.append(replay.sample(online_count))
        origin.append(np.zeros(online_count, dtype=bool))
    merged = {key: np.concatenate([batch[key] for batch in batches]) for key in _BATCH_KEYS}
    merged['demonstration'] = np.concatenate(origin)
    order = replay.rng.permutation(batch_size)
    return {key: value[order] for key, value in merged.items()}


def entropy_masked_pretrain_step(agent, batch, protocol):
    """Archived Soft-SARSA step, except demonstration rows bootstrap without the entropy term.

    On-policy rows: r + discount * (min Q'(s', a'_recorded) - pretrain_alpha * log pi(a'|s')).
    Demonstration rows: r + discount * min Q'(s', a'_recorded). Same eligibility rule,
    critic step, Polyak update and counter as SACAgent.pretrain_critic_step.
    """
    if not agent.std_initialized:
        raise RuntimeError('call initialize_std() after BC before Soft-SARSA')
    values = agent._batch(batch, ('obs', 'actions', 'rewards', 'next_obs', 'terminated', 'next_actions', 'next_action_valid'))
    demonstration = torch.as_tensor(np.asarray(batch['demonstration'], dtype=bool), device=agent.device)
    assert demonstration.shape == values['terminated'].shape
    eligible = values['terminated'].bool() | values['next_action_valid'].bool()
    count = int(eligible.sum())
    if count == 0:
        return {'critic_loss': 0.0, 'pretrain_samples': 0, 'pretrain_updates': agent.pretrain_updates,
                'demonstration_rows_eligible': 0, 'entropy_free_bootstrap_rows': 0}
    values = {key: value[eligible] for key, value in values.items()}
    demonstration = demonstration[eligible]
    with torch.no_grad():
        targets = values['rewards'].clone()
        bootstrap = ~values['terminated'].bool()
        entropy_free_rows = 0
        if bootstrap.any():
            next_obs, next_actions = values['next_obs'][bootstrap], values['next_actions'][bootstrap]
            next_q = agent.target_critics(next_obs, next_actions).min(dim=0).values
            log_prob = agent.actor.log_prob(next_obs, next_actions)
            entropy = torch.where(demonstration[bootstrap], torch.zeros_like(log_prob), -agent.config.pretrain_alpha * log_prob)
            entropy_free_rows = int(demonstration[bootstrap].sum())
            targets[bootstrap] += agent.config.discount * (next_q + entropy)
    loss = agent._critic_step(values, targets)
    agent._polyak()
    agent.pretrain_updates += 1
    return {'critic_loss': float(loss), 'target_mean': float(targets.mean()), 'pretrain_samples': count,
            'pretrain_updates': agent.pretrain_updates, 'demonstration_rows_eligible': int(demonstration.sum()),
            'entropy_free_bootstrap_rows': entropy_free_rows,
            'demonstration_target_mean': float(targets[demonstration].mean()) if demonstration.any() else None}


def normalized_batch(demo, replay, protocol, phase='online'):
    # Draws copy rows; both raw buffers and their episode returns stay intact.
    batch = mixed_batch_with_origin(demo, replay, protocol['batch_size'], protocol[f'{phase}_offline_fraction'])
    raw = batch['rewards'].copy()
    batch['rewards'] = raw / np.float32(protocol['raw_reward_divisor'])
    return batch, dict(batch_raw_reward_mean=float(raw.mean()),
                       batch_normalized_reward_mean=float(batch['rewards'].mean()),
                       batch_demonstration_rows=int(batch['demonstration'].sum()))


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
def calibrate(agent, samples, path, divisor, role, online_alpha, pretrain_alpha):
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
          inherited_online_alpha_normalized=online_alpha,
          pretrain_alpha_normalized=pretrain_alpha,
          pretrain_target_normalized_mean=float(samples['soft_q_pretrain_normalized'].mean()),
          pretrain_alpha_critic_normalized_bias=pretrain_errors.mean(axis=0),
          pretrain_alpha_critic_normalized_rmse=np.sqrt((pretrain_errors ** 2).mean(axis=0)),
          pretrain_alpha_critic_raw_equivalent_rmse=np.sqrt((pretrain_errors ** 2).mean(axis=0)) * divisor,
          alpha_convention='Both empirical targets exclude current-action entropy and terminal bootstrap; pretrain_alpha fields match inherited SoftSARSA config, other fields use frozen inherited learned online alpha. Cross-alpha differences are not solely fitting error.',
          evaluation_rows_used_for_training=False if role.startswith('heldout') else None))


def evaluate(agent, ec, source, physics, protocol, label, final=False, sources=None):
    saved = rng_state()
    before = {k: tensor_hashes(getattr(agent, k)) for k in ('actor', 'critics', 'target_critics')}
    counters = (agent.bc_updates, agent.pretrain_updates, agent.online_updates)
    dest = D / 'evaluations' / label
    dest.mkdir(parents=True, exist_ok=False)
    torch.save(saved, dest / 'training_rng_before.pt')
    reports, heldout = [], []
    tasks = [(*t, None) for t in evaluation_tasks(protocol, label, final)]
    if sources:
        tasks = multisource_evaluation_tasks(protocol, sources, final) + tasks
    heldout_prefix = f"prefix{protocol['prefix_controls']}_stochastic"
    try:
        for name, prefix, seed, deterministic, noisy, source_name in tasks:
            seed_all(seed)
            env = environment_for_policy(ec, protocol['policy_interface'])
            episode = None
            try:
                task_source, task_physics = (sources[source_name]['source'], sources[source_name]['physics']) if source_name else (source, physics)
                noisy_last = handoff_perturbation(prefix, dict(protocol, handoff_noise_prefixes=[prefix]) if source_name else protocol, np.random.default_rng(seed), task_source) if noisy else None
                assert (noisy_last is not None) == noisy
                episode = Episode(env, prefix, task_source, task_physics, protocol['physical_reset_seed'], name, seed, protocol['raw_reward_divisor'], noisy_last)
                if source_name:
                    episode.source_name, episode.suffix = source_name, sources[source_name]['length'] - prefix
                initial_rng = rng_state()
                while not (episode.term or episode.trunc):
                    episode.advance(command(agent, episode.obs, deterministic))
                extra = dict(policy_sampling_seed=seed, deterministic=deterministic, perturbed_handoff=noisy,
                             handoff_noise_seed=seed if noisy else None,
                             learner_counters=dict(bc_updates=agent.bc_updates, pretrain_updates=agent.pretrain_updates, online_updates=agent.online_updates),
                             heldout_never_inserted_in_replay=True)
                episode.save(dest / name, extra)
                torch.save(initial_rng, dest / name / 'rng_initial.pt')
                torch.save(rng_state(), dest / name / 'rng_final.pt')
                reports.append(dict(episode.report(), **extra))
                if name.startswith(heldout_prefix):
                    heldout.append(terminal_returns(episode, agent.config.discount,
                                   protocol['warmup_alpha_for_calibration'], protocol['raw_reward_divisor'], protocol['pretrain_alpha']))
            finally:
                if episode is not None:
                    episode.close()
                env.close()
    finally:
        restore_rng(saved)
        restored = rng_state()
        torch.save(restored, dest / 'training_rng_after.pt')
        assert_tree_equal(saved, restored)
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
                    inherited_learner_counters=protocol['inherited_counters'],
                    new_learner_updates=dict(bc_updates=0, pretrain_updates=counts['new_pretrain_updates'], online_updates=counts['new_online_updates']),
                    fixed_prefix_controls=None, prefix_set=protocol['prefix_set'], primary_prefix_controls=protocol['prefix_controls'], reward_divisor=protocol['raw_reward_divisor'],
                    eligible_for_primitive=False, main_mpc_dataset_eligible=False,
                    frozen_checkpoint_collection=False, source_checkpoint_sha256=sha(D / 'initial.pt'),
                    demonstration_mixing=dict(pretrain_offline_fraction=protocol['pretrain_offline_fraction'], online_offline_fraction=protocol['online_offline_fraction'], demonstration_rows=protocol['demonstration_rows'], demonstration_replay_sha256=sha(D / 'demonstrations_replay.npz'), pretraining_demonstration_target='entropy-free', actor_bc_lambda=protocol['actor_bc_lambda'], actor_bc_rows=protocol.get('actor_bc_rows', 'all'), actor_on_demonstration_states=protocol['actor_on_demonstration_states'], handoff_noise_sigma=protocol.get('handoff_noise_sigma', 0.), handoff_noise_prefixes=protocol.get('handoff_noise_prefixes', [])))
    write(temp / 'agent.pt.metadata.json', metadata)
    replay.save(temp / 'replay.npz')
    torch.save(rng_state(), temp / 'rng.pt')
    state = dict(counts=counts, label=label, learner_counters=metadata['learner_counters'],
                 normalized_alpha=float(agent.alpha.detach()), raw_equivalent_alpha=float(agent.alpha.detach()) * protocol['raw_reward_divisor'],
                 replay_rows=len(replay), replay_pending_slot=replay._pending,
                 unfinished_episode=bool(episode is not None and not (episode.term or episode.trunc)),
                 inherited_learner_counters=protocol['inherited_counters'],
                 optimizer_steps={name: optimizer_steps(getattr(agent, name + '_optimizer')) for name in ('critic', 'actor', 'alpha')},
                 replay_policy=f"fresh-stage-only online replay plus immutable {protocol['demonstration_rows']}-row demonstration buffer (demonstrations_replay.npz); pretraining batches {protocol['pretrain_offline_fraction']:.2f} / online batches {protocol['online_offline_fraction']:.2f} demonstration share; inherited open replay archived without insertion",
                 auxiliary_rng_states=dict(prefix_rng=PREFIX_RNG.bit_generator.state if PREFIX_RNG is not None else None,
                                           demonstration_rng=replay_demo_rng_state(), handoff_rng=HANDOFF_RNG.bit_generator.state if HANDOFF_RNG is not None else None),
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
    if protocol['status'] != 'frozen-ready-for-separate-explicit-execution' or type(protocol['prefix_controls']) is not int or not protocol['prefix_set']:
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
    inherited = protocol['inherited_counters']
    assert (agent.obs_dim, agent.action_dim) == (60, 16)
    assert dict(bc_updates=agent.bc_updates, pretrain_updates=agent.pretrain_updates, online_updates=agent.online_updates) == inherited
    assert agent.config.ensemble_size == 2 and agent.config.online_target_critics == 2
    inherited_actor_lr = float(protocol.get('inherited_actor_lr', 3e-4))
    assert agent.config.actor_warmup_updates == 5000 and agent.config.critic_lr == 3e-4 and agent.config.actor_lr == inherited_actor_lr
    assert agent.config.initial_std == .05 and agent.std_initialized  # Learned std weights remain untouched.
    for name, expected in protocol['inherited_optimizer_steps'].items():
        assert optimizer_steps(getattr(agent, name + '_optimizer')) == [expected]
    assert float(agent.alpha.detach()) == protocol['inherited_learned_alpha']
    assert agent.config.pretrain_alpha == protocol['pretrain_alpha']
    source_hashes = {k: tensor_hashes(getattr(agent, k)) for k in ('actor', 'critics', 'target_critics')}
    source_actor_optimizer = copy.deepcopy(agent.actor_optimizer.state_dict())
    source_alpha_optimizer = copy.deepcopy(agent.alpha_optimizer.state_dict())
    source_log_alpha = agent.log_alpha.detach().clone()
    write(D / 'resolved_learner.json', asdict(agent.config))
    with np.load(D / 'prefix_transitions.npz', allow_pickle=False) as z:
        source = {k: z[k].copy() for k in z.files}
    with np.load(D / 'prefix_physics.npz', allow_pickle=False) as z:
        assert tuple(z['columns']) == PHYSICS_COLUMNS
        physics = z['samples'].copy()
    assert len(source['actions']) == 9 and source['actions'].dtype == np.float32 and source['terminated'][-1]
    assert protocol['prefix_controls'] < len(source['actions']) and max(protocol['prefix_set']) < len(source['actions'])
    global PREFIX_RNG, HANDOFF_RNG, DEMO_BUFFER
    prefix_rng = PREFIX_RNG = np.random.default_rng(protocol['prefix_seed'])
    handoff_rng = HANDOFF_RNG = np.random.default_rng(protocol['handoff_noise_seed'])
    seed_all(protocol['train_seed'])
    replay = ReplayBuffer(protocol['replay_capacity'], protocol['train_seed'])
    assert len(replay) == 0 and replay._pending is None
    sources = load_sources(protocol)
    if sources:
        # Training sources are a subset of the loaded sources; held-out sources appear only in evaluations.
        assert set(protocol['prefix_source_names']) <= set(sources) and set(protocol['eval_source_names']) <= set(sources)
        assert not (set(protocol.get('holdout_source_names', [])) & set(protocol['prefix_source_names']))
        demonstrations = multisource_demonstrations(protocol, sources)
    else:
        demonstrations = load_demonstrations(protocol)
    demo = DEMO_BUFFER = ReplayBuffer(protocol['replay_capacity'], protocol['demonstration_seed'])
    for item in demonstrations:
        demo.add_episode(item['arrays'])
    assert demonstration_gate(demo, protocol)
    demo.save(D / 'demonstrations_replay.npz')
    write(D / 'demonstration_buffer.json', dict(rows=len(demo), episodes=demo.episode_count,
          sources=[item['provenance'] for item in demonstrations], pretrain_offline_fraction=protocol['pretrain_offline_fraction'], online_offline_fraction=protocol['online_offline_fraction'],
          batch_demonstration_rows=dict(pretrain=int(np.floor(protocol['batch_size'] * protocol['pretrain_offline_fraction'] + .5)), online=int(np.floor(protocol['batch_size'] * protocol['online_offline_fraction'] + .5))),
          collected_prefix_rows_never_appended='physically replayed (and, when enabled, perturbed) prefix controls of collected episodes are never appended to the online replay; the demonstration buffer separately holds the recorded source rows listed in demonstration_recorded_rows, which may coincide with source transitions that prefix replay re-executes', immutable_during_stage=True,
          replay_sha256=sha(D / 'demonstrations_replay.npz')))
    lineage = protocol.get('prior_replay_lineage', dict(rows=2808, open_episode=True))
    with np.load(D / 'prior_replay_lineage_only.npz', allow_pickle=False) as prior:
        prior_metadata = json.loads(str(prior['metadata']))
        assert prior_metadata['open_episode'] == lineage['open_episode'] and len(prior['obs']) == lineage['rows']
        if lineage['open_episode']:
            assert not prior['terminated'][-1] and not prior['truncated'][-1] and not prior['next_action_valid'][-1]
    write(D / 'replay_lineage.json', dict(policy='fresh-stage-only online replay plus separate immutable demonstration buffer', inherited_rows_imported=0,
          prior_replay_sha256=sha(D / 'prior_replay_lineage_only.npz'), prior_rows=lineage['rows'],
          prior_final_open_row_preserved=lineage['open_episode'], prior_row_closed_or_linked_to_new_reset=False,
          inherited_checkpoint=protocol.get('inherited_checkpoint', 'bootstrap_suffix_sac_0001/checkpoints/online_2000')))
    counts = dict(warmup_complete_episodes=0, warmup_policy_interactions=0,
                  online_policy_interactions=0, physical_prefix_controls=0,
                  warmup_assisted_completed_episodes=0, warmup_assisted_force_successes=0,
                  online_assisted_completed_episodes=0, online_assisted_force_successes=0,
                  warmup_cold_completed_episodes=0, warmup_cold_force_successes=0,
                  online_cold_completed_episodes=0, online_cold_force_successes=0,
                  new_pretrain_updates=0, new_online_updates=0,
                  warmup_raw_reward_sum=0., online_raw_reward_sum=0.,
                  warmup_episodes_by_prefix={k: 0 for k in (multisource_keys(protocol) if sources else prefix_keys(protocol))},
                  warmup_force_successes_by_prefix={k: 0 for k in (multisource_keys(protocol) if sources else prefix_keys(protocol))},
                  online_episodes_by_prefix={k: 0 for k in (multisource_keys(protocol) if sources else prefix_keys(protocol))},
                  online_force_successes_by_prefix={k: 0 for k in (multisource_keys(protocol) if sources else prefix_keys(protocol))},
                  handoff_terminated_before_policy=0)
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
        key = episode_key(episode)
        counts[phase + '_episodes_by_prefix'][key] += 1
        counts[phase + '_force_successes_by_prefix'][key] += int(report['finger_supported_rotation_success'])
        counts['handoff_terminated_before_policy'] += int(episode.handoff_terminated_before_policy)
        episode.save(D / 'episodes' / phase / f'{episode.number:06d}',
                     dict(learner_online_updates=agent.online_updates, learner_pretrain_updates=agent.pretrain_updates))
        log('episode_complete', report=report)
        episode.close()

    try:
        save_checkpoint('initialized', agent, replay, None, counts, protocol, ec)
        assert_tree_equal(torch.load(D / 'initial.pt', map_location='cpu', weights_only=True),
                          torch.load(D / 'checkpoints/initialized/agent.pt', map_location='cpu', weights_only=True))
        if protocol.get('actor_lr_override') is not None:
            # Applied only after the inherited payload is proven identical: a stage-level change of the
            # actor step size; the ramp fraction and every other config field stay inherited.
            from dataclasses import replace as dc_replace
            inherited_actor_lr = agent.config.actor_lr
            agent.config = dc_replace(agent.config, actor_lr=float(protocol['actor_lr_override']))
            for group in agent.actor_optimizer.param_groups:
                group['lr'] = agent.config.actor_lr
            write(D / 'actor_lr_override.json', dict(inherited_actor_lr=inherited_actor_lr, actor_lr=agent.config.actor_lr,
                  applied_after='checkpoints/initialized', ramp='cumulative actor_warmup_updates 5000 unchanged',
                  later_checkpoints_carry_override=True))
            log('actor_lr_override', inherited_actor_lr=inherited_actor_lr, actor_lr=agent.config.actor_lr)
        if protocol.get('alpha_override') is not None or protocol.get('alpha_freeze', False):
            # Temperature change applied only after the inherited payload is proven identical.
            inherited_alpha = float(agent.alpha.detach())
            if protocol.get('alpha_override') is not None:
                with torch.no_grad():
                    agent.log_alpha.fill_(math.log(float(protocol['alpha_override'])))
            write(D / 'temperature_override.json', dict(inherited_normalized_alpha=inherited_alpha, normalized_alpha=float(agent.alpha.detach()),
                  raw_equivalent_alpha=float(agent.alpha.detach()) * protocol['raw_reward_divisor'], frozen=bool(protocol.get('alpha_freeze', False)),
                  applied_after='checkpoints/initialized', alpha_optimizer_state_untouched=True))
            log('temperature_override', inherited_normalized_alpha=inherited_alpha, normalized_alpha=float(agent.alpha.detach()), frozen=bool(protocol.get('alpha_freeze', False)))
        # Reference copies for the 'critic pretraining leaves actor and temperature untouched' checks, taken
        # after any stage-level override so that the checks compare against the stage's own starting state.
        source_actor_optimizer = copy.deepcopy(agent.actor_optimizer.state_dict())
        source_alpha_optimizer = copy.deepcopy(agent.alpha_optimizer.state_dict())
        source_log_alpha = agent.log_alpha.detach().clone()
        for number in range(protocol['warmup_complete_episodes']):
            current = new_episode(env, protocol, source, physics, 'warmup', number, prefix_rng, handoff_rng, sources)
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
        assert tensor_hashes(agent.actor) == source_hashes['actor']
        assert agent.online_updates == inherited['online_updates'] and agent.pretrain_updates == inherited['pretrain_updates']
        warmup = concatenate_returns(train_targets)
        np.savez_compressed(D / 'warmup_empirical_terminal_returns.npz', **warmup)
        save_checkpoint('warmup_complete', agent, replay, current, counts, protocol, ec)
        heldout = evaluate(agent, ec, source, physics, protocol, 'pretraining_baseline', sources=sources)
        calibrate(agent, warmup, D / 'warmup_calibration_before.npz', protocol['raw_reward_divisor'], 'training-warmup', protocol['warmup_alpha_for_calibration'], protocol['pretrain_alpha'])
        calibrate(agent, heldout, D / 'heldout_calibration_before.npz', protocol['raw_reward_divisor'], 'heldout-phase-baseline', protocol['warmup_alpha_for_calibration'], protocol['pretrain_alpha'])
        calibrate_demonstrations(agent, demonstrations, D / 'demonstration_calibration_before.json', protocol, source['obs'][MPC_HANDOFF_PREFIX])
        supported = demonstration_gate(demo, protocol)
        write(D / 'warmup_record.json', dict(learning_gate='verified off-policy demonstration rows', passed=supported,
              demonstration_rows=len(demo), demonstration_episodes=demo.episode_count,
              warmup_force_successes_informational=counts['warmup_assisted_force_successes'] + counts['warmup_cold_force_successes'],
              complete_warmup_episodes=counts['warmup_complete_episodes'], warmup_successes_not_required=True))
        log('warmup_record', passed=supported)
        if not supported:
            assert_tree_equal(torch.load(D / 'initial.pt', map_location='cpu', weights_only=True),
                              torch.load(D / 'checkpoints/warmup_complete/agent.pt', map_location='cpu', weights_only=True))
            assert counts['new_pretrain_updates'] == counts['new_online_updates'] == 0
            verify_manifest()
            write(D / 'summary.json', dict(status='warmup_only_no_positive_support', learning_started=False,
                  counts=counts, runtime=runtime, inherited_learner_counters=inherited,
                  learner_counters=dict(bc_updates=agent.bc_updates, pretrain_updates=agent.pretrain_updates, online_updates=agent.online_updates),
                  prefix_set=protocol['prefix_set'], completed_warmup_episodes=protocol['warmup_complete_episodes'],
                  unfinished_last_suffix=False, eligible_for_primitive=False, main_mpc_dataset_eligible=False,
                  trajectory_exported=False, no_automatic_fallback=True, reward_divisor=protocol['raw_reward_divisor'],
                  wall_seconds=time.monotonic() - start, caveat=protocol['normalization_caveat']))
        else:
            for _ in range(protocol['critic_pretrain_updates']):
                batch, stats = normalized_batch(demo, replay, protocol, 'pretrain')
                result = entropy_masked_pretrain_step(agent, batch, protocol)
                if result['pretrain_samples'] == 0:
                    raise RuntimeError('Prescribed SoftSARSA update received no eligible actual successor or terminal rows')
                counts['new_pretrain_updates'] += 1
                log('critic_pretrain_update', **result, **stats)
            assert agent.pretrain_updates == inherited['pretrain_updates'] + protocol['critic_pretrain_updates']
            assert agent.online_updates == inherited['online_updates']
            assert tensor_hashes(agent.actor) == source_hashes['actor']
            assert_tree_equal(source_actor_optimizer, agent.actor_optimizer.state_dict())
            assert_tree_equal(source_alpha_optimizer, agent.alpha_optimizer.state_dict())
            assert torch.equal(agent.log_alpha, source_log_alpha)
            calibrate(agent, warmup, D / 'warmup_calibration_after.npz', protocol['raw_reward_divisor'], 'training-warmup', protocol['warmup_alpha_for_calibration'], protocol['pretrain_alpha'])
            calibrate(agent, heldout, D / 'heldout_calibration_after.npz', protocol['raw_reward_divisor'], 'heldout-phase-baseline', protocol['warmup_alpha_for_calibration'], protocol['pretrain_alpha'])
            calibrate_demonstrations(agent, demonstrations, D / 'demonstration_calibration_after_pretraining.json', protocol, source['obs'][MPC_HANDOFF_PREFIX])
            save_checkpoint('pretrained', agent, replay, current, counts, protocol, ec)
            current, online_episode = None, 0
            for step in range(1, protocol['online_policy_interactions'] + 1):
                while current is None or current.term or current.trunc:
                    current = new_episode(env, protocol, source, physics, 'online', online_episode, prefix_rng, handoff_rng, sources)
                    online_episode += 1
                    counts['physical_prefix_controls'] += current.prefix
                    if current.term or current.trunc:
                        # A perturbed final prefix control ended the episode before any policy control:
                        # record it (zero policy interactions) and draw the next episode.
                        complete_episode(current, 'online')
                row = current.advance(command(agent, current.obs))
                replay.add_transition(*(row[k] for k in REPLAY_KEYS))
                counts['online_policy_interactions'] += 1
                counts['online_raw_reward_sum'] += row['rewards']
                batch, stats = normalized_batch(demo, replay, protocol, 'online')
                result = runner_update(agent, batch, protocol, demonstrations)
                counts['new_online_updates'] += 1
                log('online_update', **result, **stats,
                    raw_transition_reward=row['rewards'], normalized_transition_reward=row['normalized_rewards'],
                    alpha_raw_equivalent=float(agent.alpha.detach()) * protocol['raw_reward_divisor'])
                assert agent.online_updates == inherited['online_updates'] + counts['online_policy_interactions']
                assert counts['online_policy_interactions'] == step
                expected_lr = agent.config.actor_lr * min(agent.online_updates / agent.config.actor_warmup_updates, 1.)
                assert result['actor_lr'] == expected_lr
                if current.term or current.trunc:
                    complete_episode(current, 'online')
                if step in protocol['checkpoints']:
                    save_checkpoint(f'online_{step}', agent, replay, current, counts, protocol, ec)
                    calibrate_demonstrations(agent, demonstrations, D / f'demonstration_calibration_online_{step}.json', protocol, source['obs'][MPC_HANDOFF_PREFIX])
                    evaluate(agent, ec, source, physics, protocol, f'online_{step}', final=step == protocol['online_policy_interactions'], sources=sources)
            if not (current.term or current.trunc):
                current.save(D / 'unfinished_suffix', dict(collection_budget_reached=True,
                             terminal_not_fabricated=True, replay_pending_preserved=True))
            assert dict(bc_updates=agent.bc_updates, pretrain_updates=agent.pretrain_updates, online_updates=agent.online_updates) == protocol['expected_final_counters']
            for name, expected in protocol['expected_final_optimizer_steps'].items():
                assert optimizer_steps(getattr(agent, name + '_optimizer')) == [expected]
            verify_manifest()
            write(D / 'summary.json', dict(status='complete_sac_stage', learning_started=True, counts=counts, runtime=runtime,
                        alpha_frozen=bool(protocol.get('alpha_freeze', False)), alpha_override=protocol.get('alpha_override'),
                        demonstration_rows=len(demo),
                        inherited_learner_counters=inherited,
                        learner_counters=dict(bc_updates=agent.bc_updates, pretrain_updates=agent.pretrain_updates, online_updates=agent.online_updates),
                        unfinished_last_suffix=not bool(current.term or current.trunc),
                        prefix_set=protocol['prefix_set'], primary_prefix_controls=protocol['prefix_controls'],
                        actor_bc_lambda=protocol['actor_bc_lambda'], actor_bc_rows=protocol.get('actor_bc_rows', 'all'), actor_on_demonstration_states=protocol['actor_on_demonstration_states'],
                        handoff_noise_sigma=protocol.get('handoff_noise_sigma', 0.), handoff_noise_prefixes=protocol.get('handoff_noise_prefixes', []),
                        pretrain_offline_fraction=protocol['pretrain_offline_fraction'], online_offline_fraction=protocol['online_offline_fraction'],
                        eligible_for_primitive=False,
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
