"""Demonstration-initialized SAC from cold reset (artifact-level replica of the paper's BC -> critic -> online stages).

Inputs (hash-bound by prepare.py): verified cold-reset MPC demonstrations (episodes/seed_*/transitions.npz),
the frozen numerical source snapshot, the environment config and the SAC0001 learner config. Stages:
1. fresh SACAgent; `bc_updates` mean-only BC steps on demonstration rows (episode-weighted sampling as in the
   archived learner), then initialize_std;
2. `critic_pretrain_updates` archived Soft-SARSA steps on demonstration rows (actual next actions);
3. `online_policy_interactions` cold-reset SAC interactions with one archived update per step on 30/70
   demonstration/online batches; sampled batch rewards divided by `raw_reward_divisor`.
Evaluations at every checkpoint: cold deterministic plus `eval_stochastic_trials` stochastic trials, in separate
environments with training RNG restored; nothing from evaluation enters replay. The demonstrations come from the
measured diagnostic prior, not a SAC-derived primitive (disclosed deviation); nothing here is primitive-eligible.
"""
from __future__ import annotations
from pathlib import Path
import argparse, copy, hashlib, json, math, os, random, sys, time
from dataclasses import asdict

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D / 'source'))
import mujoco, numpy as np, torch
from mpc_dex.env import AllegroRotationEnv, EnvConfig
from mpc_dex.learning import SACAgent, LearnerConfig
from mpc_dex.replay import ReplayBuffer, sample_mixed
from mpc_dex.contracts import write_checkpoint_metadata
from scripts.audit_policy import PhysicsObserver, PHYSICS_COLUMNS
for name in ('mpc_dex.env', 'mpc_dex.learning', 'mpc_dex.replay', 'scripts.audit_policy'):
    assert Path(sys.modules[name].__file__).resolve().is_relative_to((D / 'source').resolve())

SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
REPLAY_KEYS = ('obs', 'actions', 'rewards', 'next_obs', 'terminated', 'truncated')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def jsonable(v):
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, np.generic): return v.item()
    if isinstance(v, dict): return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [jsonable(x) for x in v]
    return v
def write(path, value):
    path = Path(path); tmp = path.with_name(path.name + '.tmp'); tmp.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + '\n'); tmp.replace(path)
def rng_state(): return dict(torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state_all(), numpy=np.random.get_state(), python=random.getstate())
def restore_rng(s): torch.set_rng_state(s['torch_cpu']); torch.cuda.set_rng_state_all(s['torch_cuda']); np.random.set_state(s['numpy']); random.setstate(s['python'])
def seed_all(seed): torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed); random.seed(seed)
def tensor_hashes(m): return {k: hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k, v in m.state_dict().items()}
def integration_state(env):
    s = np.empty(mujoco.mj_stateSize(env.model, SPEC)); mujoco.mj_getState(env.model, env.data, s, SPEC); return s
def verify_manifest():
    m = json.loads((D / 'execution_manifest.json').read_text())
    bad = [k for k, v in m.items() if not (D / k).is_file() or sha(D / k) != v]
    if bad: raise RuntimeError('Frozen execution inputs changed: ' + ', '.join(bad[:5]))


@torch.no_grad()
def command(agent, obs, deterministic=False):
    value = torch.as_tensor(obs[None], dtype=torch.float32, device=agent.device)
    mean, log_std = agent.actor(value)
    if deterministic:
        action, density = mean.tanh(), agent.actor._log_prob(mean, log_std, mean)
    else:
        action, density = agent.actor.sample(value)
    return action[0].cpu().numpy(), float(density[0])


class ColdEpisode:
    def __init__(self, env, seed, name, number):
        self.env, self.name, self.number = env, name, number
        self.obs, self.info = env.reset(seed=seed); self.observer = PhysicsObserver(env)
        self.rows = []; self.term = self.trunc = False; self.closed = False
    def advance(self, action, log_prob):
        assert not self.term and not self.trunc and action.dtype == np.float32 and action.shape == (16,) and np.isfinite(action).all()
        before = self.obs.copy(); self.obs, reward, self.term, self.trunc, self.info = self.env.step(action)
        self.rows.append(dict(obs=before, actions=action.copy(), rewards=float(reward), next_obs=self.obs.copy(), terminated=bool(self.term), truncated=bool(self.trunc), log_prob=log_prob))
        return self.rows[-1]
    def report(self):
        ob = self.observer; physics = np.asarray(ob.rows, np.float64); col = {k: physics[:, i] for i, k in enumerate(PHYSICS_COLUMNS)}
        external = ('palm_force_norm', 'table_force_norm', 'other_force_norm', 'table_normal_force', 'other_normal_force', 'object_applied_wrench_norm', 'object_applied_generalized_force_norm')
        no_external = not ob.palm_support_violation and all(np.max(col[k], initial=0) <= 1e-6 for k in external)
        success = bool(self.term and not self.trunc and self.info.get('success') and not self.info.get('drop') and col['force_bearing_fingers'][-1] >= 2 and ob.raw_angle >= 2 * np.pi and ob.qualified_angle >= 2 * np.pi and ob.force_qualified_angle >= 2 * np.pi and col['force_retained'][-1] and not ob.independent_drop and no_external and self.env.model.neq == 0)
        return dict(name=self.name, number=self.number, controls=len(self.rows), terminated=bool(self.term), truncated=bool(self.trunc), environment_success=bool(self.info.get('success')), failure_reason=self.info.get('failure_reason'), finger_supported_rotation_success=success, cold_start_policy_success=success, independent_drop=bool(ob.independent_drop), palm=bool(ob.palm_support_violation), no_external_support=bool(no_external), final_force_qualified_angle=ob.force_qualified_angle, max_force_qualified_angle=float(np.max(col['force_qualified_angle'], initial=0)), raw_return=float(sum(r['rewards'] for r in self.rows)))
    def save(self, dest, extra=None):
        dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest / 'transitions.npz', **{k: np.asarray([r[k] for r in self.rows]) for k in self.rows[0]})
        np.savez_compressed(dest / 'physics.npz', samples=np.asarray(self.observer.rows, np.float64), columns=np.asarray(PHYSICS_COLUMNS))
        write(dest / 'report.json', dict(self.report(), **(extra or {})))
    def close(self):
        if not self.closed: self.observer.close(); self.closed = True


def load_demonstrations(protocol):
    demos = []
    for name in protocol['demonstration_episodes']:
        d = D / 'demonstrations' / name
        rep = json.loads((d / 'report.json').read_text()); assert rep['strict_force_success'] and rep['stored'] and rep['independent_audit']['bitexact_replay']
        assert sha(d / 'transitions.npz') == rep['transitions_sha256']
        with np.load(d / 'transitions.npz', allow_pickle=False) as z:
            arrays = {k: z[k].copy() for k in REPLAY_KEYS}
        assert arrays['terminated'][-1] and not arrays['terminated'][:-1].any() and not arrays['truncated'].any()
        assert np.array_equal(arrays['next_obs'][:-1], arrays['obs'][1:]) and np.abs(arrays['actions']).max() <= 1
        demos.append(dict(name=name, arrays=arrays, rows=len(arrays['obs']), raw_return=float(arrays['rewards'].sum())))
    return demos


def normalized(batch, divisor):
    raw = batch['rewards'].copy(); batch['rewards'] = raw / np.float32(divisor); return batch, float(raw.mean())


def evaluate(agent, ec, protocol, label, dest_root):
    saved = rng_state(); before = {k: tensor_hashes(getattr(agent, k)) for k in ('actor', 'critics', 'target_critics')}
    counters = (agent.bc_updates, agent.pretrain_updates, agent.online_updates)
    dest = dest_root / label; dest.mkdir(parents=True, exist_ok=False); reports = []
    tasks = [('cold_deterministic', protocol['eval_deterministic_seed'], True)] + [(f'cold_stochastic_{s}', s, False) for s in protocol['eval_stochastic_seeds']]
    try:
        for name, seed, det in tasks:
            seed_all(seed); env = AllegroRotationEnv(ec); ep = None
            try:
                ep = ColdEpisode(env, protocol['physical_reset_seed'], name, seed)
                while not (ep.term or ep.trunc):
                    a, lp = command(agent, ep.obs, det); ep.advance(a, lp)
                ep.save(dest / name, dict(deterministic=det, policy_sampling_seed=seed, learner_counters=dict(bc_updates=agent.bc_updates, pretrain_updates=agent.pretrain_updates, online_updates=agent.online_updates)))
                reports.append(dict(ep.report(), deterministic=det))
            finally:
                if ep is not None: ep.close()
                env.close()
    finally:
        restore_rng(saved)
    assert before == {k: tensor_hashes(getattr(agent, k)) for k in before} and counters == (agent.bc_updates, agent.pretrain_updates, agent.online_updates)
    summary = dict(reports=reports, cold_deterministic_success=reports[0]['finger_supported_rotation_success'], cold_stochastic_successes=sum(r['finger_supported_rotation_success'] for r in reports[1:]), cold_stochastic_trials=len(reports) - 1, eligible_for_primitive=False)
    write(dest / 'summary.json', summary); return summary


def save_checkpoint(label, agent, replay, counts, protocol, ec):
    parent = D / 'checkpoints'; parent.mkdir(exist_ok=True); temp = parent / (label + '.pending'); temp.mkdir(exist_ok=False)
    agent.save(temp / 'agent.pt')
    meta = write_checkpoint_metadata(temp / 'agent.pt', ec, agent.config, stage='demonstration-initialized-cold-sac', policy_interface={'kind': 'absolute_joint_target'})
    meta.update(learner_counters=dict(bc_updates=agent.bc_updates, pretrain_updates=agent.pretrain_updates, online_updates=agent.online_updates), eligible_for_primitive=False, main_mpc_dataset_eligible=False, frozen_checkpoint_collection=False, demonstration_source='diagnostic-prior MPC from cold reset (not a SAC primitive)')
    write(temp / 'agent.pt.metadata.json', meta); replay.save(temp / 'replay.npz'); torch.save(rng_state(), temp / 'rng.pt')
    write(temp / 'state.json', dict(counts=counts, label=label, learner_counters=meta['learner_counters'], normalized_alpha=float(agent.alpha.detach()), replay_rows=len(replay)))
    write(temp / 'manifest.json', {str(p.relative_to(temp)): sha(p) for p in sorted(temp.rglob('*')) if p.is_file()})
    temp.rename(parent / label)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--execute', action='store_true'); args = ap.parse_args()
    if not args.execute: raise SystemExit('Prepared runner only; pass --execute after review.')
    if (D / 'runtime.json').exists(): raise SystemExit('This stage already started')
    verify_manifest(); protocol = json.loads((D / 'protocol.json').read_text()); assert protocol['status'] == 'frozen-ready-for-separate-explicit-execution'
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    write(D / 'runtime.json', dict(torch=str(torch.__version__), mujoco=mujoco.__version__, numpy=np.__version__, python=sys.version, device=torch.cuda.get_device_name(), frozen_execution_manifest_sha256=sha(D / 'execution_manifest.json')))
    ec = EnvConfig(**json.loads((D / 'config.json').read_text())['environment']); assert ec.task == 'rotation' and ec.object_diameter == .04 and ec.horizon_steps == 100 and ec.terminate_on_success
    seed_all(protocol['train_seed'])
    config = LearnerConfig(**json.loads((D / 'learner_config.json').read_text()))
    agent = SACAgent(60, 16, config, 'cuda'); write(D / 'resolved_learner.json', asdict(agent.config))
    demos = load_demonstrations(protocol)
    demo = ReplayBuffer(protocol['replay_capacity'], protocol['demonstration_seed'])
    for item in demos: demo.add_episode(item['arrays'])
    assert demo.episode_count == len(demos) == protocol['demonstration_count'] and len(demo) == sum(d['rows'] for d in demos)
    demo.save(D / 'demonstrations_replay.npz'); write(D / 'demonstration_buffer.json', dict(episodes=[{k: v for k, v in d.items() if k != 'arrays'} for d in demos], rows=len(demo)))
    replay = ReplayBuffer(protocol['replay_capacity'], protocol['train_seed'])
    counts = dict(bc_updates=0, pretrain_updates=0, online_policy_interactions=0, online_updates=0, online_completed_episodes=0, online_cold_force_successes=0, online_raw_reward_sum=0.)
    metrics = (D / 'metrics.jsonl').open('x'); start = time.monotonic()
    def log(event, **v): metrics.write(json.dumps(jsonable(dict(event=event, elapsed_seconds=time.monotonic() - start, counts=copy.deepcopy(counts), **v)), allow_nan=False) + '\n'); metrics.flush()
    current = None; env = AllegroRotationEnv(ec)
    try:
        for i in range(protocol['bc_updates']):
            result = agent.bc_step(demo.sample(protocol['batch_size'], weighted_episodes=True, temperature=1.0)); counts['bc_updates'] += 1
            if i % 100 == 0 or i == protocol['bc_updates'] - 1: log('bc_update', **result)
        agent.initialize_std(); save_checkpoint('bc_complete', agent, replay, counts, protocol, ec)
        evaluate(agent, ec, protocol, 'bc_complete', D / 'evaluations')
        for i in range(protocol['critic_pretrain_updates']):
            batch, raw_mean = normalized(demo.sample(protocol['batch_size']), protocol['raw_reward_divisor'])
            result = agent.pretrain_critic_step(batch); counts['pretrain_updates'] += 1
            if i % 100 == 0 or i == protocol['critic_pretrain_updates'] - 1: log('critic_pretrain_update', **result, batch_raw_reward_mean=raw_mean)
        save_checkpoint('pretrained', agent, replay, counts, protocol, ec); evaluate(agent, ec, protocol, 'pretrained', D / 'evaluations')
        online_episode = 0
        for step in range(1, protocol['online_policy_interactions'] + 1):
            if current is None or current.term or current.trunc:
                current = ColdEpisode(env, protocol['physical_reset_seed'], 'online', online_episode); online_episode += 1
            a, lp = command(agent, current.obs); row = current.advance(a, lp)
            replay.add_transition(*(row[k] for k in REPLAY_KEYS)); counts['online_policy_interactions'] += 1; counts['online_raw_reward_sum'] += row['rewards']
            if len(replay) >= protocol['learning_starts']:
                batch, raw_mean = normalized(sample_mixed(demo, replay, protocol['batch_size'], protocol['offline_fraction']), protocol['raw_reward_divisor'])
                result = agent.update(batch); counts['online_updates'] += 1; log('online_update', **result, batch_raw_reward_mean=raw_mean, transition_reward=row['rewards'])
            if current.term or current.trunc:
                rep = current.report(); counts['online_completed_episodes'] += 1; counts['online_cold_force_successes'] += int(rep['finger_supported_rotation_success'])
                current.save(D / 'episodes' / f'{current.number:06d}', dict(learner_online_updates=agent.online_updates)); log('episode_complete', report=rep); current.close()
            if step in protocol['checkpoints']:
                save_checkpoint(f'online_{step}', agent, replay, counts, protocol, ec); s = evaluate(agent, ec, protocol, f'online_{step}', D / 'evaluations')
                log('evaluation', label=f'online_{step}', cold_deterministic=s['cold_deterministic_success'], cold_stochastic=f"{s['cold_stochastic_successes']}/{s['cold_stochastic_trials']}")
        verify_manifest()
        write(D / 'summary.json', dict(status='complete', counts=counts, learner_counters=dict(bc_updates=agent.bc_updates, pretrain_updates=agent.pretrain_updates, online_updates=agent.online_updates), demonstration_count=len(demos), demonstration_rows=len(demo), eligible_for_primitive=False, wall_seconds=time.monotonic() - start))
    except BaseException as error:
        write(D / 'failure.json', dict(type=type(error).__name__, message=str(error), counts=counts)); raise
    finally:
        metrics.close()
        if current is not None: current.close()
        env.close()
    write(D / 'manifest.json', {str(p.relative_to(D)): sha(p) for p in sorted(D.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name not in ('manifest.json', 'run.log')})


if __name__ == '__main__':
    main()
