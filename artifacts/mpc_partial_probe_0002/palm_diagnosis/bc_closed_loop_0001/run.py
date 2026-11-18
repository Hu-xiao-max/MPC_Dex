"""Bounded BC-only feasibility diagnostic; no critic or online updates."""
from pathlib import Path
from dataclasses import asdict, replace
import hashlib
import json
import sys
import time

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D / 'source'))
import mujoco
import numpy as np
import torch
from mpc_dex.env import AllegroRotationEnv, EnvConfig
from mpc_dex.learning import SACAgent, LearnerConfig
from mpc_dex.contracts import sha256, write_checkpoint_metadata
from scripts.audit_policy import PHYSICS_COLUMNS, audit_episode, write_json
from scripts.audit_recorded_controls import RecordedPolicy


def module_hash(module):
    digest = hashlib.sha256()
    for key, tensor in sorted(module.state_dict().items()):
        digest.update(key.encode()); digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def invariant_hashes(agent):
    return dict(critics=module_hash(agent.critics), target_critics=module_hash(agent.target_critics),
                log_alpha=hashlib.sha256(agent.log_alpha.detach().cpu().numpy().tobytes()).hexdigest())


class RecordingAgent:
    def __init__(self, agent):
        self.agent = agent
        self.obs = []
        self.actions = []

    def act(self, obs, deterministic=True):
        action = self.agent.act(obs, deterministic=deterministic)
        self.obs.append(obs.copy()); self.actions.append(action.copy())
        return action


def run_audit(agent, ec, output, limit):
    output.mkdir()
    env = AllegroRotationEnv(ec)
    policy = RecordingAgent(agent)
    original_step = env.step
    next_obs, rewards, dones, truncations, qpos, qvel, targets = [], [], [], [], [], [], []
    initial = {}
    original_reset = env.reset

    def capture_reset(*args, **kwargs):
        result = original_reset(*args, **kwargs)
        initial.update(qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(),
                       ctrl=env.data.ctrl.copy(), joint_positions=env.data.qpos[env.q_ids].copy())
        return result

    def capture_step(action):
        result = original_step(action)
        ob, reward, done, truncated, info = result
        next_obs.append(ob.copy()); rewards.append(reward); dones.append(done); truncations.append(truncated)
        qpos.append(env.data.qpos.copy()); qvel.append(env.data.qvel.copy()); targets.append(env.data.ctrl.copy())
        return result

    env.reset = capture_reset
    env.step = capture_step
    try:
        result, physics, commands = audit_episode(env, policy, 0, limit, True)
        result.update(production_eligible=False, policy_kind='diagnostic-behavior-cloning-only',
                      environment=asdict(ec), model_equality_constraints=int(env.model.neq))
        np.savez_compressed(output / 'transitions.npz', obs=np.asarray(policy.obs),
            actions=np.asarray(policy.actions), next_obs=np.asarray(next_obs), rewards=np.asarray(rewards),
            terminated=np.asarray(dones), truncated=np.asarray(truncations),
            qpos=np.asarray(qpos), qvel=np.asarray(qvel), targets=np.asarray(targets),
            initial_qpos=initial['qpos'], initial_qvel=initial['qvel'], initial_ctrl=initial['ctrl'],
            initial_joint_positions=initial['joint_positions'])
        np.savez_compressed(output / 'physics.npz', samples=physics, columns=np.asarray(PHYSICS_COLUMNS))
        write_json(output / 'commands.json', commands)
        write_json(output / 'report.json', result)
        return result, np.asarray(policy.actions), np.asarray(policy.obs)
    finally:
        env.close()


def fit_metrics(agent, obs, actions, ec):
    with torch.no_grad():
        values = torch.as_tensor(obs, dtype=torch.float32, device=agent.device)
        means = agent.actor(values)[0].cpu().numpy()
    epsilon = agent.config.bc_action_epsilon
    targets = np.arctanh(np.clip(actions, -1 + epsilon, 1 - epsilon))
    predictions = np.tanh(means)
    delta = predictions - actions
    env = AllegroRotationEnv(ec)
    scale = .5 * (env.high - env.low)
    env.close()
    return dict(latent_target_mse=float(np.mean((means - targets) ** 2)),
        latent_target_max_abs_error=float(np.max(np.abs(means - targets))),
        teacher_action_mse=float(np.mean(delta ** 2)), teacher_action_max_abs_error=float(np.max(np.abs(delta))),
        clipped_teacher_action_mse=float(np.mean((predictions - np.tanh(targets)) ** 2)),
        max_joint_target_error_radians=float(np.max(np.abs(delta * scale))),
        action_error_by_step_max=np.max(np.abs(delta), axis=1).tolist(),
        teacher_saturated_exact_count=int(np.count_nonzero(np.abs(actions) == 1.)),
        teacher_clipped_count=int(np.count_nonzero(np.abs(actions) > 1 - epsilon)),
        predicted_saturated_098_count=int(np.count_nonzero(np.abs(predictions) >= .98)),
        predicted_saturated_0999999_count=int(np.count_nonzero(np.abs(predictions) >= .999999)),
        target_latent_abs_max=float(np.max(np.abs(targets))),
        pure_clipping_action_mse=float(np.mean((np.tanh(targets) - actions) ** 2)),
        pure_clipping_action_max_abs_error=float(np.max(np.abs(np.tanh(targets) - actions))),
        pure_clipping_joint_target_max_abs_error_radians=float(np.max(np.abs((np.tanh(targets) - actions) * scale)))), means, predictions, targets


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('The authorized diagnostic requires CUDA')
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = json.loads((D / 'config.json').read_text())
    ec = EnvConfig(**cfg['environment'])
    with np.load(D / 'teacher_transitions.npz', allow_pickle=False) as data:
        obs, actions = data['obs'].copy(), data['actions'].copy()
    assert obs.shape == (19, 60) and actions.shape == (19, 16)
    settings = dict(purpose='Closed-loop actor BC feasibility diagnostic, not a SAC-trained policy or production primitive',
        production_eligible=False, training_seed=17001, evaluation_seed=0,
        epsilons=[.02, 1e-6], bc_lr=3e-4, batch_size=19, batch_sampling='full actual demonstration every update',
        maximum_updates_per_condition=6000, evaluation_updates=[1000, 3000, 6000],
        evaluation_max_steps=100, architecture=[256, 256], critics=2, critic_updates=0, online_updates=0,
        device='cuda', torch_threads=1, interop_threads=1, tf32=False,
        torch_version=str(torch.__version__), mujoco_version=mujoco.__version__,
        input_sha256=sha256(D / 'teacher_transitions.npz'), run_source_sha256=sha256(__file__),
        status='running', conditions=[])
    write_json(D / 'report.json', settings)
    teacher, _, replay_obs = run_audit(RecordedPolicy(actions), ec, D / 'teacher_replay', 19)
    teacher['max_replay_observation_error'] = float(np.max(np.abs(replay_obs - obs)))
    assert teacher['finger_supported_rotation_success'], teacher
    assert teacher['max_replay_observation_error'] < 1e-6
    settings['teacher_revalidation'] = teacher
    first_validated = None
    begin = time.monotonic()
    for epsilon in settings['epsilons']:
        label = 'epsilon_002' if epsilon == .02 else 'epsilon_0000001'
        folder = D / label
        folder.mkdir()
        torch.manual_seed(settings['training_seed']); np.random.seed(settings['training_seed'])
        learner = LearnerConfig(hidden_sizes=(256, 256), ensemble_size=2, online_target_critics=2,
                                bc_lr=3e-4, bc_action_epsilon=epsilon)
        agent = SACAgent(60, 16, learner, device='cuda')
        agent.initialize_std()
        frozen = invariant_hashes(agent)
        condition = dict(epsilon=epsilon, label=label, learner_config=asdict(learner),
                         invariant_initial=frozen, initial_actor_sha256=module_hash(agent.actor), results=[])
        clipped = np.clip(actions, -1 + epsilon, 1 - epsilon)
        clipping_result, _, _ = run_audit(RecordedPolicy(np.vstack((clipped, np.repeat(clipped[-1:], 81, axis=0)))),
                                       ec, folder / 'clipped_teacher_open_loop', 100)
        condition['clipped_teacher_open_loop'] = clipping_result
        settings['conditions'].append(condition)
        write_json(folder / 'settings.json', condition)
        batch = dict(obs=obs, actions=actions)
        for update in range(1, 6001):
            train_metrics = agent.bc_step(batch)
            if update % 250 == 0:
                print(json.dumps(dict(condition=label, update=update, **train_metrics,
                                      wall_seconds=time.monotonic() - begin)), flush=True)
            if update not in settings['evaluation_updates']:
                continue
            checkpoint = folder / f'checkpoint_{update:05d}.pt'
            agent.save(checkpoint)
            metadata = write_checkpoint_metadata(checkpoint, ec, learner, stage='diagnostic-bc-only',
                                                offline_replay_sha256=settings['input_sha256'])
            metadata.update(production_eligible=False, actor_kind='behavior-cloning-only',
                            critic_updates=0, online_updates=0, source_snapshot=str(D / 'source'))
            write_json(str(checkpoint) + '.metadata.json', metadata)
            metrics, mean, prediction, targets = fit_metrics(agent, obs, actions, ec)
            np.savez_compressed(folder / f'teacher_fit_{update:05d}.npz', observations=obs, actual_actions=actions,
                                latent_targets=targets, predicted_latent_means=mean, predicted_actions=prediction)
            evaluation, executed_actions, _ = run_audit(agent, ec, folder / f'audit_{update:05d}', 100)
            frozen_after = invariant_hashes(agent)
            assert frozen_after == frozen and agent.online_updates == 0 and agent.pretrain_updates == 0
            result = dict(update=update, training=train_metrics, teacher_fit=metrics, evaluation=evaluation,
                          frozen_critic_and_alpha_hashes_unchanged=frozen_after == frozen,
                          checkpoint_sha256=sha256(checkpoint))
            condition['results'].append(result)
            if evaluation['finger_supported_rotation_success'] and first_validated is None:
                first_validated = dict(condition=label, update=update)
                settings['first_validated_full_turn'] = first_validated
                aftermath_ec = replace(ec, terminate_on_success=False)
                total = min(len(executed_actions) + 20, 100)
                held = np.vstack((executed_actions, np.repeat(executed_actions[-1:], 20, axis=0)))
                result['hold_aftermath'], _, _ = run_audit(RecordedPolicy(held), aftermath_ec,
                    folder / f'hold_aftermath_{update:05d}', total)
                result['continuous_policy_aftermath'], _, _ = run_audit(agent, aftermath_ec,
                    folder / f'continuous_aftermath_{update:05d}', total)
            settings['wall_seconds'] = time.monotonic() - begin
            write_json(D / 'report.json', settings)
            print(json.dumps(dict(condition=label, update=update, teacher_fit=metrics,
                                  evaluation=evaluation)), flush=True)
        condition['invariant_final'] = invariant_hashes(agent)
    settings.update(status='complete', wall_seconds=time.monotonic() - begin,
                    first_validated_full_turn=first_validated)
    write_json(D / 'report.json', settings)
    manifest = {str(p.relative_to(D)): sha256(p) for p in sorted(D.rglob('*'))
                if p.is_file() and '__pycache__' not in p.parts and p.name not in ('manifest.json', 'run.log')}
    write_json(D / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
