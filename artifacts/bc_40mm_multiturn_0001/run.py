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
        result.update(production_eligible=False, policy_kind=('recorded-control-replay' if isinstance(agent, RecordedPolicy) else 'diagnostic-behavior-cloning-only'),
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
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required for the authorized diagnostic')
    torch.backends.cuda.matmul.allow_tf32=False
    ec=EnvConfig(**json.loads((D/'config.json').read_text())['environment'])
    with np.load(D/'teacher_transitions.npz',allow_pickle=False) as archive:
        all_obs,all_actions=archive['obs'].copy(),archive['actions'].copy()
    assert all_obs.shape==(60,60) and all_actions.shape==(60,16)
    settings=dict(purpose='Bounded diagnostic BC comparison of first-turn versus multi-turn actual teacher coverage',
        production_eligible=False, training_seed=17001, evaluation_seed=0, epsilon=1e-6, bc_lr=3e-4,
        conditions=[],dataset_sizes=[19,60],batch_sampling='Full selected dataset each update',
        maximum_updates_per_condition=6000,evaluation_updates=[1000,3000,6000],
        evaluation_max_steps=100,terminate_on_success=True,architecture=[256,256],critics=2,
        critic_updates=0,online_updates=0,device='cuda',torch_threads=1,interop_threads=1,tf32=False,
        torch_version=str(torch.__version__),mujoco_version=mujoco.__version__,
        input_sha256=sha256(D/'teacher_transitions.npz'),run_source_sha256=sha256(__file__),status='running',
        aftermath_policy='Continue deterministic learned feedback for two seconds after any independently qualified actor turn; never replace with a constant held command')
    write_json(D/'report.json',settings)
    teacher,_,teacher_obs=run_audit(RecordedPolicy(all_actions),ec,D/'teacher_strict_replay',100)
    assert teacher['finger_supported_rotation_success'] and len(teacher_obs)==19
    teacher['max_replay_observation_error']=float(abs(teacher_obs-all_obs[:19]).max())
    assert teacher['max_replay_observation_error']==0.
    clipped=np.clip(all_actions,-1+1e-6,1-1e-6)
    clipping,_,_=run_audit(RecordedPolicy(clipped),ec,D/'teacher_clipped_strict_replay',100)
    assert clipping['finger_supported_rotation_success']
    settings['teacher_revalidation']=teacher
    settings['clipped_teacher_revalidation']=clipping
    begin=time.monotonic();first_validated=None
    for count in settings['dataset_sizes']:
        label=f'first_{count:02d}'
        folder=D/label;folder.mkdir()
        obs,actions=all_obs[:count],all_actions[:count]
        np.savez_compressed(folder/'training_inputs.npz',obs=obs,actions=actions)
        torch.manual_seed(17001);np.random.seed(17001)
        learner=LearnerConfig(hidden_sizes=(256,256),ensemble_size=2,online_target_critics=2,
                              bc_lr=3e-4,bc_action_epsilon=1e-6)
        agent=SACAgent(60,16,learner,device='cuda');agent.initialize_std()
        frozen=invariant_hashes(agent)
        condition=dict(label=label,training_rows=count,training_input_sha256=sha256(folder/'training_inputs.npz'),
            learner_config=asdict(learner),invariant_initial=frozen,initial_actor_sha256=module_hash(agent.actor),
            teacher_saturation_count=int((abs(actions)==1).sum()),teacher_scalar_action_count=int(actions.size),results=[])
        settings['conditions'].append(condition);write_json(folder/'settings.json',condition)
        for update in range(1,6001):
            training=agent.bc_step(dict(obs=obs,actions=actions))
            if update%250==0:
                print(json.dumps(dict(condition=label,update=update,**training,wall_seconds=time.monotonic()-begin)),flush=True)
            if update not in settings['evaluation_updates']:continue
            checkpoint=folder/f'checkpoint_{update:05d}.pt';agent.save(checkpoint)
            metadata=write_checkpoint_metadata(checkpoint,ec,learner,stage='diagnostic-bc-only',
                                                offline_replay_sha256=condition['training_input_sha256'])
            metadata.update(production_eligible=False,actor_kind='behavior-cloning-only',
                            critic_updates=0,online_updates=0,source_snapshot=str(D/'source'))
            write_json(str(checkpoint)+'.metadata.json',metadata)
            metrics,mean,predictions,targets=fit_metrics(agent,obs,actions,ec)
            first_fit,_,_,_=fit_metrics(agent,all_obs[:19],all_actions[:19],ec)
            full_fit,_,_,_=fit_metrics(agent,all_obs,all_actions,ec)
            np.savez_compressed(folder/f'teacher_fit_{update:05d}.npz',observations=obs,actual_actions=actions,
                latent_targets=targets,predicted_latent_means=mean,predicted_actions=predictions)
            evaluation,executed,_=run_audit(agent,ec,folder/f'audit_{update:05d}',100)
            assert invariant_hashes(agent)==frozen and agent.online_updates==0 and agent.pretrain_updates==0
            result=dict(update=update,training=training,teacher_fit=metrics,first19_teacher_fit=first_fit,
                        full60_teacher_fit=full_fit,evaluation=evaluation,
                        frozen_critic_and_alpha_hashes_unchanged=True,checkpoint_sha256=sha256(checkpoint))
            condition['results'].append(result)
            if evaluation['finger_supported_rotation_success']:
                if first_validated is None:first_validated=dict(condition=label,update=update)
                total=len(executed)+20
                continuous_ec=replace(ec,terminate_on_success=False,horizon_steps=max(100,total))
                continuation,_,_=run_audit(agent,continuous_ec,folder/f'continuous_aftermath_{update:05d}',total)
                with np.load(folder/f'audit_{update:05d}'/'physics.npz') as a,np.load(folder/f'continuous_aftermath_{update:05d}'/'physics.npz') as b:
                    same=bool(np.array_equal(a['samples'],b['samples'][:len(a['samples'])]))
                assert same
                continuation.update(requested_seconds_after_first_success=2.,original_success_step=len(executed),
                    strict_prefix_physics_bitexact=same,completed_post_success_seconds=max(0.,continuation['simulated_seconds']-len(executed)*.1),
                    completed_two_seconds_without_failure=bool(continuation['steps']==total and not continuation['independent_drop'] and
                        not continuation['independent_palm_support_violation']))
                result['continuous_policy_aftermath']=continuation
                write_json(folder/f'continuous_aftermath_{update:05d}'/'report.json',continuation)
            settings.update(wall_seconds=time.monotonic()-begin,first_validated_full_turn=first_validated)
            write_json(D/'report.json',settings)
            print(json.dumps(dict(condition=label,update=update,teacher_fit=metrics,evaluation=evaluation)),flush=True)
        condition['invariant_final']=invariant_hashes(agent)
    settings.update(status='complete',wall_seconds=time.monotonic()-begin,first_validated_full_turn=first_validated)
    write_json(D/'report.json',settings)
    originals=json.loads((D/'inputs_manifest.json').read_text())
    assert all(sha256(D/key)==value for key,value in originals.items())
    manifest={str(p.relative_to(D)):sha256(p) for p in sorted(D.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p.name not in ('manifest.json','run.log')}
    write_json(D/'manifest.json',manifest)


if __name__=='__main__':main()

