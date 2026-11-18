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



from audit import run_audit


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    validated=json.loads((D/'audit_report.json').read_text())
    assert validated['bc_eligible_strict_prefix'] and validated['torch_clamped_actions_bitwise_unchanged']
    gate=json.loads((D/'noise_gate_report.json').read_text())
    assert gate['all_three_qualified'] and gate['status']=='complete', 'Three-case robustness gate did not pass'
    ec=EnvConfig(**json.loads((D/'strict_config.json').read_text())['environment'])
    with np.load(D/'strict_first_success/transitions.npz') as archive:
        obs,actions=archive['obs'].copy(),archive['actions'].copy()
    assert len(obs)==validated['first_success_prefix_controls'] and np.array_equal(actions,np.clip(actions,-.98,.98))
    out=D/'bc_paper';out.mkdir()
    learner=LearnerConfig(hidden_sizes=(256,256),ensemble_size=2,online_target_critics=2)
    assert learner.bc_lr==3e-3 and learner.bc_action_epsilon==.02
    torch.manual_seed(17001);rng=np.random.default_rng(17001)
    agent=SACAgent(60,16,learner,device='cuda')
    frozen=invariant_hashes(agent)
    report=dict(purpose='Single bounded BC-only clone of actual strict first-success action-restricted teacher prefix',
        production_eligible=False,trained_with_SAC=False,learner_config=asdict(learner),
        source_prefix_sha256=sha256(D/'strict_first_success/transitions.npz'),
        audit_report_sha256=sha256(D/'audit_report.json'),source_script_sha256=sha256(__file__),
        source_settings_url='https://real-world-rl-with-mpc.github.io/',
        source_settings_sha256=sha256(D/'source/project.html'),
        settings=dict(seed=17001,evaluation_seed=42,updates=12000,batch_size=256,bc_lr=.003,
            epsilon=.02,sampling='Uniform transition sampling with replacement within one actual episode',
            mean_only=True,std_after_each_evaluated_BC_checkpoint=.3,evaluation_updates=[1000,3000,6000,9000,12000],
            torch_threads=1,interop_threads=1,device='cuda',tf32=False,critic_updates=0,online_updates=0,
            diagnostic_critic_count=2,paper_critic_count=10,
            note='Unused critic count does not change the actor BC objective; only one local simulation episode is provided, not the paper 20 hardware episodes'),
        torch_version=str(torch.__version__),numpy_version=np.__version__,mujoco_version=mujoco.__version__,
        invariant_initial=frozen,results=[],status='running')
    write_json(out/'report.json',report)
    batches=np.empty((12000,256),dtype=np.uint8);begin=time.monotonic()
    for update in range(1,12001):
        indices=rng.integers(len(obs),size=256);batches[update-1]=indices
        training=agent.bc_step(dict(obs=obs[indices],actions=actions[indices]))
        if update%250==0:
            row=dict(update=update,**training,wall_seconds=time.monotonic()-begin)
            with (out/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
        if update not in report['settings']['evaluation_updates']:continue
        agent.initialize_std()
        checkpoint=out/f'checkpoint_{update:05d}.pt';agent.save(checkpoint)
        metadata=write_checkpoint_metadata(checkpoint,ec,learner,stage='diagnostic-bc-only',
            offline_replay_sha256=report['source_prefix_sha256'])
        metadata.update(production_eligible=False,actor_kind='behavior-cloning-only',critic_updates=0,online_updates=0)
        write_json(str(checkpoint)+'.metadata.json',metadata)
        fit,mean,predictions,targets=fit_metrics(agent,obs,actions,ec)
        np.savez_compressed(out/f'teacher_fit_{update:05d}.npz',observations=obs,actual_actions=actions,
            latent_targets=targets,predicted_latent_means=mean,predicted_actions=predictions)
        result,executed,_=run_audit(agent,ec,out/f'audit_{update:05d}',100)
        result['policy_kind']='diagnostic-behavior-cloning-only'
        write_json(out/f'audit_{update:05d}'/'report.json',result)
        row=dict(update=update,training=training,teacher_fit=fit,evaluation=result,
                 checkpoint_sha256=sha256(checkpoint),frozen_critic_and_alpha_hashes_unchanged=invariant_hashes(agent)==frozen)
        assert row['frozen_critic_and_alpha_hashes_unchanged'] and agent.online_updates==agent.pretrain_updates==0
        if result['finger_supported_rotation_success']:
            total=len(executed)+20;continuous_ec=replace(ec,terminate_on_success=False,horizon_steps=max(100,total))
            aftermath,_,_=run_audit(agent,continuous_ec,out/f'continuous_aftermath_{update:05d}',total)
            with np.load(out/f'audit_{update:05d}'/'physics.npz') as a,np.load(out/f'continuous_aftermath_{update:05d}'/'physics.npz') as b:
                assert np.array_equal(a['samples'],b['samples'][:len(a['samples'])])
            aftermath.update(policy_kind='diagnostic-behavior-cloning-only',requested_post_success_seconds=2.,
                completed_post_success_seconds=max(0.,aftermath['simulated_seconds']-.1*len(executed)),
                completed_two_seconds_without_failure=bool(aftermath['steps']==total and not aftermath['independent_drop'] and
                    not aftermath['independent_palm_support_violation']))
            row['continuous_policy_aftermath']=aftermath
            write_json(out/f'continuous_aftermath_{update:05d}'/'report.json',aftermath)
        report['results'].append(row);report['wall_seconds']=time.monotonic()-begin
        write_json(out/'report.json',report)
        print(json.dumps(dict(update=update,teacher_fit=fit,evaluation=result)),flush=True)
    np.savez_compressed(out/'sampled_batch_indices.npz',indices=batches)
    report.update(status='complete',invariant_final=invariant_hashes(agent),wall_seconds=time.monotonic()-begin,
                  batch_sequence_sha256=sha256(out/'sampled_batch_indices.npz'))
    write_json(out/'report.json',report)


if __name__=='__main__':main()
