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
    torch.manual_seed(17001);rng=np.random.default_rng(17001)
    ec=EnvConfig(**json.loads((D/'strict_config.json').read_text())['environment'])
    with np.load(D/'supervised_pairs.npz') as a:obs,actions=a['obs'].copy(),a['actions'].copy()
    assert obs.shape==(25,60) and actions.shape==(25,16)
    baseline_hash=sha256(D/'initial_polished.pt')
    agent=SACAgent.load(D/'initial_polished.pt','cuda')
    assert agent.config.bc_lr==.003 and agent.config.bc_action_epsilon==.02
    original_updates=agent.bc_updates;frozen=invariant_hashes(agent)
    out=D/'bc_feedback';out.mkdir()
    report=dict(purpose='One bounded mean-only feedback-data imitation round, not main-stage replay or a SAC policy',
        production_eligible=False,source_script_sha256=sha256(__file__),initial_checkpoint_sha256=baseline_hash,
        supervised_data_sha256=sha256(D/'supervised_pairs.npz'),
        supervision=dict(original_teacher_rows=22,actual_mpc_correction_rows=3,excluded_actor_prefix_rows=6,
            group_sampling='Uniform across 25 pairs; no group oversampling',no_next_actions_created=True),
        settings=dict(additional_bc_updates=12000,lr=.003,epsilon=.02,batch_size=256,seed=17001,
            evaluation_updates=[3000,6000,12000],evaluation_max_controls=100,device='cuda',torch_threads=1,
            optimizer_initialization='Restore the unchanged polished checkpoint BC optimizer state; no restart',
            initial_bc_update_counter=original_updates,critic_updates=0,online_updates=0),
        learner_config=asdict(agent.config),invariant_initial=frozen,results=[],status='running')
    initial_fit=fit_metrics(agent,obs,actions,ec)[0]
    report['initial_fit']=initial_fit
    report['initial_original22_fit']=fit_metrics(agent,obs[:22],actions[:22],ec)[0]
    report['initial_correction3_fit']=fit_metrics(agent,obs[22:],actions[22:],ec)[0]
    write_json(out/'report.json',report)
    batches=np.empty((12000,256),dtype=np.uint8);started=time.monotonic()
    for update in range(1,12001):
        idx=rng.integers(len(obs),size=256);batches[update-1]=idx
        training=agent.bc_step(dict(obs=obs[idx],actions=actions[idx]))
        if update%250==0:
            row=dict(additional_updates=update,**training,wall_seconds=time.monotonic()-started)
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
        if update not in report['settings']['evaluation_updates']:continue
        agent.initialize_std();checkpoint=out/f'checkpoint_{update:05d}.pt';agent.save(checkpoint)
        metadata=write_checkpoint_metadata(checkpoint,ec,agent.config,stage='diagnostic-feedback-imitation-only',
            offline_replay_sha256=report['supervised_data_sha256'])
        metadata.update(production_eligible=False,actor_kind='One actual-MPC-correction supervised round; no SAC',
            initial_checkpoint_sha256=baseline_hash,additional_bc_updates=update,critic_updates=0,online_updates=0)
        write_json(str(checkpoint)+'.metadata.json',metadata)
        fit,means,predictions,targets=fit_metrics(agent,obs,actions,ec)
        group_original=fit_metrics(agent,obs[:22],actions[:22],ec)[0]
        group_correction=fit_metrics(agent,obs[22:],actions[22:],ec)[0]
        np.savez_compressed(out/f'teacher_fit_{update:05d}.npz',observations=obs,actions=actions,
            latent_targets=targets,predicted_means=means,predicted_actions=predictions)
        evaluation,executed,_=run_audit(agent,ec,out/f'audit_{update:05d}',100)
        evaluation['policy_kind']='diagnostic-one-round-feedback-imitation'
        write_json(out/f'audit_{update:05d}'/'report.json',evaluation)
        row=dict(additional_updates=update,total_bc_updates=agent.bc_updates,training=training,all25_fit=fit,
            original22_fit=group_original,correction3_fit=group_correction,evaluation=evaluation,
            checkpoint_sha256=sha256(checkpoint))
        if evaluation['finger_supported_rotation_success']:
            total=len(executed)+20
            continuous_ec=replace(ec,terminate_on_success=False,horizon_steps=max(100,total))
            aftermath,_,_=run_audit(agent,continuous_ec,out/f'continuous_aftermath_{update:05d}',total)
            with np.load(out/f'audit_{update:05d}'/'physics.npz') as a,np.load(out/f'continuous_aftermath_{update:05d}'/'physics.npz') as b:
                assert np.array_equal(a['samples'],b['samples'][:len(a['samples'])])
            aftermath.update(policy_kind='diagnostic-one-round-feedback-imitation',requested_post_success_seconds=2.,
                completed_post_success_seconds=max(0.,aftermath['simulated_seconds']-.1*len(executed)),
                completed_two_seconds_without_failure=bool(aftermath['steps']==total and not aftermath['independent_drop'] and
                    not aftermath['independent_palm_support_violation']))
            row['continuous_policy_aftermath']=aftermath
            write_json(out/f'continuous_aftermath_{update:05d}'/'report.json',aftermath)
        assert invariant_hashes(agent)==frozen and agent.pretrain_updates==agent.online_updates==0
        report['results'].append(row);report['wall_seconds']=time.monotonic()-started
        write_json(out/'report.json',report)
        print(json.dumps(dict(additional_updates=update,original22_fit=group_original,correction3_fit=group_correction,
                              evaluation=evaluation)),flush=True)
    np.savez_compressed(out/'sampled_batch_indices.npz',indices=batches)
    assert sha256(D/'initial_polished.pt')==baseline_hash
    report.update(status='complete',invariant_final=invariant_hashes(agent),
        wall_seconds=time.monotonic()-started,batch_sequence_sha256=sha256(out/'sampled_batch_indices.npz'))
    write_json(out/'report.json',report)


if __name__=='__main__':main()

