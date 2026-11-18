"""Five fixed-mean, matched-noise physical trials; no learning or feedback."""
from pathlib import Path
from dataclasses import asdict
import copy
import hashlib
import json
import math
import random
import sys

sys.dont_write_bytecode = True
D = Path(__file__).resolve().parent
sys.path.insert(0, str(D / 'source'))
import mujoco
import numpy as np
import torch
from mpc_dex.env import AllegroRotationEnv, EnvConfig
from mpc_dex.learning import SACAgent
from mpc_dex.contracts import environment_contract, validate_checkpoint
from scripts.audit_policy import audit_episode, PHYSICS_COLUMNS, write_json

SEEDS = [17021, 17022, 17023, 17024, 17025]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: value.copy() for key, value in data.items()}


def rng():
    return dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore(state):
    torch.set_rng_state(state['cpu'])
    torch.cuda.set_rng_state_all(state['cuda'])


def equal_rng(a, b):
    return torch.equal(a['cpu'], b['cpu']) and len(a['cuda']) == len(b['cuda']) and all(
        torch.equal(x, y) for x, y in zip(a['cuda'], b['cuda']))


def normal_draws(count, initial):
    restore(initial)
    normal = torch.distributions.Normal(torch.zeros(16, device='cuda'), torch.ones(16, device='cuda'))
    return torch.stack([normal.rsample() for _ in range(count)]), rng()


def main():
    assert not (D / 'report.json').exists()
    assert torch.cuda.is_available()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    ec = EnvConfig(**json.loads((D / 'config.json').read_text())['environment'])
    assert ec.terminate_on_success and ec.object_diameter == .04
    metadata = validate_checkpoint(D / 'source_policy.pt', ec)
    assert metadata['policy_interface']['kind'] == 'absolute_joint_target'
    actor = SACAgent.load(D / 'source_policy.pt', 'cuda')
    assert actor.bc_updates == 12000 and actor.pretrain_updates == actor.online_updates == 0
    assert actor.config.min_std == 1e-5
    original_weights = {key:value.detach().cpu().clone() for key,value in actor.actor.state_dict().items()}
    exploratory = copy.deepcopy(actor)
    with torch.no_grad():
        exploratory.actor.log_std_head.weight.zero_()
        exploratory.actor.log_std_head.bias.fill_(math.log(actor.config.min_std))
    fixed_source = load(D / 'deterministic_transitions.npz')
    assert fixed_source['obs'].shape == (9, 60)
    means, stds, noiseless = [], [], []
    with torch.no_grad():
        for obs, recorded_action in zip(fixed_source['obs'], fixed_source['actions']):
            mean, log_std = exploratory.actor(torch.as_tensor(obs, device='cuda', dtype=torch.float32))
            command = actor.act(obs, deterministic=True)
            np.testing.assert_array_equal(command, recorded_action)
            np.testing.assert_array_equal(mean.tanh().cpu().numpy(), recorded_action)
            means.append(mean.clone()); stds.append(log_std.exp().clone()); noiseless.append(command.copy())
    means, stds = torch.stack(means), torch.stack(stds)
    np.savez_compressed(D / 'fixed_nominal_means.npz', observations=fixed_source['obs'],
        raw_latent_means=means.cpu().numpy(), effective_latent_std=stds.cpu().numpy(),
        noiseless_actions=np.asarray(noiseless), actual_recorded_actions=fixed_source['actions'])
    assert torch.equal(stds, stds[0,0].expand_as(stds))
    inputs = {str(path.relative_to(D)):sha(path) for path in D.rglob('*')
              if path.is_file() and '__pycache__' not in path.parts and path.name != 'run.log'}
    write_json(D / 'inputs_manifest.json', inputs)
    report = dict(status='running', purpose='Paired fixed-nominal-latent-mean noise mechanism diagnostic',
        production_eligible=False, learned_policy_success_claim=False, robustness_rate_claim=False,
        training_updates=0, planning_calls=0, physical_trial_limit=5, strict_control_limit=9,
        prespecified_seeds=SEEDS, feedback_used_for_control=False, additional_hold_controls=0,
        requested_latent_std=actor.config.min_std, effective_float32_std=float(stds[0,0]),
        deterministic_commands_bitexact=True, source_checkpoint_sha256=sha(D / 'source_policy.pt'),
        raw_means_source='Exact singleton CUDA forward at saved actual deterministic strict-trajectory observations; not atanh(actions)',
        rng_protocol='Same trial seeds and saved initial Torch CPU/all-CUDA RNG states; one singleton float32 CUDA Normal(mean,std).rsample per actual command; same16-vector shape. Standardized draws regenerated directly from N(0,1), verified through raw mean + eps * std arithmetic.',
        sampling_capture_note='Feedback actions/RNG and fixed-mean draws are reconstructed without physics before each physical trial. The physical trial resets RNG to the same initial state and samples once per executed command.',
        runtime=dict(torch=str(torch.__version__), numpy=np.__version__, mujoco=mujoco.__version__,
            device='cuda', gpu=torch.cuda.get_device_name(), torch_threads=1, torch_interop_threads=1,
            cuda_matmul_tf32=False, environment_contract=environment_contract(ec)), trials=[])
    write_json(D / 'report.json', report)
    print('FIXED_MEANS_VERIFIED ' + json.dumps(dict(commands=9, bitexact=True, effective_std=float(stds[0,0]))), flush=True)
    for seed in SEEDS:
        feedback_path = D / 'feedback' / f'seed_{seed}'
        feedback = load(feedback_path / 'transitions.npz')
        initial_rng = torch.load(feedback_path / 'torch_rng_initial.pt', weights_only=True)
        final_feedback_rng = torch.load(feedback_path / 'torch_rng_final.pt', weights_only=True)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed); random.seed(seed)
        assert equal_rng(rng(), initial_rng)
        # Reconstruct the archived policy calls only, on already recorded observations.
        feedback_means, feedback_stds = [], []
        with torch.no_grad():
            for i, obs in enumerate(feedback['obs']):
                mean, log_std = exploratory.actor(torch.as_tensor(obs, device='cuda', dtype=torch.float32))
                actual = exploratory.act(obs, deterministic=False)
                np.testing.assert_array_equal(actual, feedback['actions'][i])
                np.testing.assert_array_equal(mean.tanh().cpu().numpy(), feedback['conditional_mean_actions'][i])
                np.testing.assert_array_equal(log_std.exp().cpu().numpy(), feedback['latent_std'][i])
                feedback_means.append(mean.clone()); feedback_stds.append(log_std.exp().clone())
        assert equal_rng(rng(), final_feedback_rng)
        feedback_means, feedback_stds = torch.stack(feedback_means), torch.stack(feedback_stds)
        feedback_eps, noise_feedback_final = normal_draws(len(feedback['obs']), initial_rng)
        assert equal_rng(noise_feedback_final, final_feedback_rng)
        feedback_raw = feedback_means + feedback_eps * feedback_stds
        np.testing.assert_array_equal(feedback_raw.tanh().cpu().numpy(), feedback['actions'])
        fixed_eps, fixed_noise_final = normal_draws(9, initial_rng)
        shared = min(9, len(feedback_eps))
        assert torch.equal(fixed_eps[:shared], feedback_eps[:shared])
        restore(initial_rng)
        expected_raw = torch.stack([torch.distributions.Normal(means[i], stds[i]).rsample() for i in range(9)])
        assert equal_rng(rng(), fixed_noise_final)
        assert torch.equal(expected_raw, means + fixed_eps * stds)
        expected_actions = expected_raw.tanh().cpu().numpy()
        output = D / f'seed_{seed}'
        output.mkdir()
        np.savez_compressed(output / 'matched_noise.npz', fixed_raw_means=means.cpu().numpy(),
            fixed_noiseless_actions=np.asarray(noiseless), fixed_latent_std=stds.cpu().numpy(),
            fixed_standardized_noise=fixed_eps.cpu().numpy(), fixed_sampled_latents=expected_raw.cpu().numpy(),
            fixed_sampled_actions=expected_actions, feedback_raw_means=feedback_means.cpu().numpy(),
            feedback_standardized_noise=feedback_eps.cpu().numpy(), feedback_latent_std=feedback_stds.cpu().numpy(),
            feedback_sampled_latents=feedback_raw.cpu().numpy(), feedback_sampled_actions=feedback['actions'])
        # Actual physical trial: fixed means ignore visited observations entirely.
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed); random.seed(seed)
        assert equal_rng(rng(), initial_rng)
        torch.save(rng(), output / 'torch_rng_initial.pt')
        env = AllegroRotationEnv(ec)
        records = {key:[] for key in ['obs','actions','latent_mean','latent_std','sampled_latent','standardized_noise',
                   'next_obs','rewards','terminated','truncated','qpos','qvel','targets','integration_state']}
        original_reset, original_step = env.reset, env.step
        initial = {}
        def integration():
            spec = mujoco.mjtState.mjSTATE_INTEGRATION
            values = np.empty(mujoco.mj_stateSize(env.model,spec))
            mujoco.mj_getState(env.model,env.data,values,spec)
            return values
        def reset(*args,**kwargs):
            result = original_reset(*args,**kwargs)
            initial.update(qpos=env.data.qpos.copy(),qvel=env.data.qvel.copy(),ctrl=env.data.ctrl.copy(),
                joint_positions=env.data.qpos[env.q_ids].copy(),integration_state=integration())
            return result
        def step(action):
            result = original_step(action)
            obs,reward,done,truncated,_ = result
            for key,value in dict(next_obs=obs.copy(),rewards=reward,terminated=done,truncated=truncated,
                qpos=env.data.qpos.copy(),qvel=env.data.qvel.copy(),targets=env.data.ctrl.copy(),integration_state=integration()).items():
                records[key].append(value)
            return result
        class FixedMeanPolicy:
            def act(self, obs, deterministic=True):
                index = len(records['actions'])
                assert index < 9
                raw = torch.distributions.Normal(means[index],stds[index]).rsample()
                assert torch.equal(raw, expected_raw[index])
                action = raw.tanh().cpu().numpy()
                for key,value in dict(obs=obs.copy(),actions=action.copy(),latent_mean=means[index].cpu().numpy(),
                    latent_std=stds[index].cpu().numpy(),sampled_latent=raw.cpu().numpy(),
                    standardized_noise=fixed_eps[index].cpu().numpy()).items(): records[key].append(value)
                return action
        env.reset, env.step = reset, step
        try:
            result, physics, commands = audit_episode(env,FixedMeanPolicy(),seed,9,False)
            result.update(policy_kind='fixed recorded nominal latent means plus matched Gaussian noise; open loop',
                production_eligible=False, learned_policy_success_claim=False, physical_control_cap=9,
                classification=('strict_full_turn' if result['finger_supported_rotation_success'] else
                    'terminal_failure' if result['terminated'] or result['truncated'] else 'partial_at_nine_control_cap'))
            for key,value in initial.items():
                np.testing.assert_array_equal(value,feedback['initial_'+key])
            executed_count = result['steps']
            np.testing.assert_array_equal(records['actions'],expected_actions[:executed_count])
            actual_final_rng = rng()
            _, expected_final_rng = normal_draws(executed_count,initial_rng)
            assert equal_rng(actual_final_rng,expected_final_rng)
            restore(actual_final_rng)
            torch.save(actual_final_rng,output/'torch_rng_final.pt')
            np.savez_compressed(output/'transitions.npz',**{key:np.asarray(value) for key,value in records.items()},
                                **{'initial_'+key:value for key,value in initial.items()})
            np.savez_compressed(output/'physics.npz',samples=physics,columns=np.asarray(PHYSICS_COLUMNS))
            write_json(output/'commands.json',commands);write_json(output/'report.json',result)
            old_report=json.loads((feedback_path/'report.json').read_text())
            delta_open=np.asarray(records['actions']).astype(np.float64)-np.asarray(noiseless)[:executed_count]
            comparison_count=min(executed_count,len(feedback['actions']))
            delta_feedback=feedback['actions'][:comparison_count].astype(np.float64)-np.asarray(noiseless)[:comparison_count]
            summary=dict(seed=seed,open_loop=result,feedback_saved_result=old_report,
                feedback_action_and_final_rng_reconstruction_bitexact=True,standardized_noise_shared_steps=shared,
                standardized_noise_identity_bitexact=True,actual_sampling_matches_prescribed_noise_bitexact=True,
                compared_action_steps=comparison_count,
                open_loop_action_deviation_max_by_step=np.max(abs(delta_open),axis=1).tolist(),
                feedback_action_deviation_max_by_step=np.max(abs(delta_feedback),axis=1).tolist(),
                feedback_mean_action_deviation_max_by_step=np.max(abs(feedback['conditional_mean_actions'][:comparison_count].astype(np.float64)-np.asarray(noiseless)[:comparison_count]),axis=1).tolist(),
                feedback_horizon_caveat='Existing feedback trials allowed100controls; seed17023 terminates at10, beyond the present9-control cap. Comparison of outcomes is not a success-rate estimate.')
            report['trials'].append(summary)
            write_json(D/'report.json',report)
            print(json.dumps(dict(seed=seed,steps=executed_count,classification=result['classification'],
                force=result['final_force_qualified_angle'],drop=result['independent_drop'],
                palm=result['independent_palm_support_violation'],shared_draws_bitexact=shared)),flush=True)
        finally:
            env.close()
    assert all(torch.equal(value,actor.actor.state_dict()[key].cpu()) for key,value in original_weights.items())
    assert actor.bc_updates==12000 and actor.pretrain_updates==actor.online_updates==0
    assert all(sha(D/name)==digest for name,digest in inputs.items())
    report.update(status='complete',source_checkpoint_and_all_inputs_unchanged=True,
        interpretation='Five prespecified paired mechanism examples, not a learned-policy success or robustness-rate claim.')
    write_json(D/'report.json',report)
    write_json(D/'manifest.json',{str(path.relative_to(D)):sha(path) for path in sorted(D.rglob('*'))
        if path.is_file() and '__pycache__' not in path.parts and path.name not in ('manifest.json','run.log')})


if __name__=='__main__':
    with torch.no_grad():
        main()
