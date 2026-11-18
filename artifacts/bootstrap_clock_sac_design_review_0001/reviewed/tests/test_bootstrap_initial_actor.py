"""Mean-only bootstrap imports must preserve actions and restart learning honestly."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mpc_dex.contracts import environment_contract, metadata_path, sha256, write_checkpoint_metadata
from mpc_dex.env import AllegroRotationEnv
from mpc_dex.learning import LearnerConfig, SACAgent
from mpc_dex.replay import ReplayBuffer
from scripts import train_bootstrap

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def fixture(tmp_path):
    config = json.loads((ROOT/'configs/local_sim.json').read_text())
    config['bootstrap'] = {'learner': {'hidden_sizes': [16, 16], 'initial_std': .07}}
    path = tmp_path/'config.json'
    path.write_text(json.dumps(config))
    _, ec = train_bootstrap.read_environment(path)
    torch.manual_seed(731)
    source = SACAgent(60, 16, LearnerConfig(hidden_sizes=(16, 16), ensemble_size=3))
    rng = np.random.default_rng(45)
    batch = dict(obs=rng.normal(size=(4, 60)), actions=rng.uniform(-.8, .8, (4, 16)),
        next_obs=rng.normal(size=(4, 60)), next_actions=rng.uniform(-.8, .8, (4, 16)),
        rewards=np.zeros(4), terminated=np.zeros(4), next_action_valid=np.ones(4))
    source.bc_step(batch)
    source.bc_step(batch)
    source.initialize_std()
    source.pretrain_critic_step(batch)
    source.update(batch)
    checkpoint = tmp_path/'source.pt'
    source.save(checkpoint)
    write_checkpoint_metadata(checkpoint, ec, source.config, stage='unit-test-source')
    args = train_bootstrap.parser().parse_args([
        '--config', str(path), '--steps', '0', '--output', str(tmp_path/'run'),
        '--initial-actor', str(checkpoint), '--num-envs', '1', '--seed', '31',
        '--batch-size', '4', '--replay-capacity', '64'])
    return args, source, checkpoint, ec, batch


def test_import_preserves_exact_means_actions_and_fresh_learning_state(tmp_path):
    args, source, path, ec, batch = fixture(tmp_path)
    digests = sha256(path), sha256(metadata_path(path))
    runner = train_bootstrap.BootstrapRunner(args)
    try:
        agent = runner.agent
        obs = np.vstack([runner.observations[0], batch['obs']]).astype(np.float32)
        with torch.no_grad():
            mean, logstd = agent.actor(torch.as_tensor(obs))
            original_mean, _ = source.actor(torch.as_tensor(obs))
        assert torch.equal(mean, original_mean)
        np.testing.assert_array_equal(agent.act(obs, True), source.act(obs, True))
        torch.testing.assert_close(logstd.exp(), torch.full((len(obs), 16), .07))
        assert agent.std_initialized and agent.bc_updates == source.bc_updates == 2
        assert agent.pretrain_updates == agent.online_updates == 0
        assert runner.environment_steps == runner.successes == runner.eval_successes == 0
        assert runner.completed_episodes == 0 and not runner.success_records
        torch.manual_seed(args.seed)
        fresh = SACAgent(60, 16, agent.config)
        for name in ('critics', 'target_critics'):
            assert all(torch.equal(v, getattr(fresh, name).state_dict()[k])
                       for k, v in getattr(agent, name).state_dict().items())
        assert torch.equal(agent.log_alpha, fresh.log_alpha)
        for name in ('bc_optimizer', 'actor_optimizer', 'critic_optimizer', 'alpha_optimizer'):
            assert not getattr(agent, name).state
            assert getattr(source, name).state  # Source optimizer moments were real and discarded.
        provenance = runner.provenance['initial_actor']
        assert provenance['source_counters'] == dict(bc_updates=2, pretrain_updates=1, online_updates=1)
        state = json.loads((Path(args.output)/'run.json').read_text())
        assert state['bc_updates'] == state['imported_bc_updates'] == 2
        assert state['bootstrap_bc_updates'] == state['pretrain_updates'] == state['online_updates'] == 0
        directory = Path(args.output)/json.loads((Path(args.output)/'latest.json').read_text())['checkpoint_directory']
        metadata = json.loads(metadata_path(directory/'agent.pt').read_text())
        assert metadata['initial_actor'] == provenance
        assert (sha256(path), sha256(metadata_path(path))) == digests
    finally:
        runner.close()


@pytest.mark.parametrize('mismatch,match', [
    ('environment', 'environment/action contract'), ('interface', 'policy interface'),
    ('dimensions', 'dimensions 60/16'), ('hidden', 'hidden sizes'),
    ('metadata', 'configuration disagrees'), ('hash', 'hash mismatch'),
    ('nonfinite', 'finite-value mismatch'),
])
def test_import_rejects_incompatible_or_unbound_source(tmp_path, mismatch, match):
    args, source, path, ec, _ = fixture(tmp_path)
    if mismatch == 'environment':
        write_checkpoint_metadata(path, replace(ec, horizon_steps=ec.horizon_steps+1), source.config, stage='test')
    elif mismatch == 'interface':
        write_checkpoint_metadata(path, ec, source.config, stage='test',
                                  policy_interface={'kind': 'bootstrap_joint_increment'})
    elif mismatch in ('dimensions', 'hidden', 'nonfinite'):
        payload = torch.load(path, weights_only=True)
        if mismatch == 'dimensions': payload['obs_dim'] = 61
        if mismatch == 'hidden': payload['config']['hidden_sizes'] = [32, 32]
        if mismatch == 'nonfinite': payload['actor']['mean_head.weight'][0, 0] = float('nan')
        torch.save(payload, path)
        write_checkpoint_metadata(path, ec, LearnerConfig(**payload['config']), stage='test')
    elif mismatch == 'metadata':
        metadata = json.loads(metadata_path(path).read_text())
        metadata['learner_config']['initial_std'] = .2
        metadata_path(path).write_text(json.dumps(metadata))
    else:
        with path.open('ab') as f: f.write(b'changed')
    with pytest.raises(ValueError, match=match):
        train_bootstrap.BootstrapRunner(args)
    assert not (Path(args.output)/'latest.json').exists()


def add_actual_partial_seed(args, ec, tmp_path, *, bc_updates=0, critic_updates=2):
    env = AllegroRotationEnv(ec)
    replay = ReplayBuffer(8)
    try:
        obs, _ = env.reset()
        for i in range(3):
            action = env.to_action(env.data.ctrl)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            assert not terminated and not truncated
            replay.add_transition(obs, action, reward, next_obs, False, i == 2)
            obs = next_obs
    finally:
        env.close()
    path = tmp_path/'seed.npz'
    replay.save(path)
    metadata = dict(kind='partial-bootstrap-seed', replay_sha256=sha256(path),
        environment_contract=environment_contract(ec), policy_interface={'kind': 'absolute_joint_target'},
        transitions=3, successful_full_turns=0)
    metadata_path(path).write_text(json.dumps(metadata))
    config = json.loads(Path(args.config).read_text())
    config['bootstrap']['seed'] = dict(replay=str(path), bc_updates=bc_updates, critic_updates=critic_updates)
    Path(args.config).write_text(json.dumps(config))


def test_seed_critic_pretraining_allowed_without_mean_retraining(tmp_path):
    args, source, path, ec, batch = fixture(tmp_path)
    add_actual_partial_seed(args, ec, tmp_path)
    runner = train_bootstrap.BootstrapRunner(args)
    try:
        np.testing.assert_array_equal(runner.agent.act(batch['obs'], True), source.act(batch['obs'], True))
        assert runner.agent.bc_updates == 2 and runner.agent.pretrain_updates == 2
        assert runner.agent.online_updates == runner.environment_steps == runner.successes == 0
        assert len(runner.buffers[0]) == 3
        assert runner.agent.critic_optimizer.state and not runner.agent.actor_optimizer.state
    finally:
        runner.close()


def test_import_rejects_nonzero_seed_bc(tmp_path):
    args, _, _, ec, _ = fixture(tmp_path)
    add_actual_partial_seed(args, ec, tmp_path, bc_updates=1)
    with pytest.raises(ValueError, match='zero seed bc_updates'):
        train_bootstrap.BootstrapRunner(args)


def test_resume_restores_checkpoint_without_import_std_reset_or_seed_retraining(tmp_path, monkeypatch):
    args, source, path, ec, batch = fixture(tmp_path)
    add_actual_partial_seed(args, ec, tmp_path)
    runner = train_bootstrap.BootstrapRunner(args)
    # A real unit-test update distinguishes saved learning state from initialization.
    runner.agent.update(batch)
    expected = runner.agent.act(batch['obs'], True).copy()
    with torch.no_grad():
        expected_std = runner.agent.actor(torch.as_tensor(batch['obs'], dtype=torch.float32))[1].clone()
    runner.checkpoint('test-resume')
    runner.close()
    original_hashes = sha256(path), sha256(metadata_path(path))

    def forbidden(*args, **kwargs):
        raise AssertionError('resume must not reinitialize or retrain')

    monkeypatch.setattr(train_bootstrap, 'import_initial_actor', forbidden)
    monkeypatch.setattr(SACAgent, 'initialize_std', forbidden)
    monkeypatch.setattr(SACAgent, 'bc_step', forbidden)
    monkeypatch.setattr(SACAgent, 'pretrain_critic_step', forbidden)
    args.initial_actor = None  # Resume resolves the already-bound source automatically.
    resumed = train_bootstrap.BootstrapRunner(args)
    try:
        np.testing.assert_array_equal(resumed.agent.act(batch['obs'], True), expected)
        with torch.no_grad():
            actual_std = resumed.agent.actor(torch.as_tensor(batch['obs'], dtype=torch.float32))[1]
        assert torch.equal(expected_std, actual_std)
        assert resumed.agent.bc_updates == 2 and resumed.agent.pretrain_updates == 2
        assert resumed.agent.online_updates == 1
        assert len(resumed.buffers[0]) == 3
        assert resumed.agent.actor_optimizer.state and resumed.agent.critic_optimizer.state
        assert (sha256(path), sha256(metadata_path(path))) == original_hashes
    finally:
        resumed.close()


@pytest.mark.parametrize('mutation', ['weights', 'metadata'])
def test_resume_requires_both_original_source_hashes(tmp_path, mutation):
    args, source, path, ec, _ = fixture(tmp_path)
    runner = train_bootstrap.BootstrapRunner(args)
    runner.close()
    if mutation == 'weights':
        source.bc_step(dict(obs=np.zeros((1, 60)), actions=np.zeros((1, 16))))
        source.save(path)
        write_checkpoint_metadata(path, ec, source.config, stage='changed-source')
    else:
        metadata = json.loads(metadata_path(path).read_text())
        metadata['note'] = 'changed metadata only'
        metadata_path(path).write_text(json.dumps(metadata))
    args.initial_actor = None
    with pytest.raises(ValueError, match='source differs at initial_actor'):
        train_bootstrap.BootstrapRunner(args)


@pytest.mark.parametrize('setting', ['threads', 'matmul_precision', 'determinism', 'cudnn_benchmark'])
def test_resume_rejects_changed_actual_numerical_runtime(tmp_path, setting):
    args, _, _, _, _ = fixture(tmp_path)
    runner = train_bootstrap.BootstrapRunner(args)
    saved = runner.provenance['inference_runtime']
    runner.close()
    assert saved['device'] == 'cpu' and saved['torch_threads'] == torch.get_num_threads()
    assert saved['torch_interop_threads'] == torch.get_num_interop_threads()
    assert saved['torch_version'] == str(torch.__version__)
    old_threads = torch.get_num_threads()
    old_precision = torch.get_float32_matmul_precision()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    old_benchmark = torch.backends.cudnn.benchmark
    try:
        if setting == 'threads':
            torch.set_num_threads(old_threads + 1)
        elif setting == 'matmul_precision':
            torch.set_float32_matmul_precision('high' if old_precision != 'high' else 'highest')
        elif setting == 'determinism':
            torch.use_deterministic_algorithms(not old_deterministic)
        else:
            torch.backends.cudnn.benchmark = not old_benchmark
        with pytest.raises(ValueError, match='source differs at inference_runtime'):
            train_bootstrap.BootstrapRunner(args)
    finally:
        torch.set_num_threads(old_threads)
        torch.set_float32_matmul_precision(old_precision)
        torch.use_deterministic_algorithms(old_deterministic, warn_only=old_warn_only)
        torch.backends.cudnn.benchmark = old_benchmark
