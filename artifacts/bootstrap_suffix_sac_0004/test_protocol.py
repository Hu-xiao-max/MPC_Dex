"""Synthetic unit fixtures only: no environment, physics or success proof; one tiny CPU agent."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('suffix_runner_0004_review', HERE / 'run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
torch = runner.torch


def small_buffers(seed_demo=5, seed_online=3):
    demo = runner.ReplayBuffer(16, seed_demo)
    online = runner.ReplayBuffer(16, seed_online)
    obs = np.zeros(60, np.float32)
    action = np.zeros(16, np.float32)
    demo.add_episode(dict(obs=np.stack([obs, obs + 1]), actions=np.stack([action, action + .1]),
                          rewards=np.asarray([240., -80.]), next_obs=np.stack([obs + 1, obs + 2]),
                          terminated=np.asarray([False, True])))
    online.add_transition(obs + 5, action + .2, 20., obs + 6, False)
    online.add_transition(obs + 6, action + .3, 40., obs + 7, True)
    return demo, online


def test_mixed_batch_with_origin_reproduces_sample_mixed_and_marks_rows():
    demo, online = small_buffers()
    demo_copy, online_copy = small_buffers()
    reference = runner.sample_mixed(demo_copy, online_copy, 10, .3)
    batch = runner.mixed_batch_with_origin(demo, online, 10, .3)
    for key in reference:
        np.testing.assert_array_equal(batch[key], reference[key])
    assert batch['demonstration'].sum() == 3 and batch['demonstration'].dtype == bool
    rewards = batch['rewards'].astype(np.float64)
    assert np.all(np.isclose(rewards[batch['demonstration']], 240.) | np.isclose(rewards[batch['demonstration']], -80.))
    assert np.all(np.isclose(rewards[~batch['demonstration']], 20.) | np.isclose(rewards[~batch['demonstration']], 40.))


def test_normalized_batch_keeps_mask_and_scales_only_rewards():
    demo, online = small_buffers()
    batch, stats = runner.normalized_batch(demo, online, {'batch_size': 10, 'raw_reward_divisor': 200., 'offline_fraction': .3})
    assert stats['batch_demonstration_rows'] == 3 and 'demonstration' in batch
    rewards = batch['rewards'].astype(np.float64)
    assert np.all(np.isclose(rewards[batch['demonstration']], 1.2) | np.isclose(rewards[batch['demonstration']], -.4))


def tiny_agent():
    agent = runner.SACAgent(60, 16, device='cpu')
    agent.initialize_std()
    return agent


def test_entropy_masked_step_removes_entropy_only_for_demonstration_rows():
    torch.manual_seed(0)
    agent = tiny_agent()
    obs = np.random.default_rng(1).standard_normal((4, 60)).astype(np.float32)
    actions = np.full((4, 16), .9, np.float32)  # improbable under std .3 around ~0 mean
    next_actions = actions.copy()
    batch = dict(obs=obs, actions=actions, rewards=np.asarray([1., 1., 1., 1.], np.float32), next_obs=obs + .5,
                 terminated=np.asarray([False, False, True, False]), truncated=np.zeros(4, bool),
                 next_actions=next_actions, next_action_valid=np.asarray([True, True, False, False]),
                 demonstration=np.asarray([True, False, True, True]))
    captured = {}
    original_step = agent._critic_step

    def capture(values, targets):
        captured['values'], captured['targets'] = values, targets.clone()
        return original_step(values, targets)
    agent._critic_step = capture
    with torch.no_grad():
        next_obs = torch.as_tensor(batch['next_obs'][:2])
        next_act = torch.as_tensor(next_actions[:2])
        next_q = agent.target_critics(next_obs, next_act).min(dim=0).values
        log_prob = agent.actor.log_prob(next_obs, next_act)
    result = runner.entropy_masked_pretrain_step(agent, batch, {})
    targets = captured['targets']
    assert result['pretrain_samples'] == 3 and result['entropy_free_bootstrap_rows'] == 1 and result['demonstration_rows_eligible'] == 2
    # Eligible rows in order: row0 (demo, bootstrap), row1 (online, bootstrap), row2 (demo, terminal).
    assert torch.isclose(targets[0], 1 + agent.config.discount * next_q[0], atol=1e-5)
    assert torch.isclose(targets[1], 1 + agent.config.discount * (next_q[1] - agent.config.pretrain_alpha * log_prob[1]), atol=1e-5)
    assert torch.isclose(targets[2], torch.tensor(1.))
    assert abs(float(log_prob[1])) > 10  # the entropy term is not negligible in this fixture
    assert agent.pretrain_updates == 1 and agent.online_updates == 0


def test_entropy_masked_step_matches_archived_step_when_no_demonstration_rows():
    torch.manual_seed(0)
    reference = tiny_agent()
    subject = copy.deepcopy(reference)
    obs = np.random.default_rng(2).standard_normal((6, 60)).astype(np.float32)
    actions = (np.random.default_rng(3).standard_normal((6, 16)) * .3).clip(-.99, .99).astype(np.float32)
    batch = dict(obs=obs, actions=actions, rewards=np.linspace(-1, 1, 6, dtype=np.float32), next_obs=obs * .5,
                 terminated=np.asarray([False] * 5 + [True]), truncated=np.zeros(6, bool),
                 next_actions=actions[::-1].copy(), next_action_valid=np.asarray([True] * 5 + [False]))
    ref_result = reference.pretrain_critic_step(dict(batch))
    sub_result = runner.entropy_masked_pretrain_step(subject, dict(batch, demonstration=np.zeros(6, bool)), {})
    assert np.isclose(ref_result['critic_loss'], sub_result['critic_loss'])
    for a, b in zip(reference.critics.state_dict().values(), subject.critics.state_dict().values()):
        assert torch.equal(a, b)
    for a, b in zip(reference.target_critics.state_dict().values(), subject.target_critics.state_dict().values()):
        assert torch.equal(a, b)
    assert reference.pretrain_updates == subject.pretrain_updates == 1


def test_demonstration_gate_requires_exact_terminal_episodes():
    demo, _ = small_buffers()
    assert runner.demonstration_gate(demo, dict(demonstration_rows=2, demonstration_episodes=1))
    with pytest.raises(AssertionError):
        runner.demonstration_gate(demo, dict(demonstration_rows=3, demonstration_episodes=1))


def test_empirical_targets_exclude_initial_entropy_and_use_next_action_entropy():
    arrays = dict(obs=np.zeros((3, 60), np.float32), actions=np.zeros((3, 16), np.float32),
                  rewards=np.asarray([999., 200., 400.]), log_prob=np.asarray([np.nan, 100., 2.]))
    episode = SimpleNamespace(arrays=lambda: arrays, prefix=1, term=True, trunc=False, number=4)
    out = runner.terminal_returns(episode, .9, .05/200, 200., .02/200)
    np.testing.assert_allclose(out['soft_q_normalized'], [1 + .9 * (2 - .05/200 * 2), 2], rtol=0, atol=1e-15)


def test_actual_demonstration_rows_load_and_share_the_handoff_observation():
    protocol = dict(prefix_controls=6, demonstration_rows_mpc=5, demonstration_rows_recorded=3,
                    demonstration_rows=8, demonstration_episodes=2)
    demonstrations = runner.load_demonstrations(protocol)
    np.testing.assert_array_equal(demonstrations[0]['arrays']['obs'][0], demonstrations[1]['arrays']['obs'][0])
    demo = runner.ReplayBuffer(64, 0)
    for item in demonstrations:
        demo.add_episode(item['arrays'])
    assert runner.demonstration_gate(demo, protocol)


def test_actual_checkpoint_load_save_preserves_entire_payload_without_update(tmp_path):
    original = torch.load(HERE / 'initial.pt', map_location='cpu', weights_only=True)
    agent = runner.SACAgent.load(HERE / 'initial.pt', 'cpu')
    agent.save(tmp_path / 'unchanged.pt')
    runner.assert_tree_equal(original, torch.load(tmp_path / 'unchanged.pt', map_location='cpu', weights_only=True))
    assert runner.optimizer_steps(agent.critic_optimizer) == [2500]
