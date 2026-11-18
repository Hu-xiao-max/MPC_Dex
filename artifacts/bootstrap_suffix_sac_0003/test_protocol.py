"""Synthetic unit fixtures only: no environment, policy, training or success proof."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('suffix_runner_0003_review', HERE / 'run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def small_buffers():
    demo = runner.ReplayBuffer(16, 5)
    online = runner.ReplayBuffer(16, 3)
    obs = np.zeros(60, np.float32)
    action = np.zeros(16, np.float32)
    demo.add_episode(dict(obs=np.stack([obs, obs + 1]), actions=np.stack([action, action + .1]),
                          rewards=np.asarray([240., -80.]), next_obs=np.stack([obs + 1, obs + 2]),
                          terminated=np.asarray([False, True])))
    online.add_transition(obs + 5, action + .2, 20., obs + 6, False)
    online.add_transition(obs + 6, action + .3, 40., obs + 7, True)
    return demo, online


def test_mixed_batch_normalizes_rewards_without_touching_either_buffer():
    demo, online = small_buffers()
    before = {k: v.copy() for k, v in demo._arrays.items()}, {k: v.copy() for k, v in online._arrays.items()}
    batch, stats = runner.normalized_batch(demo, online, {'batch_size': 10, 'raw_reward_divisor': 200., 'offline_fraction': .3})
    assert stats['batch_demonstration_rows'] == 3 and len(batch['rewards']) == 10
    rewards = batch['rewards'].astype(np.float64)
    is_demo = np.isclose(rewards, 1.2) | np.isclose(rewards, -.4)
    is_online = np.isclose(rewards, .1) | np.isclose(rewards, .2)
    assert (is_demo | is_online).all() and int(is_demo.sum()) == 3
    for arrays, saved in zip((demo, online), before):
        for key in saved:
            np.testing.assert_array_equal(saved[key], arrays._arrays[key])
    np.testing.assert_array_equal(demo.transitions()['next_actions'][0], np.zeros(16, np.float32) + .1)
    np.testing.assert_array_equal(demo.transitions()['next_action_valid'], [True, False])


def test_demonstration_gate_requires_exact_terminal_episodes():
    demo, _ = small_buffers()
    protocol = dict(demonstration_rows=2, demonstration_episodes=1)
    assert runner.demonstration_gate(demo, protocol)
    with pytest.raises(AssertionError):
        runner.demonstration_gate(demo, dict(demonstration_rows=3, demonstration_episodes=1))
    open_demo = runner.ReplayBuffer(16, 1)
    open_demo.add_transition(np.zeros(60, np.float32), np.zeros(16, np.float32), 1., np.ones(60, np.float32), False)
    with pytest.raises(AssertionError):
        runner.demonstration_gate(open_demo, dict(demonstration_rows=1, demonstration_episodes=0))


def test_empirical_targets_exclude_initial_entropy_and_use_next_action_entropy():
    arrays = dict(obs=np.zeros((3, 60), np.float32), actions=np.zeros((3, 16), np.float32),
                  rewards=np.asarray([999., 200., 400.]), log_prob=np.asarray([np.nan, 100., 2.]))
    episode = SimpleNamespace(arrays=lambda: arrays, prefix=1, term=True, trunc=False, number=4)
    out = runner.terminal_returns(episode, .9, .05/200, 200., .02/200)
    np.testing.assert_allclose(out['soft_q_normalized'], [1 + .9 * (2 - .05/200 * 2), 2], rtol=0, atol=1e-15)
    assert len(out['obs']) == 2
    episode.trunc = True
    assert runner.terminal_returns(episode, .9, .05/200, 200., .02/200) is None


def test_recorded_transition_successors_never_invent_final_action():
    episode = object.__new__(runner.Episode)
    episode.rows = [dict(actions=np.zeros(16, np.float32), terminated=False, truncated=False),
                    dict(actions=np.ones(16, np.float32) / 2, terminated=False, truncated=False)]
    arrays = episode.arrays()
    np.testing.assert_array_equal(arrays['next_actions'][0], episode.rows[1]['actions'])
    assert arrays['next_action_valid'].tolist() == [True, False]


def test_actual_demonstration_rows_load_and_share_the_handoff_observation():
    # Real frozen inputs only: no inference, physics or update.
    protocol = dict(prefix_controls=6, demonstration_rows_mpc=5, demonstration_rows_recorded=3,
                    demonstration_rows=8, demonstration_episodes=2)
    demonstrations = runner.load_demonstrations(protocol)
    assert [d['name'] for d in demonstrations] == ['mpc', 'recorded']
    np.testing.assert_array_equal(demonstrations[0]['arrays']['obs'][0], demonstrations[1]['arrays']['obs'][0])
    demo = runner.ReplayBuffer(64, 0)
    for item in demonstrations:
        demo.add_episode(item['arrays'])
    assert runner.demonstration_gate(demo, protocol)
    rows = demo.transitions()
    assert rows['terminated'].tolist() == [False] * 4 + [True] + [False] * 2 + [True]
    np.testing.assert_array_equal(rows['next_actions'][:4], rows['actions'][1:5])


def test_actual_checkpoint_load_save_preserves_entire_payload_without_update(tmp_path):
    original = runner.torch.load(HERE / 'initial.pt', map_location='cpu', weights_only=True)
    agent = runner.SACAgent.load(HERE / 'initial.pt', 'cpu')
    agent.save(tmp_path / 'unchanged.pt')
    copied = runner.torch.load(tmp_path / 'unchanged.pt', map_location='cpu', weights_only=True)
    runner.assert_tree_equal(original, copied)
    assert runner.optimizer_steps(agent.critic_optimizer) == [2500]
    assert runner.optimizer_steps(agent.actor_optimizer) == [2000]
    assert runner.optimizer_steps(agent.alpha_optimizer) == [2000]
