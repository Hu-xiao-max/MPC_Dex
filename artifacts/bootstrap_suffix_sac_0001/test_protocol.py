"""Synthetic unit fixtures only: no environment, policy, training or success proof."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('suffix_runner_review', HERE / 'run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_reward_normalization_preserves_raw_replay_and_episode_return():
    replay = runner.ReplayBuffer(16, 3)
    obs = np.zeros(60, np.float32)
    action = np.zeros(16, np.float32)
    replay.add_transition(obs, action, 240., obs + 1, False)
    replay.add_transition(obs + 1, action + .1, -80., obs + 2, True)
    before = {k: v.copy() for k, v in replay._arrays.items()}
    batch, _ = runner.normalized_batch(replay, {'batch_size': 20, 'raw_reward_divisor': 200.})
    assert set(batch['rewards'].tolist()) == set(np.asarray([1.2, -.4], np.float32).tolist())
    for key in before:
        np.testing.assert_array_equal(before[key], replay._arrays[key])
    np.testing.assert_array_equal(replay._arrays['episode_return'][:2], [160., 160.])
    np.testing.assert_array_equal(replay.transitions()['next_actions'][0], action + .1)
    np.testing.assert_array_equal(replay.transitions()['next_action_valid'], [True, False])


def test_empirical_targets_exclude_initial_entropy_and_use_next_action_entropy():
    arrays = dict(obs=np.zeros((3, 60), np.float32), actions=np.zeros((3, 16), np.float32),
                  rewards=np.asarray([999., 200., 400.]), log_prob=np.asarray([np.nan, 100., 2.]))
    episode = SimpleNamespace(arrays=lambda: arrays, prefix=1, term=True, trunc=False, number=4)
    out = runner.terminal_returns(episode, .9, .05/200, 200., .02/200)
    np.testing.assert_allclose(out['soft_q_normalized'], [1 + .9 * (2 - .05/200 * 2), 2], rtol=0, atol=1e-15)
    np.testing.assert_allclose(out['soft_q_pretrain_normalized'], [1 + .9 * (2 - .02/200 * 2), 2], rtol=0, atol=1e-15)
    assert len(out['obs']) == 2  # Recorded prefix is excluded.
    arrays['log_prob'][1] = -1e6
    repeated = runner.terminal_returns(episode, .9, .05/200, 200., .02/200)
    np.testing.assert_array_equal(out['soft_q_normalized'], repeated['soft_q_normalized'])
    episode.term = False
    assert runner.terminal_returns(episode, .9, .05/200, 200., .02/200) is None
    episode.term = True
    episode.trunc = True
    assert runner.terminal_returns(episode, .9, .05/200, 200., .02/200) is None


def test_unfinished_suffix_remains_open_across_serialization(tmp_path):
    replay = runner.ReplayBuffer(16, 3)
    obs = np.zeros(60, np.float32)
    action = np.zeros(16, np.float32)
    replay.add_transition(obs, action, 1., obs + 1, False, False)
    replay.save(tmp_path / 'replay.npz')
    loaded = runner.ReplayBuffer.load(tmp_path / 'replay.npz')
    assert loaded._pending is not None and loaded.episode_count == 0
    row = loaded.transitions()
    assert not row['terminated'][0] and not row['truncated'][0] and not row['next_action_valid'][0]
    loaded.add_transition(obs + 1, action + .2, 2., obs + 2, True)
    np.testing.assert_array_equal(loaded.transitions()['next_actions'][0], action + .2)
    assert loaded.episode_count == 1


def test_recorded_transition_successors_never_invent_final_action():
    episode = object.__new__(runner.Episode)
    episode.rows = [dict(actions=np.zeros(16, np.float32), terminated=False, truncated=False),
                    dict(actions=np.ones(16, np.float32) / 2, terminated=False, truncated=False)]
    arrays = episode.arrays()
    np.testing.assert_array_equal(arrays['next_actions'][0], episode.rows[1]['actions'])
    assert arrays['next_action_valid'].tolist() == [True, False]
    assert not arrays['terminated'].any() and not arrays['truncated'].any()
