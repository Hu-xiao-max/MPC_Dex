import numpy as np
import pytest

from mpc_dex.replay import ReplayBuffer, sample_mixed


def episode(size=4, offset=0, terminated=False, total_return=0):
    states = np.arange(size + 1, dtype=np.float32)[:, None] + offset
    return {"obs": states[:-1], "next_obs": states[1:],
            "actions": np.arange(size, dtype=np.float32)[:, None] / max(size, 1),
            "rewards": np.full(size, total_return / size, dtype=np.float32),
            "terminated": np.array([False] * (size - 1) + [terminated]),
            "truncated": np.array([False] * (size - 1) + [not terminated])}


def test_next_actions_never_cross_reset_or_truncation():
    replay = ReplayBuffer(20)
    replay.add_episode(episode(3, 0, False))
    replay.add_episode(episode(2, 100, True))
    data = replay.transitions()
    np.testing.assert_array_equal(data["next_action_valid"], [True, True, False, True, False])
    np.testing.assert_array_equal(data["terminated"], [False, False, False, False, True])
    np.testing.assert_array_equal(data["next_actions"][:2], data["actions"][1:3])
    assert replay.episode_count == 2


def test_explicit_time_limit_next_action_preserved():
    replay = ReplayBuffer(20)
    demo = episode(3)
    demo["next_actions"] = np.concatenate((demo["actions"][1:], [[.9]]))
    demo["next_action_valid"] = np.ones(3, dtype=bool)
    replay.add_episode(demo)
    data = replay.transitions()
    assert data["truncated"][-1] and not data["terminated"][-1]
    assert data["next_action_valid"][-1]
    assert data["next_actions"][-1, 0] == pytest.approx(.9)


def test_weighted_episode_sampling_avoids_length_bias_and_uses_return_softmax():
    replay = ReplayBuffer(100, seed=9)
    replay.add_episode(episode(1, 0, total_return=0))
    replay.add_episode(episode(9, 100, total_return=np.log(3)))
    samples = replay.sample(30_000, weighted_episodes=True)
    # Episode probabilities are 1:3 despite episode lengths being 1:9.
    assert np.mean(samples["obs"][:, 0] >= 100) == pytest.approx(.75, abs=.012)
    equalized = replay.sample(30_000, weighted_episodes=True, temperature=1e9)
    assert np.mean(equalized["obs"][:, 0] >= 100) == pytest.approx(.5, abs=.012)
    assert replay.sample(1)["obs"].shape == (1, 1)


def test_online_resume_links_only_actual_following_action(tmp_path):
    replay = ReplayBuffer(5)
    replay.add_transition([0], [.2], 1, [1], False)
    path = tmp_path / "replay.npz"
    replay.save(path)
    with np.load(path, allow_pickle=False) as archive:
        assert all(archive[key].dtype != object for key in archive.files)
    restored = ReplayBuffer.load(path)
    restored.add_transition([1], [.3], 2, [2], False, truncated=True)
    restored.add_transition([100], [.8], 3, [101], True)
    data = restored.transitions()
    np.testing.assert_array_equal(data["next_action_valid"], [True, False, False])
    assert data["next_actions"][0, 0] == pytest.approx(.3)
    assert restored.episode_count == 2


def test_ring_eviction_and_rng_save_roundtrip(tmp_path):
    replay = ReplayBuffer(5, seed=3)
    replay.add_episode(episode(4, 0))
    replay.add_episode(episode(3, 100))
    np.testing.assert_array_equal(replay.transitions()["obs"][:, 0], [2, 3, 100, 101, 102])
    replay.save(tmp_path / "replay.npz")
    restored = ReplayBuffer.load(tmp_path / "replay.npz")
    for key, value in replay.transitions().items():
        np.testing.assert_array_equal(value, restored.transitions()[key])
    resumed_sample = restored.sample(32)
    for key, value in replay.sample(32).items():
        np.testing.assert_array_equal(value, resumed_sample[key])
    assert restored.episode_count == 2
    restored.add_episode(episode(2, 200))
    np.testing.assert_array_equal(restored.transitions()["obs"][:, 0], [100, 101, 102, 200, 201])


def test_open_episode_larger_than_capacity_can_resume(tmp_path):
    replay = ReplayBuffer(2)
    for i in range(4):
        replay.add_transition([i], [.1], i, [i + 1], False)
    replay.save(tmp_path / "long.npz")
    restored = ReplayBuffer.load(tmp_path / "long.npz")
    restored.add_transition([4], [.4], 4, [5], True)
    assert restored.episode_count == 1
    np.testing.assert_array_equal(restored.transitions()["obs"][:, 0], [3, 4])
    assert restored.sample(10, weighted_episodes=True)["obs"].shape == (10, 1)


def test_mix_exact_counts_does_not_mutate_fixed_offline_replay():
    offline, online = ReplayBuffer(10), ReplayBuffer(10)
    offline.add_episode(episode(2, 0))
    online.add_episode(episode(3, 100))
    before = offline.transitions()
    mixed = sample_mixed(offline, online, 256)
    assert np.sum(mixed["obs"][:, 0] < 100) == 77
    assert np.sum(mixed["obs"][:, 0] >= 100) == 179
    for key, value in before.items():
        np.testing.assert_array_equal(value, offline.transitions()[key])
    with pytest.raises(ValueError, match="nonempty"):
        sample_mixed(offline, ReplayBuffer(10), 256)


def test_rejects_internal_boundaries_and_unmarked_resets():
    replay = ReplayBuffer(20)
    invalid = episode(3)
    invalid["terminated"][0] = True
    with pytest.raises(ValueError, match="internal"):
        replay.add_episode(invalid)
    assert len(replay) == 0
    replay.add_transition([0], [.1], 0, [1], False)
    with pytest.raises(ValueError, match="reset boundaries"):
        replay.add_transition([100], [.1], 0, [101], False)
    with pytest.raises(ValueError, match="completed episodes"):
        replay.sample(10, weighted_episodes=True)


def test_empty_save_load(tmp_path):
    replay = ReplayBuffer(10)
    replay.save(tmp_path / "empty.npz")
    restored = ReplayBuffer.load(tmp_path / "empty.npz")
    assert len(restored) == restored.episode_count == 0
    restored.add_episode(episode())
    assert len(restored) == 4
