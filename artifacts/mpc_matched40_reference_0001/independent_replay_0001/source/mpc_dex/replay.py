"""Replay with explicit episode boundaries and recorded next-action provenance.

Episode-weighted sampling first selects an episode with softmax(return / T),
then a retained transition uniformly within it. T=1 is an explicit independent
implementation choice: the paper does not publish a sampling temperature.
Saved files contain ordinary NumPy arrays and JSON, never pickled objects.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


_BATCH_KEYS = ("obs", "actions", "rewards", "next_obs", "terminated", "truncated",
               "next_actions", "next_action_valid")


class ReplayBuffer:
    def __init__(self, capacity: int = 1_000_000, seed: int = 0):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self._arrays: dict[str, np.ndarray] = {}
        self._size = self._position = self._episode_id = 0
        self._episode_return = 0.0
        self._episode_slots: list[int] = []
        self._pending: int | None = None

    def __len__(self) -> int:
        return self._size

    @property
    def episode_count(self) -> int:
        """Number of completed episodes with at least one retained transition."""
        if not self._size:
            return 0
        mask = self._arrays["episode_complete"][:self._size]
        return len(np.unique(self._arrays["episode_id"][:self._size][mask]))

    def _allocate(self, obs_dim: int, action_dim: int) -> None:
        shapes = {"obs": (obs_dim,), "next_obs": (obs_dim,), "actions": (action_dim,),
                  "next_actions": (action_dim,)}
        for key in _BATCH_KEYS + ("episode_id", "episode_return", "episode_complete"):
            dtype = (np.bool_ if key in ("terminated", "truncated", "next_action_valid", "episode_complete")
                     else np.int64 if key == "episode_id" else np.float32)
            self._arrays[key] = np.empty((self.capacity, *shapes.get(key, ())), dtype=dtype)

    @staticmethod
    def _vector(value, name: str) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32)
        if value.ndim != 1 or not value.size or not np.isfinite(value).all():
            raise ValueError(f"{name} must be a finite nonempty vector")
        return value

    def add_transition(self, obs, action, reward: float, next_obs, terminated: bool,
                       truncated: bool = False, next_action=None,
                       next_action_valid: bool | None = None) -> None:
        obs = self._vector(obs, "obs")
        action = self._vector(action, "action")
        next_obs = self._vector(next_obs, "next_obs")
        if obs.shape != next_obs.shape or not np.isfinite(reward):
            raise ValueError("inconsistent observations or non-finite reward")
        if np.any(np.abs(action) > 1):
            raise ValueError("action exceeds normalized [-1, 1] bounds")
        valid = next_action is not None if next_action_valid is None else bool(next_action_valid)
        if valid and next_action is None:
            raise ValueError("valid next action requires a recorded next action")
        next_action = np.zeros_like(action) if next_action is None else self._vector(next_action, "next_action")
        if next_action.shape != action.shape or np.any(np.abs(next_action) > 1):
            raise ValueError("invalid next action shape or bounds")
        if not self._arrays:
            self._allocate(obs.size, action.size)
        if self._arrays["obs"].shape[1:] != obs.shape or self._arrays["actions"].shape[1:] != action.shape:
            raise ValueError("transition dimensions differ from replay dimensions")
        if self._pending is not None:
            previous = self._pending
            if not np.allclose(self._arrays["next_obs"][previous], obs, atol=1e-6, rtol=1e-5):
                raise ValueError("successive observations do not match; mark reset boundaries with truncated/terminated")
            if self._arrays["next_action_valid"][previous] and not np.allclose(
                    self._arrays["next_actions"][previous], action, atol=1e-6, rtol=1e-5):
                raise ValueError("recorded next action disagrees with the following executed action")
            self._arrays["next_actions"][previous] = action
            self._arrays["next_action_valid"][previous] = True
        index = self._position
        row = {"obs": obs, "actions": action, "rewards": reward, "next_obs": next_obs,
               "terminated": terminated, "truncated": truncated,
               "next_actions": next_action, "next_action_valid": valid and not bool(terminated),
               "episode_id": self._episode_id, "episode_return": 0.0, "episode_complete": False}
        for key, value in row.items():
            self._arrays[key][index] = value
        self._position = (index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        self._episode_return += float(reward)
        self._episode_slots.append(index)
        self._pending = index
        if terminated or truncated:
            self._finish_episode()

    # A conventional alias for callers collecting individual online transitions.
    add = add_transition

    def _finish_episode(self) -> None:
        slots = np.asarray(self._episode_slots[-self.capacity:], dtype=np.int64)
        self._arrays["episode_return"][slots] = self._episode_return
        self._arrays["episode_complete"][slots] = True
        self._episode_id += 1
        self._episode_return = 0.0
        self._episode_slots = []
        self._pending = None

    def add_episode(self, episode: dict[str, np.ndarray]) -> None:
        """Add one episode, deriving next actions only within its boundary.

        Required arrays: obs, actions, rewards, next_obs, terminated. Optional:
        truncated, next_actions, next_action_valid. An explicitly recorded next
        action may bootstrap a final time-limit state; it is never invented.
        """
        if self._pending is not None:
            raise ValueError("finish the open online episode before adding an episode")
        required = ("obs", "actions", "rewards", "next_obs", "terminated")
        values = {key: np.asarray(episode[key]) for key in required}
        size = len(values["obs"])
        if size == 0:
            raise ValueError("episode cannot be empty")
        if any(len(value) != size for value in values.values()):
            raise ValueError("episode arrays have inconsistent lengths")
        if values["obs"].ndim != 2 or values["actions"].ndim != 2:
            raise ValueError("episode observations and actions must be matrices")
        if values["next_obs"].shape != values["obs"].shape:
            raise ValueError("episode next observations have the wrong shape")
        for key in ("rewards", "terminated"):
            if values[key].size != size:
                raise ValueError(f"{key} must have one value per transition")
            values[key] = values[key].reshape(size)
        truncated = np.asarray(episode.get("truncated", np.zeros(size, dtype=bool))).reshape(-1)
        if truncated.shape != (size,):
            raise ValueError("truncated must have one value per transition")
        if np.any(values["terminated"][:-1]) or np.any(truncated[:-1]):
            raise ValueError("add_episode cannot contain an internal episode boundary")
        if not all(np.isfinite(value).all() for value in values.values()):
            raise ValueError("episode contains non-finite values")
        if not np.allclose(values["next_obs"][:-1], values["obs"][1:], atol=1e-6, rtol=1e-5):
            raise ValueError("episode does not contain consecutive transitions")
        if np.any(np.abs(values["actions"]) > 1):
            raise ValueError("episode actions exceed normalized bounds")
        explicit_actions = episode.get("next_actions")
        explicit_valid = episode.get("next_action_valid")
        if explicit_valid is not None and explicit_actions is None:
            raise ValueError("next_action_valid requires next_actions")
        if explicit_actions is not None:
            explicit_actions = np.asarray(explicit_actions)
            if explicit_actions.shape != values["actions"].shape or not np.isfinite(explicit_actions).all():
                raise ValueError("invalid explicit next actions")
            if np.any(np.abs(explicit_actions) > 1):
                raise ValueError("explicit next actions exceed normalized bounds")
            explicit_valid = np.ones(size, dtype=bool) if explicit_valid is None else np.asarray(explicit_valid).reshape(-1)
            if explicit_valid.shape != (size,):
                raise ValueError("next_action_valid must have one value per transition")
            if not np.allclose(explicit_actions[:-1][explicit_valid[:-1].astype(bool)],
                               values["actions"][1:][explicit_valid[:-1].astype(bool)], atol=1e-6, rtol=1e-5):
                raise ValueError("explicit next actions disagree with executed actions")
        for i in range(size):
            self.add_transition(values["obs"][i], values["actions"][i], float(values["rewards"][i]),
                                values["next_obs"][i], bool(values["terminated"][i]), bool(truncated[i]),
                                None if explicit_actions is None else explicit_actions[i],
                                None if explicit_valid is None else bool(explicit_valid[i]))
        if self._pending is not None:
            self._finish_episode()

    def sample(self, batch_size: int, weighted_episodes: bool = False,
               temperature: float = 1.0) -> dict[str, np.ndarray]:
        if batch_size <= 0 or not self._size:
            raise ValueError("sampling requires a nonempty buffer and positive batch size")
        if weighted_episodes:
            if not np.isfinite(temperature) or temperature <= 0:
                raise ValueError("episode sampling temperature must be positive and finite")
            eligible = np.flatnonzero(self._arrays["episode_complete"][:self._size])
            if not len(eligible):
                raise ValueError("episode-weighted sampling requires completed episodes")
            _, first, inverse, counts = np.unique(self._arrays["episode_id"][eligible],
                                                 return_index=True, return_inverse=True, return_counts=True)
            returns = self._arrays["episode_return"][eligible[first]].astype(np.float64)
            logits = (returns - returns.max()) / temperature
            episode_weights = np.exp(logits)
            episode_weights /= episode_weights.sum()
            # Dividing by episode length prevents long demonstrations receiving
            # extra probability beyond the requested return-based distribution.
            probabilities = episode_weights[inverse] / counts[inverse]
            indices = self.rng.choice(eligible, size=batch_size, p=probabilities)
        else:
            indices = self.rng.integers(self._size, size=batch_size)
        return {key: self._arrays[key][indices].copy() for key in _BATCH_KEYS}

    def transitions(self) -> dict[str, np.ndarray]:
        """Return a chronological copy, primarily for inspection/export."""
        indices = self._ordered_indices()
        return {key: self._arrays[key][indices].copy() for key in _BATCH_KEYS} if self._arrays else {}

    def _ordered_indices(self) -> np.ndarray:
        if self._size < self.capacity:
            return np.arange(self._size)
        return (np.arange(self._size) + self._position) % self.capacity

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"schema_version": 1, "capacity": self.capacity, "episode_id": self._episode_id,
                    "episode_return": self._episode_return, "open_episode": self._pending is not None,
                    "position": self._position,
                    "rng_state": self.rng.bit_generator.state}
        # Preserve physical ring layout as well as RNG state for exact sampling
        # continuation. transitions() is the chronological export API.
        arrays = {key: value[:self._size] for key, value in self._arrays.items()}
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as output:
            np.savez_compressed(output, metadata=np.asarray(json.dumps(metadata)), **arrays)
        temporary.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "ReplayBuffer":
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            if metadata["schema_version"] != 1:
                raise ValueError("unsupported replay schema")
            buffer = cls(metadata["capacity"])
            if "obs" in archive:
                size = len(archive["obs"])
                if size > buffer.capacity:
                    raise ValueError("saved replay exceeds its capacity")
                buffer._allocate(archive["obs"].shape[1], archive["actions"].shape[1])
                for key, array in buffer._arrays.items():
                    saved = archive[key]
                    if saved.shape != (size, *array.shape[1:]):
                        raise ValueError(f"invalid saved {key} shape")
                    array[:size] = saved
                buffer._size, buffer._position = size, metadata["position"]
                if not 0 <= buffer._position < buffer.capacity:
                    raise ValueError("invalid saved write position")
            buffer._episode_id = metadata["episode_id"]
            buffer._episode_return = metadata["episode_return"]
            if metadata["open_episode"]:
                if not buffer._size:
                    raise ValueError("saved open episode is empty")
                buffer._pending = int(buffer._ordered_indices()[-1])
                buffer._episode_slots = np.flatnonzero(
                    buffer._arrays["episode_id"][:buffer._size] == buffer._episode_id).tolist()
            buffer.rng.bit_generator.state = metadata["rng_state"]
            return buffer


def sample_mixed(offline: ReplayBuffer, online: ReplayBuffer, batch_size: int,
                 offline_fraction: float = 0.3) -> dict[str, np.ndarray]:
    """Sample fixed offline/online counts, rounded to the nearest transition.

    At the published batch size 256, the 30% split is 77 offline + 179 online.
    An empty required source is an error: this never silently changes the mix.
    Episode-return weighting is exclusively for BC, not online mixed replay.
    """
    if batch_size <= 0 or not 0 <= offline_fraction <= 1:
        raise ValueError("invalid batch size or offline fraction")
    offline_count = int(np.floor(batch_size * offline_fraction + 0.5))
    online_count = batch_size - offline_count
    batches = []
    if offline_count:
        batches.append(offline.sample(offline_count))
    if online_count:
        batches.append(online.sample(online_count))
    merged = {key: np.concatenate([batch[key] for batch in batches]) for key in _BATCH_KEYS}
    order = online.rng.permutation(batch_size)
    return {key: value[order] for key, value in merged.items()}
