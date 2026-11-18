"""Global-step chunk scheduler for MPC-scaffolded policy training.

One controller supplies each complete action. A controller is sampled only at
global chunk boundaries; resetting an episode does not reset the curriculum or
create a new chunk. This implements the paper's default .5 to 1 probability
ramp over 6000 environment transitions with 40-transition chunks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any, Literal

import numpy as np


Controller = Literal["policy", "mpc"]


@dataclass(frozen=True)
class SchedulerConfig:
    initial_policy_probability: float = 0.5
    ramp_steps: int = 6000
    chunk_steps: int = 40

    def __post_init__(self) -> None:
        if not np.isfinite(self.initial_policy_probability) or not 0 <= self.initial_policy_probability <= 1:
            raise ValueError("initial_policy_probability must be in [0, 1]")
        for name in ("ramp_steps", "chunk_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        object.__setattr__(self, "initial_policy_probability", float(self.initial_policy_probability))
        if self.ramp_steps % self.chunk_steps:
            raise ValueError("ramp_steps must end on a global chunk boundary")


class ChunkScheduler:
    """Reproducible per-chunk controller selection using a global step count.

    Step 0 means no training environment transitions have completed. The caller
    supplies the same global count to ``select`` throughout training, including
    across episode resets. Repeating a step is harmless, decreasing it is an
    error. Skipped chunks still consume their scheduled draws so the result at
    a given global step is independent of how often the caller queried it.
    """

    def __init__(self, config: SchedulerConfig = SchedulerConfig(), seed: int = 0):
        self.config = config
        self._rng = np.random.default_rng(seed)
        self._last_step = -1
        self._chunk_index = -1
        self._controller: Controller | None = None

    @staticmethod
    def _validate_step(environment_step: int) -> int:
        if isinstance(environment_step, bool) or not isinstance(environment_step, (int, np.integer)) or environment_step < 0:
            raise ValueError("environment_step must be a nonnegative integer")
        return int(environment_step)

    def policy_probability(self, environment_step: int) -> float:
        """Curriculum probability; actual sampling uses the chunk start step."""
        step = self._validate_step(environment_step)
        fraction = min(step / self.config.ramp_steps, 1.0)
        initial = self.config.initial_policy_probability
        return initial + (1.0 - initial) * fraction

    @property
    def last_step(self) -> int:
        return self._last_step

    @property
    def chunk_start_step(self) -> int | None:
        return None if self._chunk_index < 0 else self._chunk_index * self.config.chunk_steps

    def select(self, environment_step: int) -> Controller:
        step = self._validate_step(environment_step)
        if step < self._last_step:
            raise ValueError("environment_step cannot move backward; preserve the global training counter")
        chunk = step // self.config.chunk_steps
        last_stochastic_chunk = self.config.ramp_steps // self.config.chunk_steps - 1
        for index in range(self._chunk_index + 1, min(chunk, last_stochastic_chunk) + 1):
            probability = self.policy_probability(index * self.config.chunk_steps)
            self._controller = "policy" if self._rng.random() < probability else "mpc"
        if step >= self.config.ramp_steps:
            self._controller = "policy"
        self._chunk_index = chunk
        self._last_step = step
        assert self._controller is not None
        return self._controller

    def reset_episode(self) -> None:
        """An episode boundary intentionally leaves the global chunk untouched."""

    def state_dict(self) -> dict[str, Any]:
        """Return a JSON-safe snapshot including the generator state."""
        state = {
            "schema_version": 1,
            "config": asdict(self.config),
            "last_step": self._last_step,
            "chunk_index": self._chunk_index,
            "controller": self._controller,
            "rng_state": self._rng.bit_generator.state,
        }
        return json.loads(json.dumps(state, allow_nan=False))

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a snapshot; incompatible configuration is an explicit error."""
        state = json.loads(json.dumps(state, allow_nan=False))
        if state.get("schema_version") != 1:
            raise ValueError("unsupported scheduler state schema")
        if state.get("config") != asdict(self.config):
            raise ValueError("scheduler configuration differs from checkpoint")
        step = state.get("last_step")
        chunk = state.get("chunk_index")
        controller = state.get("controller")
        if type(step) is not int or type(chunk) is not int:
            raise ValueError("invalid scheduler step or chunk state")
        if step == -1:
            if chunk != -1 or controller is not None:
                raise ValueError("invalid uninitialized scheduler state")
        elif step < 0 or chunk != step // self.config.chunk_steps or controller not in {"policy", "mpc"}:
            raise ValueError("inconsistent scheduler step, chunk or controller")
        if step >= self.config.ramp_steps and controller != "policy":
            raise ValueError("scheduler must select policy after the ramp")
        # Validate generator state on a new generator before committing anything.
        rng = np.random.default_rng()
        rng.bit_generator.state = state["rng_state"]
        self._rng = rng
        self._last_step = step
        self._chunk_index = chunk
        self._controller = controller
