"""Independent primitive-informed CEM implementation of arXiv:2609.14868v1.

The published companion settings determine the horizon, knot count, candidate
budget, elites and sampling floors. Linear interpolation, three CEM iterations,
and the timestep/control period defaults are explicit reproduction choices.
The evaluator supplies dynamics, task cost and transition-level constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .primitives import PrimitiveBasis


class NoFeasiblePlan(RuntimeError):
    """No finite-cost trajectory was found; no command may be issued."""


@dataclass(frozen=True)
class CEMConfig:
    horizon_seconds: float = 0.5
    knots: int = 6
    samples: int = 350
    elites: int = 10
    iterations: int = 3
    timestep_seconds: float = 0.01
    control_seconds: float = 0.1

    def __post_init__(self) -> None:
        for name in ("knots", "samples", "elites", "iterations"):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.knots < 2:
            raise ValueError("at least two knots are required")
        if self.samples < 2 or self.samples % 2:
            raise ValueError("samples must be even to split exploration/refinement equally")
        if self.elites > self.samples:
            raise ValueError("elites cannot exceed samples")
        for name in ("horizon_seconds", "timestep_seconds", "control_seconds"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.control_seconds > self.horizon_seconds:
            raise ValueError("control period cannot exceed the planning horizon")
        for duration in (self.horizon_seconds, self.control_seconds):
            ratio = duration / self.timestep_seconds
            if not np.isclose(ratio, round(ratio), atol=1e-9, rtol=1e-9):
                raise ValueError("horizon and control period must be multiples of timestep")

    @property
    def horizon_steps(self) -> int:
        return round(self.horizon_seconds / self.timestep_seconds)


class PrimitiveCEM:
    """CEM over primitive coefficients plus a full 16-joint residual.

    ``evaluate`` receives bounded joint targets of shape (samples, H, 16).
    Target h is held for the transition beginning at h * timestep_seconds.
    It must return one scalar cost per candidate, assigning ``inf`` whenever a
    task constraint fails at any transition. Nonfinite costs never enter elites.
    The returned action is the first command of the lowest-cost feasible sample
    across *all* iterations; ``last_sequence`` exposes its complete trajectory.

    The elite mean and standard deviation are shifted by the control period for
    the next call. At the horizon boundary, the last knot is held constant.
    Call reset after an episode reset or a discontinuous measured-state change.
    """

    def __init__(
        self,
        primitive: PrimitiveBasis,
        action_low: np.ndarray,
        action_high: np.ndarray,
        config: CEMConfig = CEMConfig(),
        seed: int = 0,
    ) -> None:
        self.primitive = primitive
        self.config = config
        self.action_low = np.array(action_low, dtype=np.float64, copy=True)
        self.action_high = np.array(action_high, dtype=np.float64, copy=True)
        if self.action_low.shape != (16,) or self.action_high.shape != (16,):
            raise ValueError("action bounds must have 16 actuator-ordered entries")
        if not np.all(np.isfinite(self.action_low)) or not np.all(np.isfinite(self.action_high)) or np.any(self.action_low >= self.action_high):
            raise ValueError("action bounds must be finite with low < high")
        self.rng = np.random.default_rng(seed)
        self.knot_times = np.linspace(0, config.horizon_seconds, config.knots)
        self.step_times = np.arange(config.horizon_steps) * config.timestep_seconds
        spread = primitive.coefficient_max - primitive.coefficient_min
        self.exploration_floor = np.concatenate([spread, np.full(16, 1.0)])
        self.refinement_floor = np.concatenate([0.3 * spread, np.full(16, 0.1)])
        self.nominal_knots = np.zeros((config.knots, primitive.components + 16))
        self.std_knots = np.zeros_like(self.nominal_knots)
        self.last_sequence: np.ndarray | None = None
        self.reset()

    def reset(self, joint_target: np.ndarray | None = None) -> None:
        """Clear warm-start state, optionally centering it on a measured posture."""
        self.nominal_knots.fill(0)
        self.std_knots.fill(0)
        self.last_sequence = None
        if joint_target is not None:
            target = np.asarray(joint_target, dtype=np.float64)
            if target.shape != (16,) or not np.all(np.isfinite(target)):
                raise ValueError("joint_target must contain 16 finite values")
            target = np.clip(target, self.action_low, self.action_high)
            coefficients = self.primitive.project(target)
            self.nominal_knots[:, :self.primitive.components] = coefficients
            self.nominal_knots[:, self.primitive.components:] = target - self.primitive.reconstruct(coefficients)

    def _interpolate(self, knots: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Vectorized piecewise linear interpolation on the penultimate axis."""
        indices = np.searchsorted(self.knot_times, times, side="right") - 1
        indices = np.clip(indices, 0, self.config.knots - 2)
        weights = (times - self.knot_times[indices]) / (
            self.knot_times[indices + 1] - self.knot_times[indices]
        )
        weights = np.clip(weights, 0, 1)
        return knots[..., indices, :] * (1 - weights[:, None]) + knots[..., indices + 1, :] * weights[:, None]

    def _decode(self, knots: np.ndarray) -> np.ndarray:
        latent = self._interpolate(knots, self.step_times)
        d = self.primitive.components
        commands = self.primitive.mean + latent[..., :d] @ self.primitive.basis.T + latent[..., d:]
        return np.clip(commands, self.action_low, self.action_high)

    def _sample(self, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        half = self.config.samples // 2
        floors = np.stack([self.exploration_floor, self.refinement_floor])
        scales = np.maximum(std[None, ...], floors[:, None, :])
        noise = self.rng.standard_normal((2, half, *mean.shape))
        return (mean[None, None, ...] + noise * scales[:, None, ...]).reshape(
            self.config.samples, *mean.shape
        )

    def plan(
        self,
        evaluate: Callable[[np.ndarray], np.ndarray],
    ) -> tuple[np.ndarray, dict]:
        mean = self.nominal_knots.copy()
        std = self.std_knots.copy()
        best_cost = float("inf")
        best_sequence = None
        history = []
        for iteration in range(self.config.iterations):
            samples = self._sample(mean, std)
            sequences = self._decode(samples)
            costs = np.asarray(evaluate(sequences), dtype=np.float64)
            if costs.shape != (self.config.samples,):
                raise ValueError("evaluate must return one cost for each candidate")
            valid = np.flatnonzero(np.isfinite(costs))
            count = min(self.config.elites, len(valid))
            if count:
                selected = valid[np.argsort(costs[valid], kind="stable")[:count]]
                candidate = selected[0]
                if costs[candidate] < best_cost:
                    best_cost = float(costs[candidate])
                    best_sequence = sequences[candidate].copy()
                mean = samples[selected].mean(axis=0)
                std = samples[selected].std(axis=0)
            history.append({
                "iteration": iteration,
                "valid_candidates": int(len(valid)),
                "elite_count": int(count),
                "iteration_best_cost": float(costs[valid].min()) if len(valid) else None,
                "best_cost": best_cost if best_sequence is not None else None,
            })
        if best_sequence is None:
            raise NoFeasiblePlan(
                f"No feasible trajectory in {self.config.samples * self.config.iterations} rollouts"
            )
        # Commit only after a feasible command exists. This preserves warm-start
        # state if evaluation raises or every candidate is rejected.
        shifted_times = self.knot_times + self.config.control_seconds
        self.nominal_knots = self._interpolate(mean, shifted_times)
        self.std_knots = self._interpolate(std, shifted_times)
        self.last_sequence = best_sequence
        diagnostics = {
            "best_cost": best_cost,
            "iterations": history,
            "rollouts": self.config.samples * self.config.iterations,
            "horizon_steps": self.config.horizon_steps,
            "timestep_seconds": self.config.timestep_seconds,
            "interpolation": "linear",
            "valid_candidates": sum(item["valid_candidates"] for item in history),
        }
        return best_sequence[0].copy(), diagnostics
