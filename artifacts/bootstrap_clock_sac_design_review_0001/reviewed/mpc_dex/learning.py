"""Independent implementation of the learning stages in arXiv:2609.14878v1.

This is not the authors' code. Unpublished choices are explicit: LayerNorm is
between each hidden linear layer and tanh, the online actor uses the ensemble
mean, and the actor learning-rate warmup starts at zero. Soft-SARSA uses all
target critics, as the paper states. Online backups use a configurable random
subset (two by default, following the project implementation table).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import copy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LearnerConfig:
    hidden_sizes: tuple[int, ...] = (256, 256)
    ensemble_size: int = 10
    online_target_critics: int = 2
    actor_critic_reduction: str = "mean"
    min_std: float = 1e-5
    max_std: float = 5.0
    bc_lr: float = 3e-3
    bc_action_epsilon: float = 0.02
    initial_std: float = 0.3
    critic_lr: float = 3e-5
    actor_lr: float = 3e-4
    actor_warmup_updates: int = 200_000
    alpha_lr: float = 1e-4
    initial_alpha: float = 0.05
    pretrain_alpha: float = 0.02
    discount: float = 0.995
    tau: float = 0.01
    target_entropy: float = -16.0

    def __post_init__(self):
        if not self.hidden_sizes or any(n <= 0 for n in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive widths")
        if not 1 <= self.online_target_critics <= self.ensemble_size:
            raise ValueError("target critic count must be within ensemble size")
        if self.actor_critic_reduction not in ("mean", "min"):
            raise ValueError("actor_critic_reduction must be mean or min")
        if not 0 < self.min_std <= self.initial_std <= self.max_std:
            raise ValueError("invalid standard deviation range")
        if not 0 < self.bc_action_epsilon < 1:
            raise ValueError("BC action clipping epsilon must be in (0, 1)")
        if not 0 <= self.discount < 1 or not 0 < self.tau <= 1:
            raise ValueError("invalid discount or Polyak coefficient")
        if self.actor_warmup_updates < 0 or self.initial_alpha <= 0:
            raise ValueError("invalid warmup or initial temperature")
        if min(self.bc_lr, self.critic_lr, self.actor_lr, self.alpha_lr) <= 0:
            raise ValueError("learning rates must be positive")


def _trunk(input_dim: int, hidden_sizes: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for width in hidden_sizes:
        layers.extend((nn.Linear(input_dim, width), nn.LayerNorm(width), nn.Tanh()))
        input_dim = width
    return nn.Sequential(*layers)


class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, config: LearnerConfig):
        super().__init__()
        self.trunk = _trunk(obs_dim, config.hidden_sizes)
        self.mean_head = nn.Linear(config.hidden_sizes[-1], action_dim)
        self.log_std_head = nn.Linear(config.hidden_sizes[-1], action_dim)
        self.log_std_bounds = (math.log(config.min_std), math.log(config.max_std))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(obs)
        return self.mean_head(features), self.log_std_head(features).clamp(*self.log_std_bounds)

    @staticmethod
    def _log_prob(mean: torch.Tensor, log_std: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        normal = torch.distributions.Normal(mean, log_std.exp())
        # Exact tanh Jacobian written in a stable form even for large raw values.
        log_jacobian = 2 * (math.log(2.0) - raw - F.softplus(-2 * raw))
        return (normal.log_prob(raw) - log_jacobian).sum(dim=-1)

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs)
        raw = torch.distributions.Normal(mean, log_std.exp()).rsample()
        return raw.tanh(), self._log_prob(mean, log_std, raw)

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        mean, log_std = self(obs)
        # A recorded action can equal +/-1, outside the open support of tanh.
        # This numerical epsilon is distinct from the paper's .02 BC clipping.
        raw = torch.atanh(actions.clamp(-1 + 1e-6, 1 - 1e-6))
        return self._log_prob(mean, log_std, raw)


class CriticEnsemble(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, config: LearnerConfig):
        super().__init__()
        self.networks = nn.ModuleList([
            nn.Sequential(_trunk(obs_dim + action_dim, config.hidden_sizes),
                          nn.Linear(config.hidden_sizes[-1], 1))
            for _ in range(config.ensemble_size)
        ])

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat((obs, actions), dim=-1)
        return torch.stack([network(inputs).squeeze(-1) for network in self.networks])


class SACAgent:
    """BC, frozen-actor Soft-SARSA, then online SAC with runner-controlled UTD.

    Batch arrays have leading batch dimension; rewards/terminated/valid are
    vectors or (batch, 1). ``update`` consumes one mixed replay batch and performs
    one critic, actor, and temperature update. The runner supplies eight such
    updates per environment transition. ``load`` is a classmethod and restores
    all optimizer states and warmup progress, not replay or environment state.
    """

    def __init__(self, obs_dim: int = 60, action_dim: int = 16,
                 config: LearnerConfig | None = None, device: str = "cpu"):
        if obs_dim <= 0 or action_dim <= 0:
            raise ValueError("observation and action dimensions must be positive")
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.config = config or LearnerConfig(target_entropy=-float(action_dim))
        self.device = torch.device(device)
        self.actor = SquashedGaussianActor(obs_dim, action_dim, self.config).to(self.device)
        self.critics = CriticEnsemble(obs_dim, action_dim, self.config).to(self.device)
        self.target_critics = copy.deepcopy(self.critics).requires_grad_(False)
        self.bc_optimizer = torch.optim.Adam(
            list(self.actor.trunk.parameters()) + list(self.actor.mean_head.parameters()),
            lr=self.config.bc_lr,
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=self.config.critic_lr)
        self.log_alpha = torch.tensor(math.log(self.config.initial_alpha), device=self.device,
                                      requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.config.alpha_lr)
        self.bc_updates = self.pretrain_updates = self.online_updates = 0
        self.std_initialized = False

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _batch(self, batch: dict, keys: tuple[str, ...]) -> dict[str, torch.Tensor]:
        values = {key: torch.as_tensor(batch[key], device=self.device, dtype=torch.float32)
                  for key in keys}
        size = values[keys[0]].shape[0]
        if size == 0:
            raise ValueError("empty training batch")
        dims = {"obs": self.obs_dim, "next_obs": self.obs_dim,
                "actions": self.action_dim, "next_actions": self.action_dim}
        for key, value in values.items():
            if key in dims:
                if value.shape != (size, dims[key]):
                    raise ValueError(f"{key} must have shape ({size}, {dims[key]})")
            else:
                if value.numel() != size:
                    raise ValueError(f"{key} must have one value per transition")
                values[key] = value.reshape(size)
            if not torch.isfinite(value).all():
                raise ValueError(f"{key} contains non-finite values")
            if key in ("actions", "next_actions") and (value.abs() > 1).any():
                raise ValueError(f"{key} exceeds normalized action bounds")
            if key in ("terminated", "next_action_valid") and ((value != 0) & (value != 1)).any():
                raise ValueError(f"{key} must be boolean")
        return values

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_tensor = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if obs_tensor.ndim not in (1, 2) or obs_tensor.shape[-1] != self.obs_dim:
            raise ValueError(f"observation must have final dimension {self.obs_dim}")
        if not torch.isfinite(obs_tensor).all():
            raise ValueError("observation contains non-finite values")
        action = self.actor(obs_tensor)[0].tanh() if deterministic else self.actor.sample(obs_tensor)[0]
        return action.cpu().numpy()

    def bc_step(self, batch: dict) -> dict[str, float]:
        values = self._batch(batch, ("obs", "actions"))
        target = torch.atanh(values["actions"].clamp(
            -1 + self.config.bc_action_epsilon, 1 - self.config.bc_action_epsilon))
        mean, _ = self.actor(values["obs"])
        loss = F.mse_loss(mean, target)
        self.bc_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.bc_optimizer.step()
        self.bc_updates += 1
        return {"bc_loss": float(loss.detach()), "bc_updates": self.bc_updates}

    @torch.no_grad()
    def initialize_std(self) -> None:
        """Call after mean-only BC; make std exactly .3 for every observation."""
        self.actor.log_std_head.weight.zero_()
        self.actor.log_std_head.bias.fill_(math.log(self.config.initial_std))
        self.std_initialized = True

    @torch.no_grad()
    def _polyak(self) -> None:
        for target, source in zip(self.target_critics.parameters(), self.critics.parameters()):
            target.lerp_(source, self.config.tau)

    def _critic_step(self, values: dict, targets: torch.Tensor) -> torch.Tensor:
        estimates = self.critics(values["obs"], values["actions"])
        loss = (estimates - targets.unsqueeze(0)).square().mean()
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.critic_optimizer.step()
        return loss.detach()

    def pretrain_critic_step(self, batch: dict) -> dict[str, float]:
        """Soft-SARSA with actual next demonstration actions and frozen actor.

        A time limit is not terminal. Nonterminal rows lacking a next demo
        action are excluded, including the final truncated demo transition.
        Terminal rows are retained with reward-only targets.
        """
        if not self.std_initialized:
            raise RuntimeError("call initialize_std() after BC before Soft-SARSA")
        values = self._batch(batch, ("obs", "actions", "rewards", "next_obs", "terminated",
                                    "next_actions", "next_action_valid"))
        eligible = values["terminated"].bool() | values["next_action_valid"].bool()
        count = int(eligible.sum())
        if count == 0:
            return {"critic_loss": 0.0, "pretrain_samples": 0, "pretrain_updates": self.pretrain_updates}
        values = {key: value[eligible] for key, value in values.items()}
        with torch.no_grad():
            targets = values["rewards"].clone()
            bootstrap = ~values["terminated"].bool()
            if bootstrap.any():
                next_obs, next_actions = values["next_obs"][bootstrap], values["next_actions"][bootstrap]
                next_q = self.target_critics(next_obs, next_actions).min(dim=0).values
                log_prob = self.actor.log_prob(next_obs, next_actions)
                targets[bootstrap] += self.config.discount * (next_q - self.config.pretrain_alpha * log_prob)
        loss = self._critic_step(values, targets)
        self._polyak()
        self.pretrain_updates += 1
        return {"critic_loss": float(loss), "target_mean": float(targets.mean()),
                "pretrain_samples": count, "pretrain_updates": self.pretrain_updates}

    def update(self, batch: dict) -> dict[str, float]:
        values = self._batch(batch, ("obs", "actions", "rewards", "next_obs", "terminated"))
        with torch.no_grad():
            targets = values["rewards"].clone()
            bootstrap = ~values["terminated"].bool()
            if bootstrap.any():
                next_actions, next_log_prob = self.actor.sample(values["next_obs"][bootstrap])
                selected = torch.randperm(self.config.ensemble_size, device=self.device)[
                    :self.config.online_target_critics]
                next_q = self.target_critics(values["next_obs"][bootstrap], next_actions)[selected].min(dim=0).values
                targets[bootstrap] += self.config.discount * (next_q - self.alpha * next_log_prob)
        critic_loss = self._critic_step(values, targets)
        # Critic parameters are constant for policy differentiation, but their
        # derivatives with respect to the sampled actions remain available.
        self.critics.requires_grad_(False)
        try:
            actions, log_prob = self.actor.sample(values["obs"])
            estimates = self.critics(values["obs"], actions)
            q = estimates.mean(dim=0) if self.config.actor_critic_reduction == "mean" else estimates.min(dim=0).values
            actor_loss = (self.alpha.detach() * log_prob - q).mean()
            warmup = self.config.actor_warmup_updates
            fraction = min((self.online_updates + 1) / warmup, 1.0) if warmup else 1.0
            actor_lr = self.config.actor_lr * fraction
            for group in self.actor_optimizer.param_groups:
                group["lr"] = actor_lr
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
        finally:
            self.critics.requires_grad_(True)
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.config.target_entropy)).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self._polyak()
        self.online_updates += 1
        return {"critic_loss": float(critic_loss), "actor_loss": float(actor_loss.detach()),
                "alpha_loss": float(alpha_loss.detach()), "alpha": float(self.alpha.detach()),
                "actor_lr": actor_lr, "entropy": float(-log_prob.detach().mean()),
                "target_mean": float(targets.mean()), "online_updates": self.online_updates}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": 1, "implementation": "independent-paper-reproduction",
                   "obs_dim": self.obs_dim, "action_dim": self.action_dim, "config": asdict(self.config),
                   "actor": self.actor.state_dict(), "critics": self.critics.state_dict(),
                   "target_critics": self.target_critics.state_dict(), "log_alpha": self.log_alpha.detach(),
                   "bc_updates": self.bc_updates, "pretrain_updates": self.pretrain_updates,
                   "online_updates": self.online_updates, "std_initialized": self.std_initialized,
                   "bc_optimizer": self.bc_optimizer.state_dict(),
                   "actor_optimizer": self.actor_optimizer.state_dict(),
                   "critic_optimizer": self.critic_optimizer.state_dict(),
                   "alpha_optimizer": self.alpha_optimizer.state_dict()}
        temporary = path.with_name(path.name + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "SACAgent":
        payload = torch.load(path, map_location=device, weights_only=True)
        if payload["schema_version"] != 1:
            raise ValueError("unsupported learner checkpoint schema")
        agent = cls(payload["obs_dim"], payload["action_dim"], LearnerConfig(**payload["config"]), device)
        for name in ("actor", "critics", "target_critics"):
            getattr(agent, name).load_state_dict(payload[name])
        with torch.no_grad():
            agent.log_alpha.copy_(payload["log_alpha"])
        for name in ("bc_optimizer", "actor_optimizer", "critic_optimizer", "alpha_optimizer"):
            getattr(agent, name).load_state_dict(payload[name])
        for name in ("bc_updates", "pretrain_updates", "online_updates", "std_initialized"):
            setattr(agent, name, payload[name])
        return agent
