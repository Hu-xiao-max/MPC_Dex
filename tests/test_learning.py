"""Mathematical contracts for the independent paper learner, not task success."""

from dataclasses import replace
import math

import numpy as np
import pytest
import torch

from mpc_dex.learning import LearnerConfig, SACAgent, SquashedGaussianActor


@pytest.fixture(autouse=True)
def small_torch_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(12)
    yield
    torch.set_num_threads(previous)


def agent(**kwargs):
    config = replace(LearnerConfig(hidden_sizes=(16, 16), target_entropy=-2), **kwargs)
    return SACAgent(3, 2, config)


def batch(size=8):
    rng = np.random.default_rng(3)
    return {"obs": rng.normal(size=(size, 3)).astype(np.float32),
            "actions": rng.uniform(-1, 1, size=(size, 2)).astype(np.float32),
            "rewards": rng.normal(size=size).astype(np.float32),
            "next_obs": rng.normal(size=(size, 3)).astype(np.float32),
            "terminated": np.zeros(size, dtype=bool),
            "next_actions": rng.uniform(-.8, .8, size=(size, 2)).astype(np.float32),
            "next_action_valid": np.ones(size, dtype=bool)}


def constant_critics(ensemble, constants):
    with torch.no_grad():
        for network, constant in zip(ensemble.networks, constants):
            for parameter in network.parameters():
                parameter.zero_()
            network[-1].bias.fill_(constant)


def test_published_network_dimensions_and_defaults():
    learner = SACAgent()
    assert len(learner.critics.networks) == 10
    assert learner.actor.trunk[0].in_features == 60
    assert learner.actor.mean_head.out_features == learner.actor.log_std_head.out_features == 16
    assert learner.actor.trunk[0].out_features == learner.actor.trunk[3].out_features == 256
    assert learner.critics.networks[0][0][0].in_features == 76
    assert learner.config.target_entropy == -16
    assert learner.config.actor_warmup_updates == 200_000


def test_tanh_log_density_matches_change_of_variables_and_sampling_gradient():
    actor = SquashedGaussianActor(3, 2, LearnerConfig(hidden_sizes=(8, 8)))
    observations = torch.zeros(4, 3)
    actions = torch.tensor([[.2, -.6]]).expand(4, -1)
    mean, log_std = actor(observations)
    raw = torch.atanh(actions)
    expected = (torch.distributions.Normal(mean, log_std.exp()).log_prob(raw)
                - torch.log1p(-actions.square())).sum(-1)
    torch.testing.assert_close(actor.log_prob(observations, actions), expected)
    sampled, log_prob = actor.sample(observations)
    assert torch.isfinite(log_prob).all() and (sampled.abs() <= 1).all()
    (sampled.sum() + log_prob.sum()).backward()
    assert actor.mean_head.weight.grad is not None
    assert actor.log_std_head.weight.grad is not None
    assert torch.isfinite(actor.log_prob(observations, torch.ones_like(actions))).all()


def test_bc_pre_tanh_clipping_changes_mean_and_trunk_but_not_std():
    learner = agent()
    values = batch()
    values["actions"][:] = [1., -1.]
    before = {name: value.detach().clone() for name, value in learner.actor.named_parameters()}
    with torch.no_grad():
        mean = learner.actor(torch.as_tensor(values["obs"]))[0]
        target = torch.atanh(torch.as_tensor(values["actions"]).clamp(-.98, .98))
        expected = (mean - target).square().mean().item()
    metrics = learner.bc_step(values)
    assert metrics["bc_loss"] == pytest.approx(expected)
    after = dict(learner.actor.named_parameters())
    assert not torch.equal(before["mean_head.weight"], after["mean_head.weight"])
    assert not torch.equal(before["trunk.0.weight"], after["trunk.0.weight"])
    torch.testing.assert_close(before["log_std_head.weight"], after["log_std_head.weight"])
    torch.testing.assert_close(before["log_std_head.bias"], after["log_std_head.bias"])
    assert after["log_std_head.weight"].grad is None
    learner.initialize_std()
    _, log_std = learner.actor(torch.randn(16, 3))
    torch.testing.assert_close(log_std.exp(), torch.full((16, 2), .3))


def test_soft_sarsa_uses_recorded_actions_all_ten_min_and_frozen_actor(monkeypatch):
    learner = agent()
    learner.initialize_std()
    # The unique minimum is the tenth critic; a two-critic backup is wrong.
    constants = [5.] * 9 + [-3.]
    constant_critics(learner.target_critics, constants)
    values = batch(3)
    values["terminated"] = np.array([False, True, False])
    values["next_action_valid"] = np.array([True, False, False])
    # Third row is a time limit without a recorded next action: drop from SARSA.
    original = {name: value.detach().clone() for name, value in learner.actor.named_parameters()}
    with torch.no_grad():
        lp = learner.actor.log_prob(torch.as_tensor(values["next_obs"][:1]),
                                   torch.as_tensor(values["next_actions"][:1])).item()
    expected = np.array([values["rewards"][0] + .995 * (-3 - .02 * lp), values["rewards"][1]])
    captured = {}
    real_step = learner._critic_step

    def capture(selected, targets):
        captured["actions"] = selected["next_actions"].numpy().copy()
        captured["targets"] = targets.numpy().copy()
        return real_step(selected, targets)

    monkeypatch.setattr(learner, "_critic_step", capture)
    metrics = learner.pretrain_critic_step(values)
    assert metrics["pretrain_samples"] == 2
    np.testing.assert_allclose(captured["targets"], expected, rtol=1e-6)
    np.testing.assert_array_equal(captured["actions"], values["next_actions"][:2])
    for name, parameter in learner.actor.named_parameters():
        torch.testing.assert_close(parameter, original[name])
        assert parameter.grad is None


def test_sarsa_requires_initialized_std_and_skips_no_valid_targets():
    learner = agent()
    values = batch()
    with pytest.raises(RuntimeError, match="initialize_std"):
        learner.pretrain_critic_step(values)
    learner.initialize_std()
    values["next_action_valid"][:] = False
    metrics = learner.pretrain_critic_step(values)
    assert metrics["pretrain_samples"] == metrics["pretrain_updates"] == 0


def test_sac_terminal_mask_preserves_time_limit_bootstrap_and_warmup(monkeypatch):
    learner = agent(online_target_critics=10, actor_warmup_updates=4)
    learner.initialize_std()
    constant_critics(learner.target_critics, [2.] * 10)
    values = batch(2)
    values["terminated"] = np.array([True, False])
    values["truncated"] = np.array([False, True])
    captures = {}
    real_sample, real_step = learner.actor.sample, learner._critic_step

    def sample(observations):
        actions, log_prob = real_sample(observations)
        if observations.shape[0] == 1:
            captures["backup_log_prob"] = log_prob.detach().numpy()
        return actions, log_prob

    def capture(selected, targets):
        captures["targets"] = targets.numpy().copy()
        return real_step(selected, targets)

    monkeypatch.setattr(learner.actor, "sample", sample)
    monkeypatch.setattr(learner, "_critic_step", capture)
    original_actor = learner.actor.mean_head.weight.detach().clone()
    metrics = learner.update(values)
    assert captures["targets"][0] == pytest.approx(values["rewards"][0])
    expected = values["rewards"][1] + .995 * (2 - .05 * captures["backup_log_prob"][0])
    assert captures["targets"][1] == pytest.approx(expected)
    assert metrics["actor_lr"] == pytest.approx(3e-4 / 4)
    assert not torch.equal(learner.actor.mean_head.weight, original_actor)
    assert metrics["alpha"] != pytest.approx(.05, abs=1e-8)
    assert all(parameter.requires_grad for parameter in learner.critics.parameters())
    assert all(not parameter.requires_grad for parameter in learner.target_critics.parameters())
    for _ in range(4):
        metrics = learner.update(values)
    assert metrics["actor_lr"] == pytest.approx(3e-4)


def test_checkpoint_restores_actor_optimizers_and_warmup(tmp_path):
    learner = agent(actor_warmup_updates=5)
    values = batch()
    learner.bc_step(values)
    learner.initialize_std()
    learner.pretrain_critic_step(values)
    learner.update(values)
    path = tmp_path / "agent.pt"
    learner.save(path)
    restored = SACAgent.load(path)
    np.testing.assert_array_equal(learner.act(values["obs"], True), restored.act(values["obs"], True))
    assert restored.std_initialized and restored.online_updates == restored.pretrain_updates == restored.bc_updates == 1
    torch.manual_seed(88)
    original_metrics = learner.update(values)
    torch.manual_seed(88)
    restored_metrics = restored.update(values)
    assert original_metrics == pytest.approx(restored_metrics)
    np.testing.assert_array_equal(learner.act(values["obs"], True), restored.act(values["obs"], True))
    assert learner.act(values["obs"][0]).shape == (2,)
    with pytest.raises(ValueError, match="dimension"):
        learner.act(np.zeros(4))
