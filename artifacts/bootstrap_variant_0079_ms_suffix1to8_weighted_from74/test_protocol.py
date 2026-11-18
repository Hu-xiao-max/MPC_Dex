"""Synthetic unit fixtures plus two small real-physics checks; no learning or success proof."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('variant_runner_review', HERE / 'run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
torch = runner.torch


def tiny_agent(seed=0):
    torch.manual_seed(seed)
    agent = runner.SACAgent(60, 16, device='cpu')
    agent.initialize_std()
    return agent


def synthetic_batch(n=12, demo_rows=4, seed=1):
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal((n, 60)).astype(np.float32)
    actions = (rng.standard_normal((n, 16)) * .3).clip(-.97, .97).astype(np.float32)
    terminated = np.zeros(n, bool); terminated[-1] = True
    return dict(obs=obs, actions=actions, rewards=rng.standard_normal(n).astype(np.float32), next_obs=obs * .5 + .1,
                terminated=terminated, truncated=np.zeros(n, bool), next_actions=actions[::-1].copy(),
                next_action_valid=~terminated, demonstration=np.arange(n) < demo_rows)


def synthetic_demonstrations():
    rng = np.random.default_rng(7)
    obs = rng.standard_normal((3, 60)).astype(np.float32)
    return [dict(name='synthetic', arrays=dict(obs=obs, actions=(rng.standard_normal((3, 16)) * .5).clip(-.99, .99).astype(np.float32)))]


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


def state(agent):
    return dict(actor=copy.deepcopy(agent.actor.state_dict()), critics=copy.deepcopy(agent.critics.state_dict()),
                targets=copy.deepcopy(agent.target_critics.state_dict()), log_alpha=agent.log_alpha.detach().clone(),
                actor_opt=copy.deepcopy(agent.actor_optimizer.state_dict()), critic_opt=copy.deepcopy(agent.critic_optimizer.state_dict()),
                alpha_opt=copy.deepcopy(agent.alpha_optimizer.state_dict()))


def test_runner_update_is_bit_identical_to_archived_update_without_additions():
    reference, subject = tiny_agent(), tiny_agent()
    batch = synthetic_batch()
    torch.manual_seed(99)
    ref = reference.update({k: v for k, v in batch.items() if k != 'demonstration'})
    torch.manual_seed(99)
    sub = runner.runner_update(subject, batch, dict(actor_on_demonstration_states=True, actor_bc_lambda=0.), synthetic_demonstrations())
    runner.assert_tree_equal(state(reference), state(subject))
    for key in ref:
        assert ref[key] == sub[key], key
    assert sub['actor_bc_loss'] is None and sub['actor_rows_used'] == 12 and sub['entropy'] == sub['entropy_all_rows']


def test_bc_term_rows_option_and_manual_mse():
    demos = synthetic_demonstrations()
    obs = torch.as_tensor(demos[0]['arrays']['obs']); act = torch.as_tensor(demos[0]['arrays']['actions'])
    for rows, expected_rows in (('all', 3), ('first', 1), ('first_k', 2)):
        subject, control = tiny_agent(), tiny_agent()
        with torch.no_grad():
            sl = slice(0, 1) if rows == 'first' else slice(0, 2) if rows == 'first_k' else slice(None)
            expected_bc = float(torch.nn.functional.mse_loss(subject.actor(obs[sl])[0], torch.atanh(act[sl].clamp(-.98, .98))))
        torch.manual_seed(5)
        out = runner.runner_update(subject, synthetic_batch(), dict(actor_on_demonstration_states=True, actor_bc_lambda=10., actor_bc_rows=rows, actor_bc_first_rows=2), demos)
        torch.manual_seed(5)
        runner.runner_update(control, synthetic_batch(), dict(actor_on_demonstration_states=True, actor_bc_lambda=0.), demos)
        assert np.isclose(out['actor_bc_loss'], expected_bc, rtol=1e-5) and out['actor_bc_rows'] == expected_rows
        assert any(not torch.equal(a, b) for a, b in zip(subject.actor.state_dict().values(), control.actor.state_dict().values()))
        for a, b in zip(subject.critics.state_dict().values(), control.critics.state_dict().values()):
            assert torch.equal(a, b)


def test_std_anchor_adds_a_log_std_penalty_only_when_requested():
    with_anchor, without = tiny_agent(), tiny_agent()
    demos = synthetic_demonstrations()
    torch.manual_seed(3)
    out_a = runner.runner_update(with_anchor, synthetic_batch(), dict(actor_on_demonstration_states=True, actor_bc_lambda=10., actor_bc_rows='first', actor_bc_std_target=.02, actor_bc_std_lambda=10.), demos)
    torch.manual_seed(3)
    out_b = runner.runner_update(without, synthetic_batch(), dict(actor_on_demonstration_states=True, actor_bc_lambda=10., actor_bc_rows='first'), demos)
    assert out_a['actor_bc_std_loss'] is not None and out_b['actor_bc_std_loss'] is None
    assert np.isclose(out_a['actor_bc_loss'], out_b['actor_bc_loss'])  # same mean term before the step
    obs = torch.as_tensor(demos[0]['arrays']['obs'][:1])
    with torch.no_grad():
        std_a = with_anchor.actor(obs)[1].exp().mean(); std_b = without.actor(obs)[1].exp().mean()
    assert std_a < std_b  # one anchored step moves the spread toward the target (.02 < initial .3)


def test_alpha_freeze_skips_the_temperature_step_but_keeps_everything_else():
    frozen, live = tiny_agent(), tiny_agent()
    batch = synthetic_batch()
    torch.manual_seed(7)
    out_frozen = runner.runner_update(frozen, batch, dict(actor_on_demonstration_states=True, actor_bc_lambda=0., alpha_freeze=True), synthetic_demonstrations())
    torch.manual_seed(7)
    out_live = runner.runner_update(live, batch, dict(actor_on_demonstration_states=True, actor_bc_lambda=0.), synthetic_demonstrations())
    assert torch.equal(frozen.log_alpha, tiny_agent().log_alpha) and not torch.equal(live.log_alpha, tiny_agent().log_alpha)
    assert frozen.alpha_optimizer.state_dict()['state'] == {} and live.alpha_optimizer.state_dict()['state'] != {}
    for a, b in zip(frozen.actor.state_dict().values(), live.actor.state_dict().values()):
        assert torch.equal(a, b)  # same actor step: alpha is read before the temperature update in both paths
    assert out_frozen['critic_loss'] == out_live['critic_loss'] and np.isclose(out_frozen['alpha_loss'], out_live['alpha_loss'])


def test_actor_mask_restricts_rows_entropy_metric_and_rejects_demo_only_batches():
    agent = tiny_agent()
    out = runner.runner_update(agent, synthetic_batch(n=12, demo_rows=4), dict(actor_on_demonstration_states=False, actor_bc_lambda=0.), synthetic_demonstrations())
    assert out['actor_rows_used'] == 8 and out['entropy'] != out['entropy_all_rows']
    with pytest.raises(AssertionError):
        runner.runner_update(agent, synthetic_batch(n=6, demo_rows=6), dict(actor_on_demonstration_states=False, actor_bc_lambda=0.), synthetic_demonstrations())


def test_mixed_batch_with_origin_reproduces_sample_mixed_and_marks_rows():
    demo, online = small_buffers()
    demo_copy, online_copy = small_buffers()
    reference = runner.sample_mixed(demo_copy, online_copy, 10, .3)
    batch = runner.mixed_batch_with_origin(demo, online, 10, .3)
    for key in reference:
        np.testing.assert_array_equal(batch[key], reference[key])
    assert batch['demonstration'].sum() == 3 and batch['demonstration'].dtype == bool


def test_normalized_batch_uses_phase_fraction_and_scales_only_rewards():
    demo, online = small_buffers()
    protocol = {'batch_size': 10, 'raw_reward_divisor': 200., 'pretrain_offline_fraction': .3, 'online_offline_fraction': .1}
    pre, stats_pre = runner.normalized_batch(demo, online, protocol, 'pretrain')
    on, stats_on = runner.normalized_batch(demo, online, protocol, 'online')
    assert stats_pre['batch_demonstration_rows'] == 3 and stats_on['batch_demonstration_rows'] == 1
    rewards = pre['rewards'].astype(np.float64)
    assert np.all(np.isclose(rewards[pre['demonstration']], 1.2) | np.isclose(rewards[pre['demonstration']], -.4))
    np.testing.assert_array_equal(demo.transitions()['next_action_valid'], [True, False])


def test_entropy_masked_step_matches_archived_step_when_no_demonstration_rows():
    reference, subject = tiny_agent(), tiny_agent()
    batch = synthetic_batch(n=6, demo_rows=0, seed=2)
    ref_result = reference.pretrain_critic_step(dict(batch))
    sub_result = runner.entropy_masked_pretrain_step(subject, dict(batch), {})
    assert np.isclose(ref_result['critic_loss'], sub_result['critic_loss'])
    for a, b in zip(reference.critics.state_dict().values(), subject.critics.state_dict().values()):
        assert torch.equal(a, b)
    for a, b in zip(reference.target_critics.state_dict().values(), subject.target_critics.state_dict().values()):
        assert torch.equal(a, b)


def test_entropy_masked_step_removes_entropy_only_for_demonstration_rows():
    agent = tiny_agent()
    batch = synthetic_batch(n=4, demo_rows=2, seed=3)
    batch['terminated'][:] = [False, False, True, False]
    batch['next_action_valid'][:] = [True, True, False, False]
    batch['demonstration'][:] = [True, False, True, True]
    captured = {}
    original = agent._critic_step
    def capture(values, targets):
        captured['targets'] = targets.clone()
        return original(values, targets)
    agent._critic_step = capture
    with torch.no_grad():
        nq = agent.target_critics(torch.as_tensor(batch['next_obs'][:2]), torch.as_tensor(batch['next_actions'][:2])).min(dim=0).values
        lp = agent.actor.log_prob(torch.as_tensor(batch['next_obs'][:2]), torch.as_tensor(batch['next_actions'][:2]))
    result = runner.entropy_masked_pretrain_step(agent, batch, {})
    r = torch.as_tensor(batch['rewards'])
    assert result['entropy_free_bootstrap_rows'] == 1
    assert torch.isclose(captured['targets'][0], r[0] + agent.config.discount * nq[0], atol=1e-5)
    assert torch.isclose(captured['targets'][1], r[1] + agent.config.discount * (nq[1] - agent.config.pretrain_alpha * lp[1]), atol=1e-5)
    assert torch.isclose(captured['targets'][2], r[2])


def test_demonstration_gate_requires_exact_terminal_episodes():
    demo, _ = small_buffers()
    assert runner.demonstration_gate(demo, dict(demonstration_rows=2, demonstration_episodes=1))
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
    episode.trunc = True
    assert runner.terminal_returns(episode, .9, .05/200, 200., .02/200) is None


def test_prefix_sampling_and_handoff_perturbation_are_seeded_and_bounded():
    protocol = dict(prefix_set=[6, 7, 8], handoff_noise_sigma=.05, handoff_noise_prefixes=[7, 8])
    rng = np.random.default_rng(11)
    many = [runner.sample_prefix(rng, protocol) for _ in range(300)]
    assert set(many) == {6, 7, 8}
    source = dict(actions=np.full((9, 16), .97, np.float32))
    assert runner.handoff_perturbation(6, protocol, np.random.default_rng(0), source) is None
    assert runner.handoff_perturbation(7, dict(protocol, handoff_noise_sigma=0.), np.random.default_rng(0), source) is None
    a = runner.handoff_perturbation(7, protocol, np.random.default_rng(3), source)
    b = runner.handoff_perturbation(7, protocol, np.random.default_rng(3), source)
    np.testing.assert_array_equal(a, b)
    assert a.dtype == np.float32 and np.abs(a).max() <= 1 and not np.array_equal(a, source['actions'][6])
    assert runner.prefix_keys(protocol) == ['6', '7+noise', '8+noise'] and runner.prefix_keys(dict(prefix_set=[6])) == ['6']
    weighted_rng = np.random.default_rng(21)
    weighted = [runner.sample_prefix(weighted_rng, dict(prefix_set=[5, 6], prefix_weights=[.9, .1])) for _ in range(400)]
    assert 300 < weighted.count(5) < 400 and set(weighted) <= {5, 6}


def test_multisource_helpers_slice_suffixes_and_build_tasks():
    rng = np.random.default_rng(3)
    def fake(n):
        obs = rng.standard_normal((n, 60)).astype(np.float32)
        return dict(obs=obs, actions=(rng.standard_normal((n, 16)) * .3).clip(-.9, .9).astype(np.float32), rewards=np.arange(n, dtype=np.float64),
                    next_obs=np.vstack((obs[1:], rng.standard_normal((1, 60)).astype(np.float32))), terminated=np.r_[np.zeros(n - 1, bool), True], truncated=np.zeros(n, bool))
    sources = {'a': dict(source=fake(12), physics=None, length=12), 'b': dict(source=fake(9), physics=None, length=9)}
    protocol = dict(prefix_source_names=['a', 'b'], suffix_set=[1, 2, 3], demonstration_suffix_rows=3, eval_source_names=['a'], eval_deterministic_seed=42,
                    eval_stochastic_seeds_by_suffix={'1': [1], '2': [2], '3': [3]}, handoff_noise_sigma=.015, handoff_noise_suffixes=[2], eval_noisy_seeds_by_suffix={'2': [9]})
    demos = runner.multisource_demonstrations(protocol, sources)
    assert [d['name'] for d in demos] == ['a', 'b'] and all(len(d['arrays']['obs']) == 3 and d['arrays']['terminated'][-1] for d in demos)
    np.testing.assert_array_equal(demos[0]['arrays']['obs'], sources['a']['source']['obs'][9:])
    tasks = runner.multisource_evaluation_tasks(protocol, sources, False)
    assert [t[1] for t in tasks] == [11, 11, 10, 10, 10, 9, 9] and tasks[4][4] and tasks[4][5] == 'a'
    assert runner.multisource_keys(protocol) == ['s1', 's2+noise', 's3']
    wrng = np.random.default_rng(9)
    draws = [runner.sample_suffix(wrng, dict(suffix_set=[1, 2, 3], suffix_weights=[8, 1, 1])) for _ in range(500)]
    assert 350 < draws.count(1) < 450 and set(draws) <= {1, 2, 3}
    assert set(runner.sample_suffix(wrng, dict(suffix_set=[4, 5])) for _ in range(50)) == {4, 5}
    ep = SimpleNamespace(suffix=2, noisy_last=None, prefix=7)
    assert runner.episode_key(ep) == 's2'
    ep2 = SimpleNamespace(suffix=None, noisy_last=None, prefix=7)
    assert runner.episode_key(ep2) == '7'


def test_evaluation_tasks_cover_prefixes_noise_and_cold_schedule():
    protocol = dict(eval_prefixes=[6, 7], eval_deterministic_seed=42, eval_stochastic_seeds_by_prefix={'6': [1, 2], '7': [3, 4]},
                    eval_noisy_prefixes=[7], eval_noisy_seeds_by_prefix={'7': [5]}, handoff_noise_sigma=.05, handoff_noise_prefixes=[7],
                    eval_final_cold_stochastic_seeds=[7, 8, 9])
    tasks = runner.evaluation_tasks(protocol, 'online_500', False)
    assert [t[1] for t in tasks] == [6, 6, 6, 7, 7, 7, 7, 0] and tasks[6][4] and not tasks[5][4]
    assert len(runner.evaluation_tasks(protocol, 'online_2000', True)) == len(tasks) + 3
    assert len(runner.evaluation_tasks(dict(protocol, eval_cold_stochastic_every_point=True), 'online_500', False)) == len(tasks) + 3
    with pytest.raises(AssertionError):
        runner.evaluation_tasks(dict(protocol, eval_prefixes=[0]), 'x', False)


def test_actual_demonstration_compositions_load_consistently():
    base = dict(demonstration_include_mpc=True, demonstration_recorded_rows=[6, 9], demonstration_rows_mpc=5,
                demonstration_rows_recorded=3, demonstration_rows=8, demonstration_episodes=2)
    two = runner.load_demonstrations(base)
    np.testing.assert_array_equal(two[0]['arrays']['obs'][0], two[1]['arrays']['obs'][0])
    full = runner.load_demonstrations(dict(base, demonstration_recorded_rows=[0, 9], demonstration_rows_recorded=9, demonstration_rows=14))
    assert len(full[1]['arrays']['obs']) == 9 and full[1]['arrays']['terminated'][-1]
    only = runner.load_demonstrations(dict(base, demonstration_include_mpc=False, demonstration_rows_mpc=0, demonstration_rows=3, demonstration_episodes=1))
    assert [d['name'] for d in only] == ['recorded']
    demo = runner.ReplayBuffer(64, 0)
    for item in full:
        demo.add_episode(item['arrays'])
    assert runner.demonstration_gate(demo, dict(demonstration_rows=14, demonstration_episodes=2))


def test_actual_episode_perturbed_handoff_matches_exact_replay_when_noise_is_zero():
    # Real physics, frozen inputs, no policy: zero perturbation must reproduce the exact prefix-7 replay bit-for-bit.
    ec = runner.EnvConfig(**json.loads((HERE / 'config.json').read_text())['environment'])
    with np.load(HERE / 'prefix_transitions.npz', allow_pickle=False) as z:
        source = {k: z[k].copy() for k in z.files}
    with np.load(HERE / 'prefix_physics.npz', allow_pickle=False) as z:
        physics = z['samples'].copy()
    env = runner.environment_for_policy(ec, {'kind': 'absolute_joint_target'})
    try:
        exact = runner.Episode(env, 7, source, physics, 42, 'test', 0, 200.)
        exact_handoff = exact.handoff['integration_state'].copy(); exact.close()
        zero = runner.Episode(env, 7, source, physics, 42, 'test', 1, 200., noisy_last=source['actions'][6].copy())
        np.testing.assert_array_equal(zero.handoff['integration_state'], exact_handoff)
        assert zero.handoff_noise_l2 == 0. and zero.report()['perturbed_last_prefix_control'] and zero.report()['exact_prefix_controls'] == 6
        zero.close()
        noisy = runner.Episode(env, 7, source, physics, 42, 'test', 2, 200., noisy_last=np.clip(source['actions'][6] + np.float32(.05), -1, 1))
        assert not np.array_equal(noisy.handoff['integration_state'], exact_handoff) and noisy.handoff_noise_l2 > 0
        assert len(noisy.rows) == 7 and not noisy.rows[-1]['policy_controlled']
        noisy.close()
    finally:
        env.close()


def test_actual_checkpoint_load_save_preserves_entire_payload_without_update(tmp_path):
    original = torch.load(HERE / 'initial.pt', map_location='cpu', weights_only=True)
    agent = runner.SACAgent.load(HERE / 'initial.pt', 'cpu')
    agent.save(tmp_path / 'unchanged.pt')
    runner.assert_tree_equal(original, torch.load(tmp_path / 'unchanged.pt', map_location='cpu', weights_only=True))
    expected = json.loads((HERE / 'protocol.json').read_text())['inherited_optimizer_steps']
    assert runner.optimizer_steps(agent.critic_optimizer) == [expected['critic']]
    assert runner.optimizer_steps(agent.actor_optimizer) == [expected['actor']]
