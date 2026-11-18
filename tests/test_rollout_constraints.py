"""Physical rollout rejection must agree with the environment's contact rules."""
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mpc_dex.env import AllegroRotationEnv, EnvConfig
from mpc_dex.rollouts import MujocoRolloutEvaluator, RolloutCostConfig


GRASP = str(Path(__file__).resolve().parents[1] / 'assets/initial_grasp.npz')


@pytest.fixture
def env():
    environment = AllegroRotationEnv(EnvConfig(initial_grasp=GRASP))
    environment.reset(seed=0)
    yield environment
    environment.close()


def _integration_state(environment, data=None):
    state = np.empty(mujoco.mj_stateSize(environment.model, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(environment.model, environment.data if data is None else data,
                       state, mujoco.mjtState.mjSTATE_INTEGRATION)
    return state


def _airborne(environment):
    # This object has left the fingers but is still inside the workspace and
    # well above the table. A geometric table-height check cannot detect loss.
    environment.data.qpos[environment.object_qadr] += .1
    environment.data.qvel[environment.object_dadr + 4] = 100.
    mujoco.mj_forward(environment.model, environment.data)
    assert environment.finger_contacts() == 0
    assert not environment.dropped()


@pytest.mark.parametrize('workers', [1, 2])
def test_airborne_spin_rejected_and_stops_at_contact_loss_deadline(env, monkeypatch, workers):
    _airborne(env)
    evaluator = MujocoRolloutEvaluator(env, .01, RolloutCostConfig(workers=workers))
    try:
        before = _integration_state(env)
        target = env.data.ctrl.copy()
        # Count real physics calls as well as checking evaluator diagnostics:
        # stopped candidates must not execute their remaining 450 ms.
        calls = []
        original_step = mujoco.mj_step

        def counted_step(model, data):
            calls.append(data)
            original_step(model, data)

        monkeypatch.setattr(mujoco, 'mj_step', counted_step)
        costs = evaluator(np.broadcast_to(target, (3, 50, 16)).copy())
        assert np.isinf(costs).all()
        assert len(calls) == 3 * 25
        assert evaluator.last_diagnostics['candidate_physics_steps'] == [25] * 3
        assert evaluator.last_diagnostics['rejection_reasons'] == ['contact_loss'] * 3
        assert evaluator.last_diagnostics['early_termination_implemented']
        np.testing.assert_array_equal(_integration_state(env), before)
        assert env._contactless_seconds == 0
        for data in evaluator.worker_data[:workers]:
            assert data.xpos[env.object_id, 2] > .1
            assert not env.dropped(data)

        # Independent execution of the actual environment reaches the same
        # contact-loss failure before the object approaches the table.
        _, reward, done, _, info = env.step(env.to_action(target))
        assert done and info['drop'] and not info['success'] and reward == -50
        assert env.data.xpos[env.object_id, 2] > .1
    finally:
        evaluator.close()


def test_contact_loss_timer_is_inherited_from_current_environment_state(env):
    _airborne(env)
    env._contactless_seconds = .048
    evaluator = MujocoRolloutEvaluator(env, .01, RolloutCostConfig(workers=1))
    try:
        costs = evaluator(np.broadcast_to(env.data.ctrl, (1, 10, 16)).copy())
        assert np.isinf(costs[0])
        assert evaluator.last_diagnostics['candidate_physics_steps'] == [1]
        assert evaluator.last_diagnostics['rejection_reasons'] == ['contact_loss']
        assert env._contactless_seconds == .048
    finally:
        evaluator.close()


def test_stable_grasp_resets_contact_timer_and_exactly_matches_execution(env):
    env._contactless_seconds = .048
    # Applied force and warmstart are inputs too; syncing qpos/qvel alone is
    # insufficient to reproduce the live environment's dynamics.
    env.data.qfrc_applied[env.object_dadr] = .003
    action = env.to_action(env.data.ctrl)
    target = env.to_joint_targets(action)
    before = _integration_state(env)
    evaluator = MujocoRolloutEvaluator(env, .1, RolloutCostConfig(workers=1))
    try:
        costs = evaluator(np.broadcast_to(target, (2, 3, 16)).copy())
        assert np.isfinite(costs).all()
        np.testing.assert_array_equal(costs, np.repeat(costs[0], 2))
        assert evaluator.last_diagnostics['candidate_physics_steps'] == [150, 150]
        assert evaluator.last_diagnostics['rejection_reasons'] == [None, None]
        np.testing.assert_array_equal(_integration_state(env), before)
        assert env._contactless_seconds == .048

        predicted = _integration_state(env, evaluator.worker_data[0])
        for _ in range(3):
            _, _, terminated, truncated, info = env.step(action)
            assert not terminated and not truncated and not info['drop']
        np.testing.assert_array_equal(_integration_state(env), predicted)
        assert env._contactless_seconds == 0
    finally:
        evaluator.close()


def test_table_collision_rejects_candidate_on_first_transition(env):
    env.data.qpos[env.object_qadr + 2] = .01
    mujoco.mj_forward(env.model, env.data)
    evaluator = MujocoRolloutEvaluator(env, .01, RolloutCostConfig(workers=1))
    try:
        costs = evaluator(np.broadcast_to(env.data.ctrl, (1, 10, 16)).copy())
        assert np.isinf(costs[0])
        assert evaluator.last_diagnostics['candidate_physics_steps'] == [1]
        assert evaluator.last_diagnostics['rejection_reasons'] == ['table_contact']
    finally:
        evaluator.close()


def test_workspace_escape_rejects_before_contact_loss_deadline(env):
    env.data.qpos[env.object_qadr] += .3
    mujoco.mj_forward(env.model, env.data)
    evaluator = MujocoRolloutEvaluator(env, .01, RolloutCostConfig(workers=1))
    try:
        costs = evaluator(np.broadcast_to(env.data.ctrl, (1, 10, 16)).copy())
        assert np.isinf(costs[0])
        assert evaluator.last_diagnostics['candidate_physics_steps'] == [1]
        assert evaluator.last_diagnostics['rejection_reasons'] == ['workspace']
    finally:
        evaluator.close()


@pytest.mark.parametrize('deviation', [0., .25])
def test_rotation_tilt_cost_uses_episode_initial_axis_and_matches_environment(deviation):
    env = AllegroRotationEnv(EnvConfig(initial_grasp=GRASP, control_seconds=.002))
    evaluator = MujocoRolloutEvaluator(env, .002, RolloutCostConfig(
        position_weight=0., principal_weight=0., perpendicular_weight=7., workers=1))
    try:
        env.reset(seed=0)
        # An episode may start with the rod along palm z, rather than palm y.
        # Place it briefly clear of contacts to isolate the orientation cost;
        # the single 2 ms transition is within the contact-loss grace period.
        env.initial_orientation = np.array([np.sqrt(.5), np.sqrt(.5), 0., 0.])
        relative = np.array([np.cos((np.pi/2+deviation)/2),
                             np.sin((np.pi/2+deviation)/2), 0., 0.])
        quaternion = np.empty(4)
        mujoco.mju_mulQuat(quaternion, env.data.xquat[env.palm_id], relative)
        env.data.qpos[env.object_qadr] += .25
        env.data.qpos[env.object_qadr+3:env.object_qadr+7] = quaternion
        env.data.qvel[:] = 0.
        mujoco.mj_forward(env.model, env.data)
        env.initial_position = env.object_position().copy()
        env._last_object_quaternion = env.object_quaternion().copy()
        assert env.finger_contacts() == 0
        assert env.axis_tilt() == pytest.approx(deviation, abs=3e-8)
        action = env.to_action(env.data.ctrl)
        target = env.to_joint_targets(action)
        cost = evaluator(target[None, None, :])[0]
        _, _, terminated, truncated, info = env.step(action)
        assert not terminated and not truncated
        assert info['axis_tilt_radians'] == pytest.approx(deviation, abs=3e-8)
        assert cost == pytest.approx(7. * info['axis_tilt_radians']**2 * .002, abs=1e-12)
    finally:
        evaluator.close()
        env.close()


@pytest.mark.parametrize('condition', ['held', 'one_finger', 'no_contact', 'tilted'])
@pytest.mark.parametrize('direction', [1., -1.])
def test_retained_progress_cost_matches_executed_environment(condition, direction):
    env = AllegroRotationEnv(EnvConfig(initial_grasp=GRASP, control_seconds=.002))
    evaluators = {}
    try:
        env.reset(seed=0)
        if condition == 'one_finger':
            env.data.qpos[env.object_qadr+1] += .08
        elif condition == 'no_contact':
            env.data.qpos[env.object_qadr] += .25
        elif condition == 'tilted':
            # The current grasp is physically stable but its rod axis is now
            # far from the episode's reference axis. This isolates tilt gating
            # while actual multi-finger contacts and dynamics remain intact.
            reference = np.empty(4)
            mujoco.mju_mulQuat(reference, env.initial_orientation,
                              np.array([np.sqrt(.5), np.sqrt(.5), 0., 0.]))
            env.initial_orientation = reference
        env.data.qvel[:] = 0.
        env.data.qvel[env.object_dadr+4] = direction * 2.
        mujoco.mj_forward(env.model, env.data)
        env.initial_position = env.object_position().copy()
        env._last_object_quaternion = env.object_quaternion().copy()
        # Scoring remains relative to the planning horizon, not an implicit
        # change to the episode's remaining-angle target.
        env.angle = 1.4
        env.retained_angle = .8
        if condition == 'one_finger':
            assert env.finger_contacts() == 1
            assert sum(env.object_geom in c.geom for c in env.data.contact) > 1
        elif condition == 'no_contact':
            assert env.finger_contacts() == 0
        else:
            assert env.finger_contacts() >= 2
        action = env.to_action(env.data.ctrl)
        target = env.to_joint_targets(action)
        sequences = np.broadcast_to(target, (1, 10, 16)).copy()
        before = _integration_state(env)
        costs = {}
        for mode in ('default', 'raw', 'retained'):
            settings = dict(position_weight=0., perpendicular_weight=0., workers=1)
            if mode != 'default':
                settings['rotation_progress'] = mode
            evaluator = MujocoRolloutEvaluator(env, .002, RolloutCostConfig(**settings))
            evaluators[mode] = evaluator
            costs[mode] = evaluator(sequences)[0]
            assert np.isfinite(costs[mode])
            np.testing.assert_array_equal(_integration_state(env), before)
            assert env.angle == 1.4 and env.retained_angle == .8
        assert costs['default'] == costs['raw']

        raw, retained, contacts, tilts = [], [], [], []
        for _ in range(10):
            _, _, terminated, truncated, info = env.step(action)
            assert not terminated and not truncated
            raw.append(info['angle']-1.4)
            retained.append(info['retained_angle']-.8)
            contacts.append(info['finger_contacts'])
            tilts.append(info['axis_tilt_radians'])
        raw, retained = np.array(raw), np.array(retained)
        for mode, progress in [('raw', raw), ('retained', retained)]:
            expected = np.sum((progress-2*np.pi)**2) * env.config.timestep
            assert costs[mode] == pytest.approx(expected, abs=1e-12)
            diagnostics = evaluators[mode].last_diagnostics
            assert diagnostics['rotation_progress'] == mode
            assert diagnostics['raw_final_progress_range'] == pytest.approx([raw[-1]]*2, abs=1e-12)
            assert diagnostics['retained_final_progress_range'] == pytest.approx([retained[-1]]*2, abs=1e-12)

        if condition == 'held':
            assert min(contacts) >= 2 and max(tilts) < np.radians(30)
            np.testing.assert_allclose(retained, raw, atol=1e-12)
        else:
            if condition == 'tilted':
                assert min(contacts) >= 2 and min(tilts) > np.radians(30)
            else:
                assert max(contacts) <= 1
            assert np.all(retained <= 1e-12)
            if direction > 0:
                assert raw.max() > 0
                assert costs['retained'] > costs['raw']
            else:
                assert retained.min() < 0  # Reverse motion remains penalized.
        if condition == 'no_contact' and direction < 0:
            np.testing.assert_allclose(retained, raw, atol=1e-12)
    finally:
        for evaluator in evaluators.values():
            evaluator.close()
        env.close()


def test_rotation_progress_mode_is_explicit_and_defaults_to_raw():
    assert RolloutCostConfig().rotation_progress == 'raw'
    with pytest.raises(ValueError, match='rotation_progress'):
        RolloutCostConfig(rotation_progress='unknown')


@pytest.mark.parametrize('sticky', [False, True])
def test_palm_support_stops_physics_immediately_and_preserves_episode_history(env, monkeypatch, sticky):
    if sticky:
        assert env.palm_support_force() == 0.
        env._palm_support_violation = True
    else:
        env.data.qpos[env.object_qadr:env.object_qadr+3] = (
            env.data.xpos[env.palm_id] + env.data.xmat[env.palm_id].reshape(3, 3) @
            np.array([.015, 0., .005]))
        env.data.qvel[:] = 0.
        mujoco.mj_forward(env.model, env.data)
        assert env.palm_support_force() > env.config.palm_support_force_tolerance
    before = _integration_state(env)
    evaluator = MujocoRolloutEvaluator(env, .01, RolloutCostConfig(workers=2))
    calls = []
    original = mujoco.mj_step
    def counted_step(model, data):
        calls.append(data)
        original(model, data)
    monkeypatch.setattr(mujoco, 'mj_step', counted_step)
    try:
        assert np.isinf(evaluator(np.broadcast_to(env.data.ctrl, (3, 50, 16)).copy())).all()
        assert len(calls) == (0 if sticky else 3)
        assert evaluator.last_diagnostics['candidate_physics_steps'] == [0 if sticky else 1]*3
        assert evaluator.last_diagnostics['rejection_reasons'] == ['palm_support']*3
        assert env._palm_support_violation == sticky
        np.testing.assert_array_equal(_integration_state(env), before)
    finally:
        evaluator.close()


def test_retained_progress_option_leaves_reorientation_cost_unchanged():
    env = AllegroRotationEnv(EnvConfig(initial_grasp=GRASP, task='reorientation'))
    env.reset(seed=0)
    evaluators = [MujocoRolloutEvaluator(env, .01, RolloutCostConfig(
        workers=1, rotation_progress=mode)) for mode in ('raw', 'retained')]
    try:
        sequences = np.broadcast_to(env.data.ctrl, (2, 5, 16)).copy()
        raw = evaluators[0](sequences)
        retained = evaluators[1](sequences)
        assert np.isfinite(raw).all()
        np.testing.assert_array_equal(retained, raw)
    finally:
        for evaluator in evaluators:
            evaluator.close()
        env.close()
