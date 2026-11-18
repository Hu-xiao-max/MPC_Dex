"""Force-bearing palm contact invalidates rotation without inventing a drop."""
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mpc_dex.bootstrap_env import BootstrapControlConfig, BootstrapRotationEnv
from mpc_dex.env import AllegroRotationEnv, EnvConfig
from scripts.train_bootstrap import restore_environment, save_environment


GRASP = str(Path(__file__).resolve().parents[1] / 'assets/initial_grasp.npz')


def place_near_palm(env, center=(.033, 0., -.01)):
    env.data.qpos[env.object_qadr:env.object_qadr+3] = (
        env.data.xpos[env.palm_id] + env.data.xmat[env.palm_id].reshape(3, 3) @ center)
    env.data.qvel[:] = 0.
    mujoco.mj_forward(env.model, env.data)


@pytest.mark.parametrize('incremental', [False, True])
def test_actual_palm_support_terminates_as_distinct_task_failure(incremental):
    config = EnvConfig(initial_grasp=GRASP, control_seconds=.002)
    env = (BootstrapRotationEnv(config, BootstrapControlConfig()) if incremental
           else AllegroRotationEnv(config))
    try:
        env.reset(seed=0)
        place_near_palm(env)
        assert env.palm_support_force() > .1
        assert not env.dropped() and env.finger_contacts() >= 2
        action = np.zeros(16) if incremental else env.to_action(env.data.ctrl)
        _, reward, terminated, truncated, info = env.step(action)
        assert terminated and not truncated
        assert not info['drop'] and not info['success']
        assert info['palm_support_violation'] and info['failure_reason'] == 'palm_support'
        assert info['peak_palm_support_force'] >= info['palm_support_force'] > .1
        assert info['retained_positive_step'] == 0.
        assert reward == (-20. if incremental else -50.)
        if incremental:
            assert info['paper_reward'] == -50. and not info['retained']
        with pytest.raises(RuntimeError, match='reset'):
            env.step(action)
        _, reset_info = env.reset(seed=0)
        assert not reset_info['palm_support_violation']
        assert reset_info['peak_palm_support_force'] == 0.
        assert reset_info['failure_reason'] is None
    finally:
        env.close()


def test_geometric_palm_contacts_with_zero_normal_force_are_not_support():
    env = AllegroRotationEnv(EnvConfig(initial_grasp=GRASP, control_seconds=.002))
    try:
        env.reset(seed=0)
        place_near_palm(env, (.025, 0., .01))
        palm_contacts = [c for c in env.data.contact if env.object_geom in c.geom
                         and env.palm_id in env.model.geom_bodyid[c.geom]]
        assert len(palm_contacts) >= 1
        assert env.palm_support_force() == 0.
        _, _, terminated, truncated, info = env.step(env.to_action(env.data.ctrl))
        assert not terminated and not truncated
        assert not info['palm_support_violation']
        assert info['peak_palm_support_force'] == 0.
    finally:
        env.close()


def test_palm_force_method_reads_supplied_snapshot_without_changing_live_state():
    env = AllegroRotationEnv(EnvConfig(initial_grasp=GRASP))
    try:
        env.reset(seed=0)
        snapshot = mujoco.MjData(env.model)
        live = env.data.qpos.copy()
        place_near_palm(env)
        force = env.palm_support_force()
        mujoco.mj_copyData(snapshot, env.model, env.data)
        env.data.qpos[:] = live
        mujoco.mj_forward(env.model, env.data)
        assert env.palm_support_force() == 0.
        assert env.palm_support_force(snapshot) == pytest.approx(force)
        np.testing.assert_array_equal(env.data.qpos, live)
    finally:
        env.close()


def test_transient_support_is_sticky_and_cannot_become_success(monkeypatch):
    env = AllegroRotationEnv(EnvConfig(initial_grasp=GRASP))
    try:
        env.reset(seed=0)
        env.angle = env.retained_angle = 2*np.pi+.05
        calls = 0

        def transient_force(data=None):
            nonlocal calls
            calls += 1
            return .25 if calls == 1 else 0.

        monkeypatch.setattr(env, 'palm_support_force', transient_force)
        _, reward, terminated, truncated, info = env.step(env.to_action(env.data.ctrl))
        assert env.data.time == pytest.approx(.1)
        assert terminated and not truncated and reward == -50.
        assert not info['drop'] and not info['success'] and env.rotations == 0
        assert info['palm_support_force'] == 0.
        assert info['peak_palm_support_force'] == .25
        assert info['palm_support_violation'] and info['failure_reason'] == 'palm_support'
        assert info['retained_positive_step'] == 0.
    finally:
        env.close()


def test_reset_checks_initial_support_and_checkpoint_restores_sticky_state(tmp_path):
    env = AllegroRotationEnv(EnvConfig(initial_grasp=GRASP, control_seconds=.002))
    try:
        env.reset(seed=0)
        original = {name: getattr(env.data, name).copy() for name in ('qpos', 'qvel', 'ctrl')}
        place_near_palm(env)
        supported = tmp_path/'supported_grasp.npz'
        np.savez(supported, **{name: getattr(env.data, name).copy() for name in original})
        env.config.initial_grasp = str(supported)
        observation, info = env.reset(seed=0)
        assert info['palm_support_violation'] and info['peak_palm_support_force'] > .1
        peak = info['peak_palm_support_force']
        # Removing contact cannot clear the initial violation.
        for name, value in original.items():
            getattr(env.data, name)[:] = value
        mujoco.mj_forward(env.model, env.data)
        assert env.palm_support_force() == 0.
        checkpoint = tmp_path/'environment.npz'
        save_environment(checkpoint, env, observation, 0., [env.data.qpos[env.q_ids].copy()])
        restored = AllegroRotationEnv(env.config)
        try:
            restore_environment(checkpoint, restored)
            assert restored._palm_support_violation
            assert restored._peak_palm_support_force == peak
            _, reward, terminated, _, info = restored.step(restored.to_action(restored.data.ctrl))
            assert terminated and reward == -50.
            assert info['failure_reason'] == 'palm_support' and not info['drop']
            assert not info['success'] and info['retained_positive_step'] == 0.
        finally:
            restored.close()
    finally:
        env.close()


def test_rotation_support_rule_does_not_change_reorientation_failure_semantics():
    env = AllegroRotationEnv(EnvConfig(initial_grasp=GRASP, task='reorientation',
                                      control_seconds=.002, goal_heading=0.))
    try:
        env.reset(seed=0)
        place_near_palm(env)
        assert env.palm_support_force() > .1
        _, reward, terminated, truncated, info = env.step(env.to_action(env.data.ctrl))
        assert not terminated and not truncated and not info['drop']
        assert info['peak_palm_support_force'] > .1
        assert not info['palm_support_violation'] and info['failure_reason'] is None
        assert reward != -50.
    finally:
        env.close()


@pytest.mark.parametrize('tolerance', [-1., float('nan'), float('inf')])
def test_palm_support_tolerance_must_be_finite_and_nonnegative(tolerance):
    with pytest.raises(ValueError, match='Palm support force tolerance'):
        AllegroRotationEnv(EnvConfig(palm_support_force_tolerance=tolerance))
