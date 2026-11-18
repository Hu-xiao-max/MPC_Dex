"""Check the prerequisite controller without changing main-method semantics."""
from pathlib import Path
import math

import numpy as np
import pytest

from mpc_dex.bootstrap_env import BootstrapRotationEnv,BootstrapControlConfig,policy_interface,environment_for_policy
from mpc_dex.env import AllegroRotationEnv,EnvConfig

GRASP=str(Path(__file__).resolve().parents[1]/'assets/initial_grasp.npz')


def test_increment_action_is_rate_bounded_and_observation_records_executed_target():
    env=BootstrapRotationEnv(EnvConfig(initial_grasp=GRASP),BootstrapControlConfig())
    try:
        env.reset();before=env.data.ctrl.copy()
        action=np.linspace(-1,1,16)
        expected=env.joint_targets_for_increment(action)
        obs,reward,_,_,info=env.step(action)
        assert np.max(np.abs(expected-before))<=.020000001
        np.testing.assert_allclose(env.data.ctrl,expected,atol=1e-7)
        np.testing.assert_allclose(obs[-16:],env.to_action(expected),atol=1e-7)
        assert 'paper_reward' in info and info['bootstrap_reward']==reward
    finally:env.close()


def test_small_increment_exploration_retains_true_grasp():
    env=BootstrapRotationEnv(EnvConfig(initial_grasp=GRASP),BootstrapControlConfig())
    try:
        env.reset();rng=np.random.default_rng(8)
        for _ in range(100):
            _,_,done,truncated,info=env.step(np.clip(rng.normal(0,.05,16),-1,1))
            assert not done and not truncated and not info['drop']
        assert info['finger_contacts']>=2
    finally:env.close()


def test_bootstrap_reward_does_not_reward_reverse_clipping_cycles(monkeypatch):
    env=BootstrapRotationEnv(EnvConfig(initial_grasp=GRASP),BootstrapControlConfig())
    env.reset()
    def fake_step(_self,_action):
        info={'finger_contacts':4,'axis_tilt_radians':0.,'drop':False,'success':False,
              'palm_support_violation':False,
              'retained_positive_step':.8,'retained_negative_step':-.8}
        return np.zeros(60),-.3,False,False,info
    monkeypatch.setattr(AllegroRotationEnv,'step',fake_step)
    try:
        _,reward,_,_,info=env.step(np.zeros(16))
        assert info['rewarded_progress_radians']==pytest.approx(-.7)
        assert reward<0
    finally:env.close()


def test_policy_interface_cannot_silently_use_absolute_action_environment():
    settings={'control':{'max_delta_radians':.02}}
    interface=policy_interface(settings)
    env=environment_for_policy(EnvConfig(initial_grasp=GRASP),interface)
    assert isinstance(env,BootstrapRotationEnv)
    env.close()
    interface['implementation_sha256']='incorrect'
    with pytest.raises(ValueError,match='implementation changed'):
        environment_for_policy(EnvConfig(initial_grasp=GRASP),interface)


@pytest.mark.parametrize('incremental',[False,True])
def test_unretained_rotation_does_not_accumulate_qualified_success(incremental):
    import mujoco
    ec=EnvConfig(initial_grasp=GRASP)
    env=BootstrapRotationEnv(ec,BootstrapControlConfig()) if incremental else AllegroRotationEnv(ec)
    try:
        env.reset()
        env.data.qpos[env.object_qadr]+=.1
        env.data.qvel[env.object_dadr+4]=100.
        mujoco.mj_forward(env.model,env.data)
        obs,_,done,_,info=env.step(np.zeros(16) if incremental else env.to_action(env.data.ctrl))
        assert done and info['drop'] and not info['success']
        assert info['angle']>1.
        assert info['retained_angle']<=1e-8
        assert obs[23]==pytest.approx(info['retained_angle'] if incremental else info['angle'])
    finally:env.close()
