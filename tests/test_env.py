import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from mpc_dex.env import AllegroRotationEnv, EnvConfig
from mpc_dex.rollouts import MujocoRolloutEvaluator

GRASP=str(Path(__file__).resolve().parents[1]/'assets/initial_grasp.npz')


def test_contact_model_and_observation_contract():
    env=AllegroRotationEnv(EnvConfig(initial_grasp=GRASP))
    try:
        obs,_=env.reset(seed=1)
        assert obs.shape==(60,)
        assert env.model.nu==16 and env.model.joint('object_joint').type==mujoco.mjtJoint.mjJNT_FREE
        assert env.model.neq==0
        np.testing.assert_allclose(env.to_joint_targets(env.to_action(env.data.ctrl)),env.data.ctrl,atol=1e-7)
        check_env(env,skip_render_check=True)
    finally:env.close()


def test_true_unsupported_grasp_holds_and_no_progress_is_faked():
    env=AllegroRotationEnv(EnvConfig(initial_grasp=GRASP))
    try:
        env.reset(seed=0)
        initial=env.object_position().copy()
        action=env.to_action(env.data.ctrl)
        for _ in range(60):
            _,reward,done,truncated,info=env.step(action)
            assert not done and not truncated and not info['drop'] and not info['success']
            for c in env.data.contact:
                if env.object_geom in (c.geom1,c.geom2):
                    other=c.geom2 if c.geom1==env.object_geom else c.geom1
                    assert other != env.table_geom
                    assert env.model.geom_bodyid[other] != env.palm_id
        assert abs(info['angle'])<.1
        assert np.linalg.norm(env.object_position()-initial)<.005
        assert info['rotations']==0
    finally:env.close()


def test_drop_overrides_reward_and_stops_episode():
    env=AllegroRotationEnv(EnvConfig(initial_grasp=GRASP))
    try:
        env.reset(seed=0)
        env.data.qpos[env.object_qadr+2]=.01
        mujoco.mj_forward(env.model,env.data)
        _,reward,done,truncated,info=env.step(env.to_action(env.data.ctrl))
        assert done and not truncated and info['drop'] and reward==-50
        with pytest.raises(RuntimeError):env.step(np.zeros(16))
    finally:env.close()


def test_time_limit_is_distinct_from_terminal_failure():
    env=AllegroRotationEnv(EnvConfig(initial_grasp=GRASP,horizon_steps=1))
    try:
        env.reset()
        _,_,terminated,truncated,_=env.step(env.to_action(env.data.ctrl))
        assert not terminated and truncated
    finally:env.close()


def test_rollout_prediction_matches_executed_control_and_does_not_mutate_state():
    env=AllegroRotationEnv(EnvConfig(initial_grasp=GRASP))
    evaluator=MujocoRolloutEvaluator(env,timestep_seconds=.1)
    try:
        env.reset()
        before=env.data.qpos.copy();ctrl=env.data.ctrl.copy()
        costs=evaluator(np.broadcast_to(ctrl,(4,5,16)).copy())
        assert np.isfinite(costs).all()
        np.testing.assert_allclose(costs,costs[0])
        np.testing.assert_array_equal(env.data.qpos,before)
        state=np.empty(mujoco.mj_stateSize(env.model,evaluator.state_spec))
        mujoco.mj_getState(env.model,env.data,state,evaluator.state_spec)
        prediction,_=evaluator.pool.rollout(env.model,evaluator.worker_data,state[None],
            np.broadcast_to(ctrl,(1,env.substeps,16)).copy(),
            initial_warmstart=env.data.qacc_warmstart[None].copy())
        env.step(env.to_action(ctrl))
        np.testing.assert_allclose(env.data.qpos,prediction[0,-1,1:1+env.model.nq],atol=2e-6)
        env.data.qpos[env.object_qadr+2]=.01
        mujoco.mj_forward(env.model,env.data)
        assert np.isinf(evaluator(np.broadcast_to(ctrl,(4,5,16)).copy())).all()
    finally:evaluator.close();env.close()


def test_reorientation_success_is_one_time_bonus_and_reset_clears_error():
    env=AllegroRotationEnv(EnvConfig(initial_grasp=GRASP,task='reorientation',goal_heading=0,
                                    terminate_on_success=False))
    try:
        obs,_=env.reset()
        assert obs.shape==(61,)
        rewards=[]
        for _ in range(10):
            _,r,done,_,_=env.step(env.to_action(env.data.ctrl));rewards.append(r)
            assert not done
        assert sum(r>100 for r in rewards)==1
        env.heading_error=123
        env.reset()
        assert env.heading_error==0
    finally:env.close()


def test_principal_twist_does_not_count_perpendicular_tumbles_or_quaternion_signs():
    from mpc_dex.env import axial_increment
    from scipy.spatial.transform import Rotation
    for axis in (np.array([1.,0,0]),np.array([0.,1,0]),np.array([0.,0,1])):
        quats=Rotation.from_rotvec(np.linspace(0,2*np.pi,201)[:,None]*axis).as_quat(scalar_first=True)
        quats[::2]*=-1
        angle=sum(axial_increment(a,b) for a,b in zip(quats[:-1],quats[1:]))
        assert angle==pytest.approx(2*np.pi if axis[1] else 0,abs=1e-10)


def test_airborne_spin_cannot_be_a_success_before_object_reaches_table():
    env=AllegroRotationEnv(EnvConfig(initial_grasp=GRASP))
    try:
        env.reset()
        # Displace the free object out of the fingers while it remains far above
        # the table, and give it axial spin. This is an explicit adversarial state.
        env.data.qpos[env.object_qadr]+=0.1
        env.data.qvel[env.object_dadr+4]=100.
        mujoco.mj_forward(env.model,env.data)
        _,reward,done,_,info=env.step(env.to_action(env.data.ctrl))
        assert done and info['drop'] and not info['success'] and reward==-50
        assert env.data.xpos[env.object_id,2]>.1
    finally:env.close()
