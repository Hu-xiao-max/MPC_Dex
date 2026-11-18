"""Integration checks, never evidence of a trained rotation policy."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest

from mpc_dex.contracts import validate_checkpoint,write_checkpoint_metadata
from mpc_dex.env import AllegroRotationEnv,EnvConfig
from mpc_dex.learning import SACAgent,LearnerConfig
from mpc_dex.replay import ReplayBuffer,sample_mixed
from mpc_dex.primitives import PrimitiveBasis
from mpc_dex.mpc import PrimitiveCEM,CEMConfig
from mpc_dex.rollouts import MujocoRolloutEvaluator

GRASP=str(Path(__file__).resolve().parents[1]/'assets/initial_grasp.npz')


def test_real_contact_transitions_feed_all_learning_stages_and_reload(tmp_path):
    ec=EnvConfig(initial_grasp=GRASP,horizon_steps=4)
    env=AllegroRotationEnv(ec)
    offline=ReplayBuffer(100);online=ReplayBuffer(100)
    try:
        for buffer in (offline,online):
            obs,_=env.reset(seed=0)
            for _ in range(4):
                action=env.to_action(env.data.ctrl)
                next_obs,r,done,truncated,_=env.step(action)
                buffer.add_transition(obs,action,r,next_obs,done,truncated)
                obs=next_obs
        # Actual measured contact data exercises interfaces; it is not a successful demonstration.
        agent=SACAgent(config=LearnerConfig(hidden_sizes=(16,16),ensemble_size=10))
        assert np.isfinite(agent.bc_step(offline.sample(16,weighted_episodes=True))['bc_loss'])
        agent.initialize_std()
        assert agent.pretrain_critic_step(offline.sample(32))['pretrain_samples']>0
        assert np.isfinite(agent.update(sample_mixed(offline,online,32))['critic_loss'])
        checkpoint=tmp_path/'learner.pt'
        agent.save(checkpoint)
        write_checkpoint_metadata(checkpoint,ec,agent.config,stage='integration-test')
        validate_checkpoint(checkpoint,ec)
        restored=SACAgent.load(checkpoint)
        np.testing.assert_array_equal(agent.act(obs,True),restored.act(obs,True))
        with pytest.raises(ValueError,match='contract mismatch'):
            validate_checkpoint(checkpoint,EnvConfig(initial_grasp=GRASP,horizon_steps=4,object_mass=.18))
        checkpoint.write_bytes(checkpoint.read_bytes()+b'changed')
        with pytest.raises(ValueError,match='hash mismatch'):validate_checkpoint(checkpoint,ec)
    finally:env.close()


def test_primitive_cem_predicts_feasible_free_object_motion_without_mutation():
    env=AllegroRotationEnv(EnvConfig(initial_grasp=GRASP))
    evaluator=MujocoRolloutEvaluator(env,timestep_seconds=.1)
    try:
        env.reset()
        # Arbitrary numerical basis belongs only to this interface test. It is
        # neither stored nor admitted as a trained rotation primitive.
        frames=env.data.ctrl+np.random.default_rng(0).normal(0,.01,(40,16))
        primitive=PrimitiveBasis.from_trajectories(frames,provenance={'source':'numerical integration test only'})
        planner=PrimitiveCEM(primitive,env.low,env.high,CEMConfig(samples=100,iterations=2,timestep_seconds=.1),seed=2)
        planner.reset(env.data.ctrl)
        before=env.data.qpos.copy()
        action,metrics=planner.plan(evaluator)
        assert metrics['valid_candidates']>0
        np.testing.assert_array_equal(env.data.qpos,before)
        obs,reward,done,_,info=env.step(env.to_action(action))
        assert np.isfinite(obs).all() and np.isfinite(reward) and not info['drop']
    finally:evaluator.close();env.close()
