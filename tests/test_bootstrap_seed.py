"""Partial demonstrations must replay under the actual bootstrap controller."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mpc_dex.env import AllegroRotationEnv, EnvConfig
from mpc_dex.replay import ReplayBuffer
from scripts.prepare_bootstrap_seed import prepare
from scripts.train_bootstrap import BootstrapRunner, parser


ROOT=Path(__file__).resolve().parents[1]


def fixture_files(tmp_path):
    cfg=json.loads((ROOT/'configs/bootstrap_incremental.json').read_text())
    cfg['environment']['initial_grasp']=str(ROOT/'assets/initial_grasp.npz')
    cfg['bootstrap']['learner'].update(hidden_sizes=[16,16])
    config=tmp_path/'config.json';config.write_text(json.dumps(cfg))
    env=AllegroRotationEnv(EnvConfig(**cfg['environment']))
    env.reset();actions=[];next_obs=[]
    try:
        for _ in range(3):
            action=env.to_action(env.data.ctrl)
            obs,_,done,trunc,_=env.step(action)
            assert not done and not trunc
            actions.append(action);next_obs.append(obs)
    finally:env.close()
    source=tmp_path/'source.npz'
    np.savez(source,actions=actions,next_observations=next_obs)
    return config,source


def test_partial_seed_reexecutes_real_contact_and_never_counts_as_success(tmp_path):
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:
        config,source=fixture_files(tmp_path)
        seed=tmp_path/'seed.npz';meta=prepare(source,config,seed)
        assert meta['successful_full_turns']==0
        replay=ReplayBuffer.load(seed);rows=replay.transitions()
        assert len(replay)==3 and rows['truncated'].tolist()==[False,False,True]
        assert not rows['terminated'].any() and not rows['next_action_valid'][-1]
        cfg=json.loads(config.read_text());cfg['bootstrap']['seed']={'replay':str(seed),'bc_updates':2,'critic_updates':2}
        config.write_text(json.dumps(cfg))
        args=parser().parse_args(['--config',str(config),'--steps','0','--output',str(tmp_path/'run'),
                                 '--num-envs','1','--batch-size','4','--replay-capacity','64'])
        runner=BootstrapRunner(args)
        assert runner.agent.bc_updates==runner.agent.pretrain_updates==2
        assert runner.environment_steps==runner.completed_episodes==runner.successes==0
        assert len(runner.buffers[0])==3
        assert runner.run()==0
        args.steps=1
        resumed=BootstrapRunner(args)
        assert resumed.agent.bc_updates==2
        assert resumed.run()==0
        state=json.loads((tmp_path/'run'/'run.json').read_text())
        assert state['environment_steps']==1 and state['replay_transitions']==4
        assert state['successes']==state['successful_trajectory_episodes']==0
    finally:torch.set_num_threads(previous)


def test_partial_seed_rejects_a_nonreplaying_trajectory(tmp_path):
    config,source=fixture_files(tmp_path)
    with np.load(source) as archive:
        actions=archive['actions'];expected=archive['next_observations'].copy()
    expected[0,0]+=.01
    np.savez(source,actions=actions,next_observations=expected)
    with pytest.raises(ValueError,match='does not replay'):
        prepare(source,config,tmp_path/'seed.npz')
    assert not (tmp_path/'seed.npz').exists()
