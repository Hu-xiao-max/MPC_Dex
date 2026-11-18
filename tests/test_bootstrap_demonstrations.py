"""Verified demonstration preparation and seeding never manufacture SAC success."""
from dataclasses import asdict
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mpc_dex.contracts import sha256
from mpc_dex.env import AllegroRotationEnv, EnvConfig
from mpc_dex.replay import ReplayBuffer
from scripts.prepare_bootstrap_demonstrations import prepare
from scripts.train_bootstrap import BootstrapRunner, load_bootstrap_seed, parser, read_environment


ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'artifacts/mpc_matched40_reference_0001/transitions.npz'
BRANCH=ROOT/'artifacts/mpc_fullturn_prior_40mm_0001/branch_diagnosis/prefix19_blend25/transitions.npz'
STRICT=ROOT/'artifacts/mpc_fullturn_prior_40mm_0001/branch_diagnosis/strict_config.json'


@pytest.fixture(scope='module')
def real_demonstrations(tmp_path_factory):
    if not all(path.is_file() for path in (REFERENCE,BRANCH,STRICT)):
        pytest.skip('Measured physical full-turn artifacts are unavailable; no synthetic success substitute')
    directory=tmp_path_factory.mktemp('real_demonstrations')
    config=json.loads(STRICT.read_text())
    config.setdefault('bootstrap',{}).pop('control',None)
    path=directory/'config.json';path.write_text(json.dumps(config))
    output=directory/'replay.npz'
    metadata=prepare([REFERENCE,BRANCH],path,output)
    return path,output,metadata


def test_real_sources_stop_at_first_success_and_preserve_successor_boundaries(real_demonstrations):
    config,path,metadata=real_demonstrations
    assert metadata['demonstration_successes']==2
    assert metadata['demonstrator']=='MPC diagnostic (not SAC)'
    assert metadata['primitive_eligible'] is False and metadata['main_mpc_dataset_eligible'] is False
    _,ec=read_environment(config)
    replay,_,episodes=load_bootstrap_seed(path,ec,{'kind':'absolute_joint_target'})
    assert replay.episode_count==2 and len(episodes)==2
    reference=metadata['episodes'][0]
    assert reference['source_commands']==60
    assert reference['unused_tail_commands']>0
    assert reference['used_commands']+reference['unused_tail_commands']==60
    for episode,record in zip(episodes,metadata['episodes']):
        assert record['audit']['finger_supported_rotation_success']
        assert max(record['max_replay_errors'].values())==0.
        assert episode['terminated'][-1] and not episode['terminated'][:-1].any()
        assert not episode['truncated'].any() and not episode['next_action_valid'][-1]
        np.testing.assert_array_equal(episode['next_actions'][:-1],episode['actions'][1:])


@pytest.mark.parametrize('corruption',['count','palm','audit','force_progress','tail','terminal_next_action',
                                     'interior_next_action','missing_source_hash','invalid_source_hash'])
def test_loader_rejects_corrupted_episode_evidence_and_actions(real_demonstrations,tmp_path,corruption):
    config,source,original=real_demonstrations
    path=tmp_path/'seed.npz';replay=ReplayBuffer.load(source)
    metadata=copy.deepcopy(original)
    if corruption=='count':metadata['demonstration_successes']=1
    elif corruption=='palm':metadata['episodes'][0]['final_info']['peak_palm_support_force']=.01
    elif corruption=='audit':metadata['episodes'][0]['audit']['finger_supported_rotation_success']=False
    elif corruption=='force_progress':metadata['episodes'][0]['audit']['final_force_qualified_angle']=6.
    elif corruption=='tail':metadata['episodes'][0]['unused_tail_commands']+=1
    elif corruption=='missing_source_hash':del metadata['episodes'][0]['source_sha256']
    elif corruption=='invalid_source_hash':metadata['episodes'][0]['source_sha256']='g'*64
    elif corruption=='terminal_next_action':
        index=np.flatnonzero(replay._arrays['terminated'][:len(replay)])[0]
        replay._arrays['next_action_valid'][index]=True
    else:replay._arrays['next_action_valid'][0]=False
    replay.save(path);metadata['replay_sha256']=sha256(path)
    path.with_suffix('.npz.metadata.json').write_text(json.dumps(metadata))
    _,ec=read_environment(config)
    with pytest.raises(ValueError):load_bootstrap_seed(path,ec,{'kind':'absolute_joint_target'})


@pytest.mark.parametrize('field',['observation','action'])
def test_loader_rejects_replay_dimensions_inconsistent_with_contract(real_demonstrations,tmp_path,field):
    config,source,original=real_demonstrations
    replay=ReplayBuffer.load(source)
    rows=replay.transitions()
    keys=('obs','next_obs') if field=='observation' else ('actions','next_actions')
    for key in keys:rows[key]=rows[key][:,:-1]
    changed=ReplayBuffer(len(replay));start=0
    for end in np.flatnonzero(rows['terminated'] | rows['truncated']):
        changed.add_episode({key:value[start:end+1] for key,value in rows.items()})
        start=end+1
    path=tmp_path/'dimensions.npz';changed.save(path)
    metadata=copy.deepcopy(original);metadata['replay_sha256']=sha256(path)
    path.with_suffix('.npz.metadata.json').write_text(json.dumps(metadata))
    _,ec=read_environment(config)
    with pytest.raises(ValueError,match='dimensions'):
        load_bootstrap_seed(path,ec,{'kind':'absolute_joint_target'})


def test_float64_controls_are_converted_before_physical_replay_and_audit(real_demonstrations,tmp_path):
    config,_,_=real_demonstrations
    with np.load(REFERENCE,allow_pickle=False) as archive:
        values={key:archive[key].copy() for key in archive.files}
    original=values['actions']
    assert original.dtype==np.float32
    # A genuine sub-ULP perturbation disappears in replay storage. The exact
    # stored float32 commands must therefore be the commands physically tested.
    toward_zero=np.nextafter(original,np.float32(0)).astype(np.float64)
    promoted=original.astype(np.float64)
    values['actions']=promoted+.25*(toward_zero-promoted)
    np.testing.assert_array_equal(values['actions'].astype(np.float32),original)
    assert np.any(values['actions']!=promoted)
    source=tmp_path/'float64_source.npz';np.savez(source,**values)
    output=tmp_path/'converted.npz'
    metadata=prepare([source],config,output)
    record=metadata['episodes'][0]
    conversion=record['action_conversion']
    assert conversion['source_dtype']=='float64' and conversion['effective_dtype']=='float32'
    assert conversion['max_absolute_difference']>0 and conversion['applied_before_physics']
    assert max(record['max_replay_errors'].values())==0.
    assert record['audit']['finger_supported_rotation_success']
    replay=ReplayBuffer.load(output).transitions()
    used=record['used_commands']
    np.testing.assert_array_equal(replay['actions'],original[:used])
    directory=Path(record['audit_directory'])
    commands=json.loads((directory/'commands.json').read_text())
    audited=np.asarray([command['policy_action'] for command in commands],dtype=np.float32)
    np.testing.assert_array_equal(replay['actions'],audited)
    with np.load(directory/'actual_transitions.npz',allow_pickle=False) as archive:
        np.testing.assert_array_equal(replay['actions'],archive['actions'])


def test_preparation_rejects_changed_recorded_state_and_incremental_config(real_demonstrations,tmp_path):
    config,_,_=real_demonstrations
    with np.load(REFERENCE) as archive:values={key:archive[key].copy() for key in archive.files}
    values['qpos'][0,0]+=.01
    changed=tmp_path/'changed.npz';np.savez(changed,**values)
    with pytest.raises(ValueError,match='qpos does not replay'):
        prepare([changed],config,tmp_path/'invalid.npz')
    assert not (tmp_path/'invalid.npz').exists()
    cfg=json.loads(config.read_text());cfg.setdefault('bootstrap',{})['control']={'max_delta_radians':.02}
    incremental=tmp_path/'incremental.json';incremental.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='absolute config'):
        prepare([REFERENCE],incremental,tmp_path/'incremental.npz')
    with pytest.raises(ValueError,match='Duplicate'):
        prepare([REFERENCE,REFERENCE],config,tmp_path/'duplicates.npz')


def test_partial_motion_cannot_be_labeled_verified_demonstration(tmp_path):
    ec=EnvConfig(initial_grasp=str(ROOT/'assets/initial_grasp.npz'))
    config=tmp_path/'config.json';config.write_text(json.dumps({'environment':asdict(ec)}))
    env=AllegroRotationEnv(ec);env.reset(seed=0)
    try:action=env.to_action(env.data.ctrl)
    finally:env.close()
    source=tmp_path/'partial.npz';np.savez(source,actions=np.tile(action,(3,1)))
    with pytest.raises(ValueError,match='genuine full-turn'):
        prepare([source],config,tmp_path/'replay.npz')
    assert not (tmp_path/'replay.npz').exists()


def test_multiple_seed_episodes_survive_checkpoint_without_online_success_counts(real_demonstrations,tmp_path):
    original_config,seed,metadata=real_demonstrations
    cfg=json.loads(original_config.read_text())
    cfg['bootstrap']={'seed':{'replay':str(seed),'bc_updates':0,'critic_updates':0},
                      'learner':{'hidden_sizes':[8,8]}}
    config=tmp_path/'config.json';config.write_text(json.dumps(cfg))
    args=parser().parse_args(['--config',str(config),'--steps','0','--output',str(tmp_path/'run'),
                             '--num-envs','2','--batch-size','4','--replay-capacity','256'])
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:
        for _ in range(2):  # Initial checkpoint then an exact resume, no training updates.
            runner=BootstrapRunner(args)
            assert runner.demonstration_successes==2
            assert runner.environment_steps==runner.completed_episodes==runner.successes==0
            assert runner.agent.bc_updates==runner.agent.pretrain_updates==runner.agent.online_updates==0
            assert not runner.success_records and not runner.successful_traces
            assert sum(buffer.episode_count for buffer in runner.buffers)==2
            assert sum(map(len,runner.buffers))==metadata['transitions']
            assert runner.run()==0
        saved=json.loads((tmp_path/'run/run.json').read_text())
        assert saved['demonstration_successes']==2
        assert saved['successes']==saved['successful_trajectory_episodes']==saved['environment_steps']==0
        assert not (tmp_path/'run/successful_joint_trajectories.npz').exists()
    finally:torch.set_num_threads(previous)
