"""Artifact validation tests; synthetic records are never rotation evidence."""
from dataclasses import asdict
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mpc_dex import cli
from mpc_dex.bootstrap_env import create_bootstrap_environment, policy_interface
from mpc_dex.contracts import environment_contract, write_checkpoint_metadata
from mpc_dex.env import EnvConfig
from mpc_dex.learning import LearnerConfig, SACAgent

GRASP=str(Path(__file__).resolve().parents[1]/'assets/initial_grasp.npz')


@pytest.fixture(autouse=True)
def small_torch_pool():
    previous=torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def rotation_record():
    ec=EnvConfig(initial_grasp=GRASP)
    record={'success':True,'drop':False,'angle':2*np.pi,'retained_angle':2*np.pi,
            'finger_contacts':2,'axis_tilt_radians':0.,'steps':2,'frames':3,
            'palm_support_violation':False,'peak_palm_support_force':0.,'failure_reason':None}
    metadata={'source':'synthetic artifact-validation unit test only','task':'rotation',
              'successful_episodes':1,'environment':asdict(ec),'episodes':[record],
              'rotation_validation':cli.rotation_validation(ec),'environment_contract':environment_contract(ec)}
    return np.arange(48,dtype=float).reshape(3,16),metadata


def test_extraction_rejects_reorientation_even_if_rotation_label_is_present():
    frames,metadata=rotation_record()
    metadata['task']='reorientation'
    metadata['environment']['task']='reorientation'
    with pytest.raises(ValueError,match='rotation trajectories'):
        cli.validate_rotation_trajectories(frames,metadata)


@pytest.mark.parametrize('changed',[
    {'angle':np.pi},{'retained_angle':np.pi},{'retained_angle':float('nan')},
    {'drop':True},{'success':False},{'finger_contacts':1},{'axis_tilt_radians':np.pi/2},
    {'palm_support_violation':True},{'peak_palm_support_force':1e-5},
    {'peak_palm_support_force':float('nan')},{'peak_palm_support_force':-.1},
    {'failure_reason':'palm_support'},
])
def test_extraction_requires_each_episode_to_prove_retained_rotation(changed):
    frames,metadata=rotation_record()
    metadata['episodes'][0].update(changed)
    with pytest.raises(ValueError,match='full retained rotation'):
        cli.validate_rotation_trajectories(frames,metadata)


@pytest.mark.parametrize('field',['palm_support_violation','peak_palm_support_force'])
def test_extraction_requires_explicit_episode_palm_support_evidence(field):
    frames,metadata=rotation_record()
    metadata['episodes'][0].pop(field)
    with pytest.raises(ValueError,match='without a drop or palm support'):
        cli.validate_rotation_trajectories(frames,metadata)


def test_extraction_rejects_previous_validation_without_palm_support_gate():
    frames,metadata=rotation_record()
    metadata['rotation_validation']='incremental-principal-twist-with-grasp-retention-v2'
    with pytest.raises(ValueError,match='no-palm support'):
        cli.validate_rotation_trajectories(frames,metadata)


def test_extraction_requires_complete_episode_records_and_terminal_frames():
    frames,metadata=rotation_record()
    cli.validate_rotation_trajectories(frames,metadata)
    missing=copy.deepcopy(metadata);missing.pop('episodes')
    with pytest.raises(ValueError,match='validation record'):
        cli.validate_rotation_trajectories(frames,missing)
    with pytest.raises(ValueError,match='joint frames'):
        cli.validate_rotation_trajectories(frames[:-1],metadata)
    metadata['episodes'][0]['frames']=2
    with pytest.raises(ValueError,match='terminal joint frames'):
        cli.validate_rotation_trajectories(frames[:-1],metadata)


def make_checkpoint(tmp_path, ec, settings):
    env=create_bootstrap_environment(ec,settings)
    try:
        observation,_=env.reset()
        agent=SACAgent(len(observation),16,LearnerConfig(hidden_sizes=(16,16),ensemble_size=2))
        initial=np.zeros(16) if settings else env.to_action(env.data.ctrl)
        with torch.no_grad():
            agent.actor.mean_head.weight.zero_()
            agent.actor.mean_head.bias.copy_(torch.tensor(np.arctanh(initial),dtype=torch.float32))
        agent.initialize_std()
        checkpoint=tmp_path/'agent.pt';agent.save(checkpoint)
        write_checkpoint_metadata(checkpoint,ec,agent.config,stage='test',policy_interface=policy_interface(settings))
        config=tmp_path/'config.json';config.write_text(json.dumps({'environment':asdict(ec)}))
        return SimpleNamespace(config=str(config),checkpoint=str(checkpoint),device='cpu',
                               output=str(tmp_path/'evaluation'),episodes=1,seed=0)
    finally:
        env.close()


def test_real_reorientation_evaluation_keeps_terminal_frame_and_cannot_supply_rotation_primitive(tmp_path):
    args=make_checkpoint(tmp_path,EnvConfig(initial_grasp=GRASP,task='reorientation',goal_heading=0.,horizon_steps=5),{})
    assert cli.evaluate(args)==0
    trajectory=Path(args.output)/'successful_joint_trajectories.npz'
    with np.load(trajectory,allow_pickle=False) as archive:
        metadata=json.loads(str(archive['metadata']))
        assert metadata['rotation_validation'] is None
        assert metadata['episodes'][0]['steps']==5
        assert metadata['episodes'][0]['frames']==len(archive['joint_positions'])==6
    with pytest.raises(ValueError,match='rotation trajectories'):
        cli.extract(SimpleNamespace(trajectory=str(trajectory),output=str(tmp_path/'primitive.npz')))
    assert not (tmp_path/'primitive.npz').exists()


def test_incremental_evaluation_reports_its_interface_and_both_rewards(tmp_path):
    settings={'control':{'max_delta_radians':.02}}
    args=make_checkpoint(tmp_path,EnvConfig(initial_grasp=GRASP,horizon_steps=2),settings)
    assert cli.evaluate(args)==2
    state=json.loads((Path(args.output)/'run.json').read_text())
    assert state['policy_interface']['kind']=='bootstrap_joint_increment'
    assert state['reward_semantics']=='bootstrap retained-rotation shaping'
    result=state['results'][0]
    assert result['return']==result['return_value']
    assert result['paper_return']!=result['return']
    assert result['frames']==result['steps']+1
    assert state['policy_origin']=='untrained'
    assert state['learner_counters']['online_updates']==0
    assert state['eligible_for_primitive'] is False


def test_legacy_success_records_cannot_be_promoted_as_frozen_sac(tmp_path):
    frames,metadata=rotation_record()
    path=tmp_path/'legacy.npz'
    np.savez_compressed(path,joint_positions=frames,metadata=np.asarray(json.dumps(metadata)))
    with pytest.raises(ValueError,match='frozen simulation SAC'):
        cli.extract(SimpleNamespace(trajectory=str(path),output=str(tmp_path/'primitive.npz')))
    assert not (tmp_path/'primitive.npz').exists()


def test_main_planning_rejects_numerical_primitive_before_execution(tmp_path,monkeypatch):
    from mpc_dex.primitives import PrimitiveBasis
    primitive=tmp_path/'diagnostic.npz'
    PrimitiveBasis.from_trajectories(np.arange(48).reshape(3,16),
        provenance={'source':'synthetic numerical test only'}).save(primitive)
    env=SimpleNamespace(close=lambda:None)
    monkeypatch.setattr(cli,'planner_for',lambda *a,**kw:pytest.fail('Ineligible primitive reached planning'))
    run,state=cli.new_run(tmp_path/'main','collect',{},[primitive])
    with pytest.raises(ValueError,match='verified frozen-SAC primitive'):
        cli._planner_for_run(run,state,env,{},primitive,0)
    assert json.loads((run/'run.json').read_text())['status']=='failed'


def test_full_method_explicitly_rejects_incremental_checkpoint(tmp_path):
    args=make_checkpoint(tmp_path,EnvConfig(initial_grasp=GRASP),{'control':{'max_delta_radians':.02}})
    with pytest.raises(ValueError,match='absolute joint-target policy'):
        cli.train(args)
