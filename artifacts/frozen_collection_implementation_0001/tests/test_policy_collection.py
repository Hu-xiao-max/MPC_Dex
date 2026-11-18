"""Collector integrity tests; synthetic successful records are not physics proof."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from mpc_dex.contracts import environment_contract, metadata_path, sha256, write_checkpoint_metadata
from mpc_dex.env import EnvConfig, JOINT_NAMES, ROTATION_VALIDATION, make_model
from mpc_dex.learning import LearnerConfig, SACAgent
from mpc_dex import cli
from mpc_dex.primitives import PrimitiveBasis
from mpc_dex.trajectory_contracts import validate_primitive_source
from scripts import collect_policy_trajectories as collection

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def one_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def checkpoint_fixture(tmp_path, *, online=True, horizon=1):
    ec = EnvConfig(object_diameter=.04, initial_grasp=str(ROOT/'assets/initial_grasp.npz'), horizon_steps=horizon)
    config = tmp_path/'config.json'
    config.write_text(json.dumps({'environment':asdict(ec)}))
    torch.manual_seed(102)
    agent = SACAgent(60,16,LearnerConfig(hidden_sizes=(16,16),ensemble_size=2))
    agent.initialize_std()
    if online:
        rng = np.random.default_rng(61)
        agent.update(dict(obs=rng.normal(size=(4,60)),next_obs=rng.normal(size=(4,60)),
            actions=rng.uniform(-.3,.3,size=(4,16)),rewards=np.zeros(4),terminated=np.ones(4)))
        assert agent.online_updates == 1
    else:
        agent.bc_updates = 12000  # Counter fixture, explicitly not a training claim.
    path = tmp_path/'actor.pt'
    agent.save(path)
    write_checkpoint_metadata(path,ec,agent.config,stage='synthetic-unit-test-fixture-not-a-successful-policy')
    args = SimpleNamespace(checkpoint=str(path),config=str(config),output=str(tmp_path/'collection'),
                           seed=812,device='cpu',stochastic=False)
    return args,ec,agent


def test_imported_bc_history_cannot_enter_primitive_collection(tmp_path,monkeypatch):
    args,_,_ = checkpoint_fixture(tmp_path,online=False)
    def forbidden(*args,**kwargs):
        raise AssertionError('A zero-update BC actor must not execute a collection episode')
    monkeypatch.setattr(collection,'run_episode',forbidden)
    with pytest.raises(ValueError,match='positive online_updates'):
        collection.collect(args)
    assert not Path(args.output).exists()


@pytest.mark.parametrize('corruption,match',[
    ('checkpoint_bytes','hash mismatch'),('environment','environment/action contract'),
    ('learner_metadata','learner configuration'),('unknown_interface','policy interface'),
    ('noninteger_counter','integer learner counters'),('nonfinite_actor','nonfinite'),
])
def test_incompatible_or_unbound_checkpoint_is_rejected_before_physics(tmp_path,monkeypatch,corruption,match):
    args,ec,agent = checkpoint_fixture(tmp_path)
    checkpoint = Path(args.checkpoint)
    if corruption == 'checkpoint_bytes':
        with checkpoint.open('ab') as stream:stream.write(b'changed')
    elif corruption == 'environment':
        config=json.loads(Path(args.config).read_text());config['environment']['object_mass']=.16
        Path(args.config).write_text(json.dumps(config))
    elif corruption in ('learner_metadata','unknown_interface'):
        path=metadata_path(checkpoint);metadata=json.loads(path.read_text())
        if corruption=='learner_metadata':metadata['learner_config']['initial_std']=.12
        else:metadata['policy_interface']={'kind':'unknown-test-interface'}
        path.write_text(json.dumps(metadata))
    else:
        payload=torch.load(checkpoint,weights_only=True)
        if corruption=='noninteger_counter':payload['online_updates']=True
        else:payload['actor']['mean_head.bias'][0]=float('nan')
        torch.save(payload,checkpoint)
        write_checkpoint_metadata(checkpoint,ec,agent.config,stage='synthetic-unit-test')
    monkeypatch.setattr(collection,'run_episode',lambda *a,**k:pytest.fail('Invalid input reached physics'))
    with pytest.raises(ValueError,match=match):collection.collect(args)
    assert not Path(args.output).exists()


def test_real_failed_attempts_preserve_five_executions_and_terminal_joint_frames(tmp_path):
    args,_,_ = checkpoint_fixture(tmp_path)
    original_hash = sha256(Path(args.checkpoint))
    assert collection.collect(args) == 2
    output=Path(args.output);report=json.loads((output/'report.json').read_text())
    assert report['status']=='complete' and report['completed_episodes']==5
    assert report['successful_episodes']==0 and report['eligible_for_primitive'] is False
    assert not (output/'successful_joint_trajectories.npz').exists()
    assert report['learner_counters']['online_updates']==1
    assert sha256(Path(args.checkpoint))==original_hash==sha256(output/'checkpoint.pt')
    for episode,record in enumerate(report['episodes']):
        assert record['checkpoint_sha256']==original_hash
        assert not record['eligible_for_primitive']
        with np.load(output/record['evidence']['transitions']['path']) as data:
            assert data['obs'].shape==(1,60) and data['actions'].shape==(1,16)
            assert data['joint_positions'].shape==(2,16) and data['control_times'].shape==(2,)
            np.testing.assert_array_equal(data['joint_positions'][0],data['initial_qpos'][:16])
            np.testing.assert_array_equal(data['joint_positions'][-1],data['qpos'][-1,:16])
            assert data['control_times'][-1]==pytest.approx(.1)
            assert data['terminated'][-1] or data['truncated'][-1]
        with np.load(output/record['evidence']['physics']['path']) as data:
            assert data['samples'].shape==(50,len(collection.PHYSICS_COLUMNS))
        for item in record['evidence'].values():assert sha256(output/item['path'])==item['sha256']


def synthetic_success_bundle(tmp_path):
    """Exercise serialization/validation only; every success here is synthetic."""
    args,ec,_=checkpoint_fixture(tmp_path,horizon=2)
    output=Path(args.output);output.mkdir()
    shutil.copyfile(args.checkpoint,output/'checkpoint.pt')
    shutil.copyfile(metadata_path(args.checkpoint),output/'checkpoint.pt.metadata.json')
    counters={'bc_updates':0,'pretrain_updates':0,'online_updates':1}
    contract=environment_contract(ec);identity=sha256(output/'checkpoint.pt')
    model=make_model(ec);data=mujoco.MjData(model)
    q_ids=np.array([model.joint(name).qposadr[0] for name in JOINT_NAMES])
    state_spec=mujoco.mjtState.mjSTATE_INTEGRATION
    records,frames,times=[],[],[]
    for episode in range(5):
        directory=output/f'episode_{episode:03d}';directory.mkdir()
        measured=np.arange(48,dtype=float).reshape(3,16)/100
        obs=np.arange(120,dtype=np.float32).reshape(2,60)/1000
        next_obs=np.vstack((obs[1],obs[1]+.01));control_times=np.array([0.,.1,.2])
        obs[:,:16]=measured[:-1];next_obs[:,:16]=measured[1:]
        # Serialize internally consistent MjData; no simulation or physical-success
        # claim is made for these deliberately synthetic schema-only states.
        qpos=[];qvel=[];targets=[];states=[]
        for index in range(3):
            data.qpos[q_ids]=measured[index]
            data.time=control_times[index]
            data.ctrl[:]=model.actuator_ctrlrange.mean(axis=1)
            state=np.empty(mujoco.mj_stateSize(model,state_spec))
            mujoco.mj_getState(model,data,state,state_spec)
            qpos.append(data.qpos.copy());qvel.append(data.qvel.copy())
            targets.append(data.ctrl.copy());states.append(state)
        np.savez_compressed(directory/'transitions.npz',obs=obs,next_obs=next_obs,actions=np.zeros((2,16)),
            rewards=np.ones(2),terminated=[False,True],truncated=[False,False],joint_positions=measured,
            control_times=control_times,qpos=np.array(qpos[1:]),qvel=np.array(qvel[1:]),
            targets=np.array(targets[1:]),integration_state=np.array(states[1:]),
            initial_qpos=qpos[0],initial_qvel=qvel[0],initial_ctrl=targets[0],
            initial_integration_state=states[0],initial_time=control_times[0])
        physics=np.zeros((100,len(collection.PHYSICS_COLUMNS)))
        col=collection.PHYSICS_COLUMNS.index
        physics[:,col('time')]=np.arange(1,101)*.002
        physics[:,col('twist_increment')]=.07
        angles=np.cumsum(physics[:,col('twist_increment')])
        for field in ('raw_angle','qualified_angle','force_qualified_angle'):physics[:,col(field)]=angles
        for field in ('finger_contacts','force_bearing_fingers'):physics[:,col(field)]=2
        for field in ('retained','force_retained'):physics[:,col(field)]=1
        np.savez_compressed(directory/'physics.npz',samples=physics,columns=np.asarray(collection.PHYSICS_COLUMNS))
        result=dict(seed=episode,steps=2,environment_success=True,finger_supported_rotation_success=True,
            independent_rotation_success=True,independent_drop=False,environment_drop=False,
            independent_palm_support_violation=False,no_external_support=True,terminated=True,truncated=False,
            environment_failure_reason=None,model_equality_constraints=0,final_force_qualified_angle=float(angles[-1]),
            final_force_bearing_fingers=2,final_finger_contacts=2,final_raw_angle=float(angles[-1]),
            final_qualified_angle=float(angles[-1]),final_axis_tilt_radians=0.,initial_palm_normal_force=0.,
            peak_support_normal_forces={'palm':0.,'table':0.,'other':0.},initial_physics_sha256='0'*64,
            policy_interface={'kind':'absolute_joint_target'}, environment_contract=contract,
            checkpoint_sha256=identity,learner_counters=counters,online_updates=counters['online_updates'],
            policy_only=True,
            synthetic_unit_test_fixture=True,not_physical_evidence=True)
        (directory/'audit.json').write_text(json.dumps(result))
        record,q,t=collection.episode_record(output,episode,result,identity,counters,contract)
        records.append(record);frames.append(q);times.append(t.tolist())
    metadata=dict(kind=collection.COLLECTION_KIND,schema_version=1,eligible_for_primitive=False,
        source='SYNTHETIC UNIT TEST RECORDS ONLY; NOT PHYSICAL SUCCESS EVIDENCE',synthetic_unit_test_fixture=True,
        task='rotation',environment=asdict(ec),environment_contract=contract,policy_only=True,domain_randomization=False,
        policy_interface={'kind':'absolute_joint_target'},learner_counters=counters,
        checkpoint={'path':'checkpoint.pt','sha256':identity,'metadata_path':'checkpoint.pt.metadata.json',
                    'metadata_sha256':sha256(output/'checkpoint.pt.metadata.json')},
        alignment={'method':'reset_relative_original_samples','interpolation':False},
        episode_timestamps=times,episodes=records,successful_episodes=5,rotation_validation=ROTATION_VALIDATION)
    return output,metadata,frames


def test_successful_assembly_preserves_original_frames_without_interpolation(tmp_path):
    output,metadata,frames=synthetic_success_bundle(tmp_path)
    assert collection.export_if_eligible(output,metadata,frames)
    with np.load(output/'successful_joint_trajectories.npz') as result:
        np.testing.assert_array_equal(result['joint_positions'],np.concatenate(frames))
        stored=json.loads(str(result['metadata']))
    assert stored['synthetic_unit_test_fixture'] and stored['eligible_for_primitive']
    assert stored['alignment']['interpolation'] is False
    assert len(stored['episodes'])==5 and stored['learner_counters']['online_updates']==1


@pytest.mark.parametrize('corruption',['one_failed','only_four','mixed_checkpoint','changed_evidence','invented_times',
                                      'unsupported_force','retention_flag','table','displacement','contact_loss',
                                      'audit_identity','audit_counters','audit_interface','audit_contract','audit_policy',
                                      'joint_state_disagreement','contradictory_integration','missing_initial_state'])
def test_assembly_rejects_missing_or_contradictory_execution_evidence(tmp_path,corruption):
    output,metadata,frames=synthetic_success_bundle(tmp_path)
    record=metadata['episodes'][0]
    if corruption=='one_failed':record['eligible_for_primitive']=False
    elif corruption=='only_four':metadata['episodes'].pop();frames.pop()
    elif corruption=='mixed_checkpoint':record['checkpoint_sha256']='f'*64
    elif corruption=='changed_evidence':
        with (output/record['evidence']['physics']['path']).open('ab') as stream:stream.write(b'changed')
    elif corruption.startswith('audit_'):
        path=output/record['evidence']['audit']['path']
        audit=json.loads(path.read_text())
        field,value={'audit_identity':('checkpoint_sha256','f'*64),
            'audit_counters':('learner_counters',{'bc_updates':0,'pretrain_updates':0,'online_updates':2}),
            'audit_interface':('policy_interface',{'kind':'bootstrap_joint_increment'}),
            'audit_contract':('environment_contract',{}),'audit_policy':('policy_only',False)}[corruption]
        audit[field]=value
        path.write_text(json.dumps(audit));record['evidence']['audit']['sha256']=sha256(path)
    else:
        state_corruptions=('invented_times','joint_state_disagreement','contradictory_integration','missing_initial_state')
        name='transitions' if corruption in state_corruptions else 'physics'
        path=output/record['evidence'][name]['path']
        with np.load(path) as archive:data={k:v.copy() for k,v in archive.items()}
        if corruption=='invented_times':
            data['control_times'][-1]=.25
            record['control_times']=data['control_times'].tolist()
            record['duration_seconds']=.25
            metadata['episode_timestamps'][0]=record['control_times']
        elif corruption=='joint_state_disagreement':data['qpos'][0,0]+=.1
        elif corruption=='contradictory_integration':data['qvel'][0,0]+=.1
        elif corruption=='missing_initial_state':del data['initial_integration_state']
        else:
            column,value={'unsupported_force':('palm_normal_force',.1),'retention_flag':('force_retained',0.),
                'table':('table_contacts',1.),'displacement':('object_displacement',.181),
                'contact_loss':('contactless_seconds',.05)}[corruption]
            data['samples'][4,collection.PHYSICS_COLUMNS.index(column)]=value
        np.savez_compressed(path,**data);record['evidence'][name]['sha256']=sha256(path)
    if corruption in ('one_failed','only_four'):
        assert collection.export_if_eligible(output,metadata,frames) is False
    else:
        with pytest.raises(ValueError):collection.export_if_eligible(output,metadata,frames)
    assert not (output/'successful_joint_trajectories.npz').exists()


def test_synthetic_schema_only_collection_extracts_and_rechecks_primitive_lineage(tmp_path):
    """End-to-end contract fixture; none of these success records is physical."""
    output,metadata,frames=synthetic_success_bundle(tmp_path)
    assert collection.export_if_eligible(output,metadata,frames)
    path=tmp_path/'synthetic_fixture_primitive.npz'
    cli.extract(SimpleNamespace(trajectory=str(output/'successful_joint_trajectories.npz'),output=str(path)))
    primitive=PrimitiveBasis.load(path)
    checked=validate_primitive_source(primitive)
    assert checked['synthetic_unit_test_fixture']
    assert checked['source'].startswith('SYNTHETIC UNIT TEST')
    changed_basis=replace(primitive,basis=-primitive.basis)
    with pytest.raises(ValueError,match='differs from the 90-percent PCA'):
        validate_primitive_source(changed_basis)
    evidence=output/metadata['episodes'][0]['evidence']['physics']['path']
    with evidence.open('ab') as stream:stream.write(b'changed after extraction')
    with pytest.raises(ValueError,match='evidence is missing, changed'):
        validate_primitive_source(primitive)
