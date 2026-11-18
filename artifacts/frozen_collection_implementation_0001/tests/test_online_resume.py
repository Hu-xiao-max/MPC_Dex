"""Main-loop restart tests with real physics and small, real optimizer updates.

The tiny pretrained fixture has synthetic schedule counters solely to exercise
the existing interface gate. It is never exported as a scientific result or a
successful demonstration, and the cheap planner is only a test controller.
"""
from dataclasses import asdict
import json
from pathlib import Path
import random
import signal
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from mpc_dex import cli, online_checkpoint
from mpc_dex.contracts import sha256, validate_checkpoint, write_checkpoint_metadata
from mpc_dex.env import AllegroRotationEnv, EnvConfig
from mpc_dex.learning import LearnerConfig, SACAgent
from mpc_dex.primitives import PrimitiveBasis
from mpc_dex.replay import ReplayBuffer


GRASP=str(Path(__file__).resolve().parents[1]/'assets/initial_grasp.npz')


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    # Resume mechanics use an explicitly synthetic primitive, not task evidence.
    monkeypatch.setattr(cli,'validate_primitive_source',lambda primitive: None)
    previous=torch.get_num_threads();torch.set_num_threads(1)
    # Short true time-limit episodes exercise resets and random goal draws.
    ec=EnvConfig(initial_grasp=GRASP,horizon_steps=3)
    env=AllegroRotationEnv(ec)
    offline=ReplayBuffer(capacity=128,seed=31)
    for episode in range(20):
        obs,_=env.reset(seed=episode)
        action=env.to_action(env.data.ctrl)
        next_obs,reward,terminated,_,info=env.step(action)
        assert not info['success']
        # An actual held-control transition, cut for this interface fixture.
        offline.add_transition(obs,action,reward,next_obs,terminated,not terminated)
    replay=tmp_path/'interface_replay.npz';offline.save(replay)
    basis=PrimitiveBasis.from_trajectories(
        env.data.ctrl+np.random.default_rng(11).normal(0,.01,(24,16)),
        provenance={'source':'resume interface test only','successful_policy':False})
    primitive=tmp_path/'interface_primitive.npz';basis.save(primitive)
    learner=LearnerConfig(hidden_sizes=(8,8),ensemble_size=2)
    torch.manual_seed(11)
    agent=SACAgent(60,16,learner)
    agent.initialize_std()
    agent.bc_updates=12000;agent.pretrain_updates=50000
    pretrained=tmp_path/'interface_pretrained.pt';agent.save(pretrained)
    write_checkpoint_metadata(pretrained,ec,learner,stage='unit-test-interface-only',
                              offline_replay_sha256=sha256(replay))
    config={'environment':asdict(ec),'learner':asdict(learner),
            'planner':{'horizon_seconds':.1,'timestep_seconds':.1,'knots':2,
                       'samples':4,'elites':2,'iterations':1},
            'planner_cost':{'workers':1}}
    configuration=tmp_path/'config.json';configuration.write_text(json.dumps(config))
    env.close()
    class SmallReplay(ReplayBuffer):
        def __init__(self,capacity=128,seed=0):super().__init__(capacity,seed)
    monkeypatch.setattr(cli,'ReplayBuffer',SmallReplay)
    def cheap_plan(planner,evaluator):
        # Depend on and mutate all CEM continuation fields and global RNGs.
        offset=planner.rng.normal(0,1e-5,16)
        offset+=(np.random.random()+random.random())*1e-6
        offset+=planner.nominal_knots[0,-16:]*1e-5
        target=np.clip(evaluator.env.data.ctrl+offset,planner.action_low,planner.action_high)
        planner.nominal_knots+=planner.rng.normal(0,1e-5,planner.nominal_knots.shape)
        planner.std_knots+=np.abs(planner.rng.normal(0,1e-5,planner.std_knots.shape))
        planner.last_sequence=np.broadcast_to(target,(planner.config.horizon_steps,16)).copy()
        return target,{'test_controller':True}
    monkeypatch.setattr(cli.PrimitiveCEM,'plan',cheap_plan)
    yield SimpleNamespace(config=str(configuration),checkpoint=str(pretrained),replay=str(replay),
                          primitive=str(primitive),device='cpu',seed=0,steps=7,utd=2,
                          batch_size=4,checkpoint_every=2,resume=None)
    torch.set_num_threads(previous)


def run(args,output,**changes):
    values=dict(vars(args),output=str(output));values.update(changes)
    torch.manual_seed(123);np.random.seed(123);random.seed(123)
    return cli.train(SimpleNamespace(**values))


def equal_nested(left,right):
    assert type(left) is type(right)
    if isinstance(left,torch.Tensor):assert torch.equal(left,right)
    elif isinstance(left,np.ndarray):np.testing.assert_array_equal(left,right)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left:equal_nested(left[key],right[key])
    elif isinstance(left,(list,tuple)):
        assert len(left)==len(right)
        for a,b in zip(left,right):equal_nested(a,b)
    else:assert left==right


def assert_same_continuation(first,second):
    directory_a,state_a=online_checkpoint.read_committed(first)
    directory_b,state_b=online_checkpoint.read_committed(second)
    for file in ('agent.pt','torch_rng.pt'):
        equal_nested(torch.load(directory_a/file,weights_only=True),
                     torch.load(directory_b/file,weights_only=True))
    for file in ('environment.npz','planner.npz','online_replay.npz','offline_replay.npz'):
        with np.load(directory_a/file,allow_pickle=False) as a,np.load(directory_b/file,allow_pickle=False) as b:
            assert set(a.files)==set(b.files)
            for key in a.files:np.testing.assert_array_equal(a[key],b[key],err_msg=f'{file}: {key}')
    data=[]
    for directory in (directory_a,directory_b):
        value=mujoco.MjData.__new__(mujoco.MjData)
        value.__setstate__((directory/'environment.bin').read_bytes());data.append(value)
    # Binary mjData also contains wall-clock profiling timers and unused sparse
    # scratch memory. Compare every integration input and active physical output.
    spec=mujoco.mjtState.mjSTATE_INTEGRATION
    states=[]
    for value in data:
        state=np.empty(mujoco.mj_stateSize(value.model,spec))
        mujoco.mj_getState(value.model,value,state,spec);states.append(state)
    np.testing.assert_array_equal(*states)
    for name in ('qacc','qfrc_constraint','efc_force','xpos','xquat','actuator_force'):
        np.testing.assert_array_equal(getattr(data[0],name),getattr(data[1],name),err_msg=name)
    for name in ('geom','pos','frame','dist','efc_address'):
        np.testing.assert_array_equal(getattr(data[0].contact,name),getattr(data[1].contact,name))
    for key in ('environment_steps','online_updates','drops','successful_training_episodes',
                'episode_return','latest_metrics','environment_scalars','environment_rng',
                'planner_rng','planner_has_last_sequence','scheduler','python_rng','numpy_rng'):
        assert state_a[key]==state_b[key],key


@pytest.mark.parametrize('seed',[0,2])
def test_signal_mid_update_resumes_exactly_with_real_environment(inputs,tmp_path,monkeypatch,seed):
    baseline=tmp_path/'uninterrupted';interrupted=tmp_path/'interrupted'
    assert run(inputs,baseline,seed=seed)==0
    original=SACAgent.update
    def interrupt_mid_transition(agent,batch):
        result=original(agent,batch)
        if agent.online_updates==3:signal.raise_signal(signal.SIGINT)
        return result
    monkeypatch.setattr(SACAgent,'update',interrupt_mid_transition)
    assert run(inputs,interrupted,seed=seed)==130
    directory,state=online_checkpoint.read_committed(interrupted)
    assert state['environment_steps']==2 and state['online_updates']==4
    assert state['status']=='interrupted'
    assert state['scheduler']['controller']==('mpc' if seed==0 else 'policy')
    original_restore=online_checkpoint.restore
    def check_exact_restoration(directory,state,identity,env,*args):
        result=original_restore(directory,state,identity,env,*args)
        assert env.data.__getstate__()==(directory/'environment.bin').read_bytes()
        return result
    monkeypatch.setattr(online_checkpoint,'restore',check_exact_restoration)
    assert run(inputs,interrupted,resume=str(interrupted),checkpoint=None,seed=seed)==0
    assert_same_continuation(baseline,interrupted)


def test_budget_extension_keeps_internal_episode_open(inputs,tmp_path):
    baseline=tmp_path/'uninterrupted';extended=tmp_path/'extended'
    assert run(inputs,baseline)==0
    assert run(inputs,extended,steps=2)==0
    directory,state=online_checkpoint.read_committed(extended)
    internal=ReplayBuffer.load(directory/'online_replay.npz')
    public=ReplayBuffer.load(extended/'online_replay.npz')
    assert internal._pending is not None and internal.episode_count==0
    assert public._pending is None and public.episode_count==1
    assert not internal.transitions()['truncated'][-1] and public.transitions()['truncated'][-1]
    assert run(inputs,extended,resume=str(extended))==0
    assert_same_continuation(baseline,extended)


def test_failed_update_does_not_publish_partial_transition(inputs,tmp_path,monkeypatch):
    baseline=tmp_path/'uninterrupted';failed=tmp_path/'failed'
    assert run(inputs,baseline)==0
    original=SACAgent.update
    def failing_update(agent,batch):
        result=original(agent,batch)
        if agent.online_updates==5:raise RuntimeError('injected mid-transition failure')
        return result
    monkeypatch.setattr(SACAgent,'update',failing_update)
    with pytest.raises(RuntimeError,match='mid-transition'):
        run(inputs,failed)
    _,state=online_checkpoint.read_committed(failed)
    assert state['environment_steps']==2 and state['online_updates']==4
    manifest=json.loads((failed/'run.json').read_text())
    assert manifest['uncommitted_work_discarded']
    monkeypatch.setattr(SACAgent,'update',original)
    assert run(inputs,failed,resume=str(failed))==0
    assert_same_continuation(baseline,failed)


@pytest.mark.parametrize('changed',[{'utd':3},{'batch_size':5},{'seed':1}])
def test_incompatible_resume_rejected_without_replacing_manifest(inputs,tmp_path,changed):
    output=tmp_path/'run';assert run(inputs,output,steps=2)==0
    before=(output/'latest.json').read_bytes();run_before=(output/'run.json').read_bytes()
    with pytest.raises(ValueError,match='configuration, data, primitive, numerical source or runtime differs'):
        run(inputs,output,resume=str(output),**changed)
    assert (output/'latest.json').read_bytes()==before
    assert (output/'run.json').read_bytes()==run_before


@pytest.mark.parametrize('file',['agent.pt','online_replay.npz','environment.bin','state.json'])
def test_corrupted_generation_rejected_before_deserialization(inputs,tmp_path,file):
    output=tmp_path/'run';assert run(inputs,output,steps=1)==0
    directory,_=online_checkpoint.read_committed(output)
    with (directory/file).open('ab') as stream:stream.write(b'corrupt')
    with pytest.raises(ValueError,match=f'hash mismatch: {file}'):
        run(inputs,output,resume=str(output))


def test_failure_before_atomic_manifest_publish_keeps_previous_generation(inputs,tmp_path,monkeypatch):
    output=tmp_path/'run'
    original=online_checkpoint.atomic_json
    def fail_publication(path,data):
        if Path(path).name=='latest.json' and data['environment_steps']==2:
            raise OSError('injected publication interruption')
        return original(path,data)
    monkeypatch.setattr(online_checkpoint,'atomic_json',fail_publication)
    with pytest.raises(OSError,match='publication interruption'):
        run(inputs,output,steps=2)
    _,state=online_checkpoint.read_committed(output)
    assert state['environment_steps']==state['online_updates']==0
    monkeypatch.setattr(online_checkpoint,'atomic_json',original)
    assert run(inputs,output,resume=str(output))==0
    baseline=tmp_path/'uninterrupted';assert run(inputs,baseline)==0
    assert_same_continuation(baseline,output)
    logs=[json.loads(line) for line in (output/'metrics.jsonl').read_text().splitlines()]
    assert [row['environment_step'] for row in logs if 'event' not in row]==[2,4,6,7]
    tails=list(output.glob('metrics.uncommitted.*.jsonl'))
    assert len(tails)==1 and json.loads(tails[0].read_text())['environment_step']==2


def test_signal_during_checkpoint_publication_records_interrupted_status(inputs,tmp_path,monkeypatch):
    original=online_checkpoint.atomic_json;sent=[]
    def interrupt_publication(path,data):
        result=original(path,data)
        if Path(path).name=='latest.json' and data['environment_steps']==2 and not sent:
            sent.append(True);signal.raise_signal(signal.SIGTERM)
        return result
    monkeypatch.setattr(online_checkpoint,'atomic_json',interrupt_publication)
    output=tmp_path/'interrupted_publish'
    assert run(inputs,output)==128+signal.SIGTERM
    _,state=online_checkpoint.read_committed(output)
    assert state['environment_steps']==2 and state['online_updates']==4
    assert state['status']=='interrupted'


@pytest.mark.parametrize('failure',['run_manifest','checkpoint_metadata','public_replay'])
def test_no_work_resume_repairs_exports_after_committed_publication_failure(
        inputs,tmp_path,monkeypatch,failure):
    output=tmp_path/'export_failure'
    with monkeypatch.context() as fault:
        if failure=='run_manifest':
            original=online_checkpoint.atomic_json
            def fail_export(path,data):
                if Path(path)==output/'run.json' and data['environment_steps']==2:
                    raise OSError('injected post-commit export failure')
                return original(path,data)
            fault.setattr(online_checkpoint,'atomic_json',fail_export)
        elif failure=='checkpoint_metadata':
            original=online_checkpoint.shutil.copyfile
            def fail_export(source,target):
                if (Path(target)==output/'checkpoint.pt.metadata.json.tmp'
                        and json.loads((output/'latest.json').read_text())['environment_steps']==2):
                    raise OSError('injected post-commit export failure')
                return original(source,target)
            fault.setattr(online_checkpoint.shutil,'copyfile',fail_export)
        else:
            original=ReplayBuffer.save
            def fail_export(buffer,path):
                if Path(path)==output/'online_replay.npz' and len(buffer)==2:
                    raise OSError('injected post-commit export failure')
                return original(buffer,path)
            fault.setattr(ReplayBuffer,'save',fail_export)
        with pytest.raises(OSError,match='post-commit export failure'):
            run(inputs,output,steps=2)
    before_directory,before=online_checkpoint.read_committed(output)
    assert before['status']=='complete' and before['environment_steps']==2 and before['online_updates']==4
    assert json.loads((output/'run.json').read_text())['status']=='interrupted'
    assert len(ReplayBuffer.load(output/'online_replay.npz'))==0
    if failure=='run_manifest':
        assert SACAgent.load(output/'checkpoint.pt').online_updates==0
    elif failure=='checkpoint_metadata':
        _,ec=cli.read_config(inputs.config)
        with pytest.raises(ValueError,match='Checkpoint hash mismatch'):
            validate_checkpoint(output/'checkpoint.pt',ec)

    def no_advancement(*args,**kwargs):raise AssertionError('no-work resume advanced the process')
    monkeypatch.setattr(AllegroRotationEnv,'step',no_advancement)
    monkeypatch.setattr(SACAgent,'update',no_advancement)
    monkeypatch.setattr(cli.PrimitiveCEM,'plan',no_advancement)
    log_before=(output/'metrics.jsonl').read_bytes()
    assert run(inputs,output,resume=str(output),steps=2,checkpoint=None)==0
    after_directory,after=online_checkpoint.read_committed(output)
    assert after['generation']>before['generation']
    assert after['status']=='complete' and after['environment_steps']==2 and after['online_updates']==4
    assert json.loads((output/'run.json').read_text())==after
    assert (output/'metrics.jsonl').read_bytes()==log_before
    assert (before_directory/'environment.bin').read_bytes()==(after_directory/'environment.bin').read_bytes()
    for file in ('agent.pt','torch_rng.pt'):
        equal_nested(torch.load(before_directory/file,weights_only=True),
                     torch.load(after_directory/file,weights_only=True))
    equal_nested(torch.load(after_directory/'agent.pt',weights_only=True),
                 torch.load(output/'checkpoint.pt',weights_only=True))
    _,ec=cli.read_config(inputs.config)
    assert validate_checkpoint(output/'checkpoint.pt',ec)['stage']=='online'
    for file in ('environment.npz','planner.npz','online_replay.npz','offline_replay.npz'):
        with np.load(before_directory/file,allow_pickle=False) as a,np.load(after_directory/file,allow_pickle=False) as b:
            assert set(a.files)==set(b.files)
            for key in a.files:np.testing.assert_array_equal(a[key],b[key])
    for key in ('environment_scalars','environment_rng','planner_rng','planner_has_last_sequence',
                'scheduler','python_rng','numpy_rng','episode_return','latest_metrics'):
        assert before[key]==after[key],key
    internal=ReplayBuffer.load(after_directory/'online_replay.npz')
    public=ReplayBuffer.load(output/'online_replay.npz')
    assert len(internal)==len(public)==2
    assert internal._pending is not None and public._pending is None
    assert not internal.transitions()['truncated'][-1] and public.transitions()['truncated'][-1]


def test_run_lock_rejects_concurrent_writer_and_releases_after_failure(tmp_path):
    with online_checkpoint.RunLock(tmp_path,create=True):
        with pytest.raises(ValueError,match='Another process'):
            with online_checkpoint.RunLock(tmp_path,create=False):pass
    # A leftover lock file is harmless once the OS lock is released.
    with online_checkpoint.RunLock(tmp_path,create=False):pass


@pytest.mark.parametrize('changed',['primitive','replay','config','source'])
def test_resume_rejects_changed_input_or_source(inputs,tmp_path,monkeypatch,changed):
    output=tmp_path/'run';assert run(inputs,output,steps=1)==0
    before=(output/'latest.json').read_bytes();run_before=(output/'run.json').read_bytes()
    if changed in {'primitive','replay'}:
        with Path(getattr(inputs,changed)).open('ab') as stream:stream.write(b'changed')
    elif changed=='config':
        path=Path(inputs.config);config=json.loads(path.read_text())
        config['planner']['samples']=6;path.write_text(json.dumps(config))
    else:
        original=online_checkpoint.sha256
        def changed_source(path):
            return '0'*64 if Path(path).name=='online_checkpoint.py' else original(path)
        monkeypatch.setattr(online_checkpoint,'sha256',changed_source)
    with pytest.raises(ValueError,match='differs'):
        run(inputs,output,resume=str(output))
    assert (output/'latest.json').read_bytes()==before
    assert (output/'run.json').read_bytes()==run_before


def test_full_method_pretraining_gate_is_preserved(inputs,tmp_path):
    agent=SACAgent.load(inputs.checkpoint);agent.pretrain_updates=49999
    agent.save(inputs.checkpoint)
    _,ec=cli.read_config(inputs.config)
    write_checkpoint_metadata(inputs.checkpoint,ec,agent.config,stage='unit-test-interface-only',
                              offline_replay_sha256=sha256(inputs.replay))
    with pytest.raises(ValueError,match='completed BC and critic pretraining'):
        run(inputs,tmp_path/'refused')
    assert not (tmp_path/'refused'/'latest.json').exists()


def test_main_cli_fresh_and_resume_arguments(inputs,tmp_path,monkeypatch):
    output=tmp_path/'cli_smoke'
    common=['mpc_dex','train','--config',inputs.config,'--output',str(output),
            '--primitive',inputs.primitive,'--replay',inputs.replay,'--device','cpu',
            '--batch-size','4','--utd','1','--checkpoint-every','1']
    monkeypatch.setattr(sys,'argv',common+['--checkpoint',inputs.checkpoint,'--steps','1'])
    assert cli.main()==0
    monkeypatch.setattr(sys,'argv',common+['--resume',str(output),'--steps','2'])
    assert cli.main()==0
    _,state=online_checkpoint.read_committed(output)
    assert state['environment_steps']==state['online_updates']==2


def test_policy_only_checkpoint_is_not_resumable(tmp_path):
    with pytest.raises(ValueError,match='policy-only checkpoints cannot resume'):
        online_checkpoint.read_committed(tmp_path)
