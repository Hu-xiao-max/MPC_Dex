"""Atomic, exact restart state for the main MPC-scaffolded online learner.

Only ``latest.json`` commits a generation. It verifies every file hash before
deserialization. Legacy standalone policy checkpoints are not resume states.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import signal
import tempfile
import sys

import mujoco
import numpy as np
import torch

from .contracts import canonical, environment_contract, metadata_path, sha256, validate_checkpoint, write_checkpoint_metadata
from .learning import SACAgent
from .replay import ReplayBuffer


ENV_ARRAYS=('initial_position','initial_orientation','last_action','last_q','last_p','_last_object_quaternion')
ENV_SCALARS=('start_angle','_last_wrapped','angle','previous_angle','steps','rotations','dwell','goal',
    'heading_error','_episode_over','_contactless_seconds','_goal_rewarded','retained_angle',
    'previous_retained_angle','retained_positive_step','retained_negative_step',
    '_palm_support_violation','_peak_palm_support_force')


def atomic_json(path, data):
    path=Path(path)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w') as stream:
        json.dump(data,stream,indent=2,allow_nan=False);stream.write('\n')
        stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def _sync_directory(path):
    descriptor=os.open(path,os.O_RDONLY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def read_committed(run):
    run=Path(run)
    path=run/'latest.json'
    if not path.is_file():
        raise ValueError('Resume requires a committed main-training latest.json; policy-only checkpoints cannot resume')
    manifest=json.loads(path.read_text())
    name=manifest.get('checkpoint_directory','')
    if (manifest.get('schema_version')!=1 or not isinstance(name,str)
            or not name.startswith('checkpoint_') or Path(name).name!=name):
        raise ValueError('Invalid main-training checkpoint manifest')
    directory=run/name
    hashes=manifest.get('files',{})
    required={'agent.pt','agent.pt.metadata.json','online_replay.npz','offline_replay.npz',
              'environment.bin','environment.npz','planner.npz','torch_rng.pt','state.json'}
    if not isinstance(hashes,dict) or set(hashes)!=required:
        raise ValueError('Checkpoint manifest is incomplete')
    for name,digest in hashes.items():
        file=directory/name
        if not file.is_file() or sha256(file)!=digest:
            raise ValueError(f'Committed checkpoint hash mismatch: {name}')
    state=json.loads((directory/'state.json').read_text())
    if (state.get('schema_version')!=1 or state.get('command')!='train'
            or not state.get('resumable') or state.get('generation')!=manifest.get('generation')
            or state.get('environment_steps')!=manifest.get('environment_steps')):
        raise ValueError('Checkpoint state disagrees with its committed manifest')
    return directory,state


def identity(config, env_config, args, pretrained, planner_runtime):
    root=Path(__file__).resolve().parents[1]
    runtime={key:value for key,value in planner_runtime.items() if key!='initialized_at'}
    return canonical(dict(config=config,environment_contract=environment_contract(env_config),
        seed=args.seed,utd=args.utd,batch_size=args.batch_size,device=str(torch.device(args.device)),
        torch_threads=torch.get_num_threads(),
        runtime=dict(python=sys.version,platform=platform.platform(),machine=platform.machine(),
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            float32_matmul_precision=torch.get_float32_matmul_precision(),
            cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic),
        pretrained_checkpoint=str(Path(pretrained).resolve()),
        pretrained_sha256=sha256(pretrained),pretrained_metadata_sha256=sha256(metadata_path(pretrained)),
        offline_replay_sha256=sha256(args.replay),primitive_sha256=sha256(args.primitive),
        checkpoint_implementation_sha256=sha256(root/'mpc_dex/online_checkpoint.py'),
        planner_runtime=runtime))


def _tuple(value):
    return tuple(_tuple(item) for item in value) if isinstance(value,list) else value


def reconcile_metrics(run,state):
    """Retain crash-tail logs separately; the main log reflects committed work."""
    run=Path(run);path=run/'metrics.jsonl'
    content=path.read_bytes() if path.exists() else b''
    size=state['metrics_bytes']
    if size>len(content) or hashlib.sha256(content[:size]).hexdigest()!=state['metrics_sha256']:
        raise ValueError('Committed metrics prefix is missing or changed')
    if len(content)>size:
        suffix=run/f"metrics.uncommitted.{state['generation']:08d}.jsonl"
        index=0
        while suffix.exists():
            index+=1;suffix=run/f"metrics.uncommitted.{state['generation']:08d}.{index}.jsonl"
        suffix.write_bytes(content[size:])
        with path.open('r+b') as stream:
            stream.truncate(size);stream.flush();os.fsync(stream.fileno())


def restore(directory, state, expected_identity, env, planner, scheduler, device):
    """Restore objects first and random generators last; never run extra physics."""
    directory=Path(directory)
    if state.get('resume_identity')!=expected_identity:
        raise ValueError('Resume configuration, data, primitive, numerical source or runtime differs')
    record=validate_checkpoint(directory/'agent.pt',env.config)
    if record.get('stage')!='online' or record.get('offline_replay_sha256')!=expected_identity['offline_replay_sha256']:
        raise ValueError('Committed learner is not bound to the fixed online-training replay')
    agent=SACAgent.load(directory/'agent.pt',device)
    if (agent.online_updates!=state['environment_steps']*expected_identity['utd']
            or agent.online_updates!=state['online_updates']):
        raise ValueError('Completed transition and optimizer-update counts disagree')
    online=ReplayBuffer.load(directory/'online_replay.npz')
    offline=ReplayBuffer.load(directory/'offline_replay.npz')
    # MuJoCo's typed binary pickle state contains the full live mjData, including
    # derived contacts/solver state. No Python pickle program is deserialized.
    data=mujoco.MjData.__new__(mujoco.MjData)
    data.__setstate__((directory/'environment.bin').read_bytes())
    if data.model.__getstate__()!=env.model.__getstate__():
        raise ValueError('Serialized simulator model differs from the requested model')
    mujoco.mj_copyData(env.data,env.model,data)
    with np.load(directory/'environment.npz',allow_pickle=False) as archive:
        for name in ENV_ARRAYS:setattr(env,name,archive[name].copy())
        for name,value in state['environment_scalars'].items():setattr(env,name,value)
        observation=archive['observation'].copy()
    env.np_random.bit_generator.state=state['environment_rng']
    with np.load(directory/'planner.npz',allow_pickle=False) as archive:
        for name in ('nominal_knots','std_knots'):
            value=archive[name]
            if value.shape!=getattr(planner,name).shape or not np.isfinite(value).all():
                raise ValueError('Invalid planner continuation arrays')
            setattr(planner,name,value.copy())
        planner.last_sequence=archive['last_sequence'].copy() if state['planner_has_last_sequence'] else None
    planner.rng.bit_generator.state=state['planner_rng']
    scheduler.load_state_dict(state['scheduler'])
    expected_last=state['environment_steps']-1
    if scheduler.last_step!=expected_last:
        raise ValueError('Scheduler and completed transition counters disagree')
    numpy_rng=state['numpy_rng']
    np.random.set_state((numpy_rng[0],np.asarray(numpy_rng[1],dtype=np.uint32),*numpy_rng[2:]))
    random.setstate(_tuple(state['python_rng']))
    rng=torch.load(directory/'torch_rng.pt',map_location='cpu',weights_only=True)
    torch.set_rng_state(rng['cpu'])
    if agent.device.type=='cuda':torch.cuda.set_rng_state_all(rng['cuda'])
    return agent,online,offline,observation


def commit(run, state, agent, online, offline, env, observation, planner, scheduler):
    """Publish only completed transitions and all their updates as one unit."""
    run=Path(run)
    generation=state.get('generation',0)+1
    while (run/f'checkpoint_{generation:08d}').exists():generation+=1
    name=f'checkpoint_{generation:08d}'
    temporary=Path(tempfile.mkdtemp(prefix=name+'.pending-',dir=run))
    destination=run/name
    snapshot=dict(state,generation=generation,schema_version=1,resumable=True,
        online_updates=agent.online_updates,
        environment_scalars={key:getattr(env,key) for key in ENV_SCALARS},
        environment_rng=env.np_random.bit_generator.state,
        planner_rng=planner.rng.bit_generator.state,
        planner_has_last_sequence=planner.last_sequence is not None,
        scheduler=scheduler.state_dict(),python_rng=random.getstate())
    metrics=run/'metrics.jsonl'
    content=metrics.read_bytes() if metrics.exists() else b''
    snapshot.update(metrics_bytes=len(content),metrics_sha256=hashlib.sha256(content).hexdigest())
    if metrics.exists():
        with metrics.open('rb') as stream:os.fsync(stream.fileno())
    numpy_rng=np.random.get_state()
    snapshot['numpy_rng']=[numpy_rng[0],numpy_rng[1].tolist(),*numpy_rng[2:]]
    try:
        agent.save(temporary/'agent.pt')
        write_checkpoint_metadata(temporary/'agent.pt',env.config,agent.config,stage='online',
            offline_replay_sha256=snapshot['resume_identity']['offline_replay_sha256'])
        online.save(temporary/'online_replay.npz');offline.save(temporary/'offline_replay.npz')
        (temporary/'environment.bin').write_bytes(env.data.__getstate__())
        np.savez_compressed(temporary/'environment.npz',observation=observation,
                            **{key:getattr(env,key) for key in ENV_ARRAYS})
        np.savez_compressed(temporary/'planner.npz',nominal_knots=planner.nominal_knots,
            std_knots=planner.std_knots,last_sequence=(planner.last_sequence if planner.last_sequence is not None else np.empty((0,16))))
        rng={'cpu':torch.get_rng_state()}
        if agent.device.type=='cuda':rng['cuda']=torch.cuda.get_rng_state_all()
        torch.save(rng,temporary/'torch_rng.pt')
        atomic_json(temporary/'state.json',snapshot)
        for file in temporary.iterdir():
            with file.open('rb') as stream:os.fsync(stream.fileno())
        hashes={file.name:sha256(file) for file in temporary.iterdir()}
        _sync_directory(temporary)
        temporary.rename(destination);_sync_directory(run)
        manifest=dict(schema_version=1,checkpoint_directory=name,generation=generation,
                      environment_steps=snapshot['environment_steps'],files=hashes)
        atomic_json(run/'latest.json',manifest);_sync_directory(run)
        # From here the new generation is authoritative even if convenience
        # exports fail. They are never used by the resume loader.
        state.update(snapshot)
        atomic_json(run/'run.json',snapshot)
        for source,target in [(destination/'agent.pt',run/'checkpoint.pt'),
                              (destination/'agent.pt.metadata.json',run/'checkpoint.pt.metadata.json')]:
            intermediate=target.with_name(target.name+'.tmp')
            shutil.copyfile(source,intermediate);intermediate.replace(target)
        if snapshot['status']=='complete' and online._pending is not None:
            exported=ReplayBuffer.load(destination/'online_replay.npz')
            exported._arrays['truncated'][exported._pending]=True
            exported._finish_episode()
            exported.save(run/'online_replay.npz')
        else:
            intermediate=run/'online_replay.npz.tmp'
            shutil.copyfile(destination/'online_replay.npz',intermediate)
            intermediate.replace(run/'online_replay.npz')
        for obsolete in sorted(run.glob('checkpoint_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]'))[:-2]:
            shutil.rmtree(obsolete)
    finally:
        if temporary.exists():shutil.rmtree(temporary)


class StopAtBoundary:
    """SIGINT/SIGTERM request a checkpoint after the current complete update set."""
    def __init__(self):self.requested=False;self.signal_number=None;self.previous={}
    def _request(self,number,frame):self.requested=True;self.signal_number=number
    def __enter__(self):
        for number in (signal.SIGINT,signal.SIGTERM):
            self.previous[number]=signal.signal(number,self._request)
        return self
    def __exit__(self,*exc):
        for number,handler in self.previous.items():signal.signal(number,handler)


class RunLock:
    """One Linux/Unix process may advance a given run at a time.

    The OS releases this advisory lock on exit, including forced termination;
    a leftover lock file never implies that a process is still running.
    """
    def __init__(self,path,*,create):self.path=Path(path);self.create=create;self.stream=None
    def __enter__(self):
        import fcntl
        if self.create:self.path.mkdir(parents=True,exist_ok=True)
        if not self.path.is_dir():raise ValueError('Resume run directory does not exist')
        self.stream=(self.path/'.online-training.lock').open('a+')
        try:fcntl.flock(self.stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            self.stream.close();self.stream=None
            raise ValueError('Another process is advancing this training run') from None
        self.stream.seek(0);self.stream.truncate();self.stream.write(str(os.getpid())+'\n');self.stream.flush()
        return self
    def __exit__(self,*exc):
        if self.stream is not None:self.stream.close()
