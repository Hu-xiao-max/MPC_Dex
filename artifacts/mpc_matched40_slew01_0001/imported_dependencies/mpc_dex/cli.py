"""Commands for the independent local simulation deployment."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .env import AllegroRotationEnv, EnvConfig, ROTATION_VALIDATION
from .learning import SACAgent, LearnerConfig
from .mpc import PrimitiveCEM, CEMConfig, NoFeasiblePlan
from .primitives import PrimitiveBasis
from .replay import ReplayBuffer, sample_mixed
from .rollouts import MujocoRolloutEvaluator, RolloutCostConfig
from .scheduler import ChunkScheduler
from .contracts import environment_contract, write_checkpoint_metadata, validate_checkpoint
from .bootstrap_env import environment_for_policy

ROOT=Path(__file__).resolve().parents[1]


def read_config(path):
    config=json.loads(Path(path).read_text()) if path else {}
    env_values=config.get('environment',{}).copy()
    if env_values.get('initial_grasp'):
        p=Path(env_values['initial_grasp'])
        env_values['initial_grasp']=str(p if p.is_absolute() else ROOT/p)
    return config,EnvConfig(**env_values)


def write_json(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def hash_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()


def canonical(data):
    return json.loads(json.dumps(data,sort_keys=True,allow_nan=False))


def new_run(path, command, config, inputs=None):
    path=Path(path)
    if (path/'run.json').exists():raise ValueError(f'Run already exists: {path}; select a new output directory')
    path.mkdir(parents=True,exist_ok=True)
    state={'command':command,'implementation':'independent-paper-reproduction','target':'local-simulation',
        'created_at':datetime.now(timezone.utc).isoformat(),'config':config,'status':'running',
        'hardware_verified':False,'paper_performance_verified':False,
        'inputs':{str(p):hash_file(p) for p in (inputs or [])}}
    write_json(path/'run.json',state)
    return path,state


def log(path,row):
    with (path/'metrics.jsonl').open('a') as stream:
        stream.write(json.dumps(row,allow_nan=False)+'\n')
    print(json.dumps(row,allow_nan=False),flush=True)


def planner_for(env,config,primitive_path,seed):
    backend=config.get('planner_backend','python')
    if not isinstance(backend,str) or backend not in {'python','native'}:
        raise ValueError("planner_backend must be 'python' or 'native'")
    pc=CEMConfig(**config.get('planner',{'timestep_seconds':env.config.control_seconds}))
    if pc.timestep_seconds != env.config.control_seconds or pc.control_seconds != env.config.control_seconds:
        raise ValueError('This 10 Hz joint-command runner requires planner timestep and control period equal to env control_seconds')
    primitive=PrimitiveBasis.load(primitive_path)
    planner=PrimitiveCEM(primitive,env.low,env.high,pc,seed)
    planner.reset(env.data.ctrl)
    cost=RolloutCostConfig(**config.get('planner_cost',{}))
    if backend=='native':
        from .native_rollouts import NativeMujocoRolloutEvaluator
        evaluator=NativeMujocoRolloutEvaluator(env,pc.timestep_seconds,cost,require_native=True)
    else:
        evaluator=MujocoRolloutEvaluator(env,pc.timestep_seconds,cost)
    return planner,evaluator


def _planner_for_run(run,state,env,config,primitive_path,seed):
    """Commit numerical runtime identity before any planning/execution occurs."""
    import mujoco
    import scipy
    backend=config.get('planner_backend','python')
    state['planner_backend']=backend
    evaluator=None
    try:
        planner,evaluator=planner_for(env,config,primitive_path,seed)
        sources=['mpc_dex/cli.py','mpc_dex/env.py','mpc_dex/mpc.py',
                 'mpc_dex/rollouts.py','mpc_dex/primitives.py','mpc_dex/learning.py',
                 'mpc_dex/replay.py','mpc_dex/scheduler.py','mpc_dex/contracts.py','uv.lock']
        if backend=='native':
            sources.extend(['mpc_dex/native_rollouts.py','mpc_dex/native/exact_rollout.cc'])
        actual_backend=('exact_native_step_threads' if getattr(evaluator,'native_library',None) is not None
                        else 'exact_step_threads')
        if backend=='native' and actual_backend!='exact_native_step_threads':
            raise RuntimeError('Explicit native planner did not initialize its native library')
        state['planner_runtime']={
            'schema_version':1,
            'backend':backend,
            'evaluator_backend':actual_backend,
            'evaluator_class':type(evaluator).__module__+'.'+type(evaluator).__qualname__,
            'require_native':bool(getattr(evaluator,'require_native',False)),
            'seed':seed,
            'initialized_at':datetime.now(timezone.utc).isoformat(),
            'planner':asdict(planner.config),'cost':asdict(evaluator.config),
            'workers':evaluator.workers,'environment_contract':environment_contract(env.config),
            'numerical_source_sha256':{name:hash_file(ROOT/name) for name in sources},
            'versions':{'mujoco':mujoco.__version__,'numpy':np.__version__,
                        'scipy':scipy.__version__,'torch':str(torch.__version__)},
            'native_build':getattr(evaluator,'native_provenance',None),
        }
        write_json(run/'run.json',state)
        return planner,evaluator
    except BaseException as error:
        if evaluator is not None:evaluator.close()
        env.close()
        state.update(status='failed',planner_initialization_error=str(error))
        write_json(run/'run.json',state)
        raise


def collect(args):
    config,ec=read_config(args.config)
    run,state=new_run(args.output,'collect',config,[args.primitive])
    env=AllegroRotationEnv(ec);env.reset(seed=args.seed)
    planner,evaluator=_planner_for_run(run,state,env,config,args.primitive,args.seed)
    replay=ReplayBuffer(capacity=1_000_000,seed=args.seed)
    successful_q=[];successful_records=[];successes=0;attempt=0;start=time.monotonic()
    try:
        for attempt in range(1,args.max_attempts+1):
            obs,info=env.reset(seed=args.seed+attempt);planner.reset(env.data.ctrl)
            rows=[];qtrace=[env.data.qpos[env.q_ids].copy()];total=0.;aborted=False
            for step in range(ec.horizon_steps):
                try:target,diagnostics=planner.plan(evaluator)
                except NoFeasiblePlan:
                    aborted=True;break
                action=env.to_action(target)
                next_obs,reward,terminated,truncated,info=env.step(action)
                qtrace.append(env.data.qpos[env.q_ids].copy())
                rows.append((obs,action,reward,next_obs,terminated,truncated));total+=reward;obs=next_obs
                if terminated or truncated:break
            success=bool(info['success'] and not aborted)
            if success:
                keys=('obs','actions','rewards','next_obs','terminated','truncated')
                episode={k:np.asarray([row[i] for row in rows]) for i,k in enumerate(keys)}
                replay.add_episode(episode);successes+=1;successful_q.extend(qtrace)
                successful_records.append(trajectory_episode(info,len(rows),len(qtrace),attempt=attempt))
                replay.save(run/'replay.npz')
                np.savez_compressed(run/'successful_joint_trajectories.npz',joint_positions=np.array(successful_q),
                    metadata=np.asarray(json.dumps({'source':'MPC in free-object MuJoCo simulation',
                        'successful_episodes':successes,'task':ec.task,'environment':asdict(ec),
                        'episodes':successful_records,
                        'rotation_validation':rotation_validation(ec),
                        'policy_interface':{'kind':'absolute_joint_target'},
                        'environment_contract':environment_contract(ec)})))
            log(run,{'attempt':attempt,'success':success,'successful_episodes':successes,'drop':info['drop'],
                'palm_support_violation':info.get('palm_support_violation',False),
                'peak_palm_support_force':info.get('peak_palm_support_force',0.),
                'failure_reason':info.get('failure_reason'),
                'no_feasible_plan':aborted,'angle':info['angle'],'return':total,'steps':len(rows),
                'wall_seconds':time.monotonic()-start})
            if successes>=args.episodes:break
        state.update(status='complete' if successes>=args.episodes else 'incomplete',
            successful_episodes=successes,attempts=attempt,transitions=len(replay),
            required_episodes=args.episodes,environment=asdict(ec))
        state['environment_contract']=environment_contract(ec)
        if successes:
            state['replay_sha256']=hash_file(run/'replay.npz')
        write_json(run/'run.json',state)
        return 0 if successes>=args.episodes else 2
    finally:
        evaluator.close();env.close()


def rotation_validation(config):
    return ROTATION_VALIDATION if config.task=='rotation' else None


def trajectory_episode(info, steps, frames, **values):
    return dict(values,success=bool(info['success']),drop=bool(info['drop']),
                palm_support_violation=bool(info['palm_support_violation']),
                peak_palm_support_force=float(info['peak_palm_support_force']),
                failure_reason=info['failure_reason'],
                angle=float(info['angle']),retained_angle=float(info['retained_angle']),
                finger_contacts=int(info['finger_contacts']),axis_tilt_radians=float(info['axis_tilt_radians']),
                steps=steps,frames=frames)


def validate_rotation_trajectories(frames, metadata):
    if metadata.get('task')!='rotation' or metadata.get('environment',{}).get('task')!='rotation':
        raise ValueError('Primitive extraction requires rotation trajectories, not another task')
    count=metadata.get('successful_episodes',0)
    if type(count) is not int or count<1:
        raise ValueError('Primitive extraction requires recorded successful rotation trajectories')
    if metadata.get('rotation_validation')!=ROTATION_VALIDATION:
        raise ValueError('Trajectories lack validated principal-axis rotation, grasp retention and no-palm support')
    records=metadata.get('episodes')
    if not isinstance(records,list) or len(records)!=count:
        raise ValueError('Trajectory requires one full-rotation validation record per successful episode')
    ec=EnvConfig(**metadata['environment'])
    frame_count=0
    for record in records:
        angle=record.get('angle',float('nan'))
        retained=record.get('retained_angle',float('nan'))
        tilt=record.get('axis_tilt_radians',float('nan'))
        peak_palm_force=record.get('peak_palm_support_force',float('nan'))
        if (record.get('success') is not True or record.get('drop') is not False
            or record.get('palm_support_violation') is not False
            or record.get('failure_reason') is not None
            or not np.isfinite([angle,retained,tilt,peak_palm_force]).all() or min(angle,retained)<2*np.pi
            or not 0<=peak_palm_force<=ec.palm_support_force_tolerance
            or record.get('finger_contacts',0)<2
            or not 0<=tilt<=np.radians(ec.success_axis_tolerance_degrees)):
            raise ValueError('Every trajectory episode must prove a full retained rotation without a drop or palm support')
        steps,episode_frames=record.get('steps'),record.get('frames')
        if type(steps) is not int or steps<1 or type(episode_frames) is not int or episode_frames!=steps+1:
            raise ValueError('Every trajectory must include its initial and terminal joint frames')
        frame_count+=episode_frames
    if frames.ndim!=2 or frames.shape!=(frame_count,16) or not np.isfinite(frames).all():
        raise ValueError('Trajectory joint frames do not match the successful episode records')
    if metadata.get('environment_contract')!=environment_contract(ec):
        raise ValueError('Trajectory simulator assets or measurement source changed; revalidate before extraction')


def extract(args):
    if any((p/'invalidated.json').exists() for p in list(Path(args.trajectory).resolve().parents)[:3]):
        raise ValueError('This trajectory run was invalidated; it cannot supply motion primitives')
    with np.load(args.trajectory,allow_pickle=False) as archive:
        frames=archive['joint_positions']
        metadata=json.loads(str(archive['metadata']))
    validate_rotation_trajectories(frames,metadata)
    primitive=PrimitiveBasis.from_trajectories(frames,provenance={
        'source':metadata['source'],
        'trajectory_file':str(Path(args.trajectory).resolve()),'sha256':hash_file(args.trajectory),
        'record':metadata,'implementation':'independent-paper-reproduction'})
    primitive.save(args.output)
    print(json.dumps({'primitive':args.output,'components':primitive.components,'frames':len(frames)}))
    return 0


def pretrain(args):
    config,ec=read_config(args.config)
    replay=ReplayBuffer.load(args.replay)
    dataset_path=Path(args.replay).parent/'run.json'
    dataset=json.loads(dataset_path.read_text())
    if dataset.get('command')!='collect' or dataset.get('successful_episodes',0)<20:
        raise ValueError('Paper pretraining requires at least 20 successful MPC demonstration episodes')
    if dataset.get('environment')!=canonical(asdict(ec)):
        raise ValueError('Demonstration environment differs from requested training configuration')
    if dataset.get('status')!='complete' or dataset.get('successful_episodes')!=replay.episode_count or dataset.get('transitions')!=len(replay):
        raise ValueError('Demonstration manifest does not match the completed replay')
    if dataset.get('replay_sha256')!=hash_file(args.replay) or dataset.get('environment_contract')!=environment_contract(ec):
        raise ValueError('Demonstration data or model/grasp assets have changed')
    run,state=new_run(args.output,'pretrain',config,[args.replay])
    agent=SACAgent(60 if ec.task=='rotation' else 61,16,LearnerConfig(**config.get('learner',{})),args.device)
    def save(path):
        agent.save(path)
        write_checkpoint_metadata(path,ec,agent.config,stage='pretrain',offline_replay_sha256=hash_file(args.replay))
    try:
        for k in range(args.bc_steps):
            metrics=agent.bc_step(replay.sample(args.batch_size,weighted_episodes=True))
            if (k+1)%100==0 or k==args.bc_steps-1:log(run,dict(stage='bc',step=k+1,**metrics))
            if (k+1)%1000==0:save(run/'checkpoint.pt')
        agent.initialize_std();save(run/'bc.pt')
        for k in range(args.critic_steps):
            metrics=agent.pretrain_critic_step(replay.sample(args.batch_size))
            if (k+1)%100==0 or k==args.critic_steps-1:log(run,dict(stage='soft_sarsa',step=k+1,**metrics))
            if (k+1)%1000==0:save(run/'checkpoint.pt')
        save(run/'checkpoint.pt')
        state.update(status='complete',bc_updates=agent.bc_updates,critic_updates=agent.pretrain_updates,
            paper_pretraining_schedule_completed=agent.bc_updates>=12000 and agent.pretrain_updates>=50000,
            environment=asdict(ec))
        write_json(run/'run.json',state)
        return 0
    except BaseException as error:
        save(run/'checkpoint.pt');state.update(status='interrupted',error=str(error));write_json(run/'run.json',state)
        raise


def train(args):
    config,ec=read_config(args.config)
    checkpoint_record=validate_checkpoint(args.checkpoint,ec)
    if checkpoint_record.get('policy_interface',{'kind':'absolute_joint_target'}).get('kind')!='absolute_joint_target':
        raise ValueError('Full-method training requires an absolute joint-target policy checkpoint')
    agent=SACAgent.load(args.checkpoint,args.device)
    if checkpoint_record['learner_config']!=canonical(asdict(agent.config)) or canonical(asdict(LearnerConfig(**config.get('learner',{}))))!=canonical(asdict(agent.config)):
        raise ValueError('Requested learner configuration differs from checkpoint')
    if agent.online_updates:
        raise ValueError('Online checkpoints require restoring replay and scheduler; this fresh-training command only accepts pretrained checkpoints')
    if agent.bc_updates<12000 or agent.pretrain_updates<50000:
        raise ValueError('Online full-method training requires completed BC and critic pretraining')
    if agent.obs_dim!=(60 if ec.task=='rotation' else 61):raise ValueError('Checkpoint task observation size mismatch')
    offline=ReplayBuffer.load(args.replay)
    if checkpoint_record.get('offline_replay_sha256')!=hash_file(args.replay):
        raise ValueError('Online fixed replay differs from the pretraining dataset')
    if offline.episode_count<20:raise ValueError('Full method requires the fixed 20-episode MPC replay')
    run,state=new_run(args.output,'train',config,[args.checkpoint,args.replay,args.primitive])
    online=ReplayBuffer(seed=args.seed);scheduler=ChunkScheduler(seed=args.seed)
    env=AllegroRotationEnv(ec);obs,_=env.reset(seed=args.seed)
    planner,evaluator=_planner_for_run(run,state,env,config,args.primitive,args.seed)
    step=0;drops=0;successes=0;episode_return=0.;start=time.monotonic()
    def save():
        agent.save(run/'checkpoint.pt');online.save(run/'online_replay.npz')
        write_checkpoint_metadata(run/'checkpoint.pt',ec,agent.config,stage='online',offline_replay_sha256=hash_file(args.replay))
    try:
        for step in range(args.steps):
            controller=scheduler.select(step)
            if controller=='mpc':
                if step%40==0:planner.reset(env.data.ctrl)
                target,_=planner.plan(evaluator);action=env.to_action(target)
            else:action=agent.act(obs)
            next_obs,reward,terminated,truncated,info=env.step(action)
            # An external training-budget stop is a truncation, never a success.
            budget_end=step+1==args.steps
            online.add_transition(obs,action,reward,next_obs,terminated,truncated or (budget_end and not terminated))
            obs=next_obs;episode_return+=reward
            for _ in range(args.utd):metrics=agent.update(sample_mixed(offline,online,args.batch_size,.3))
            if terminated or truncated:
                drops+=int(info['drop']);successes+=int(info['success'])
                log(run,{'event':'episode','environment_step':step+1,'return':episode_return,'angle':info['angle'],
                    'success':info['success'],'drop':info['drop'],'controller':controller,
                    'palm_support_violation':info['palm_support_violation'],
                    'peak_palm_support_force':info['peak_palm_support_force'],
                    'failure_reason':info['failure_reason']})
                obs,_=env.reset();planner.reset(env.data.ctrl);scheduler.reset_episode();episode_return=0.
            if (step+1)%100==0 or budget_end:
                log(run,dict(environment_step=step+1,drops=drops,successes=successes,
                    policy_probability=scheduler.policy_probability(step),wall_seconds=time.monotonic()-start,**metrics))
                save()
        state.update(status='complete',environment_steps=step+1,drops=drops,successful_training_episodes=successes,
            environment=asdict(ec),policy_only_evaluation_required=True)
        write_json(run/'run.json',state)
        return 0
    except BaseException as error:
        save()
        state.update(status='interrupted',environment_steps=step,error=str(error));write_json(run/'run.json',state)
        raise
    finally:evaluator.close();env.close()


def evaluate(args):
    config,ec=read_config(args.config)
    metadata=validate_checkpoint(args.checkpoint,ec)
    interface=metadata.get('policy_interface',{'kind':'absolute_joint_target'})
    agent=SACAgent.load(args.checkpoint,args.device)
    if agent.obs_dim!=(60 if ec.task=='rotation' else 61):raise ValueError('Policy observation dimension mismatch')
    run,state=new_run(args.output,'evaluate',config,[args.checkpoint])
    env=environment_for_policy(ec,interface)
    results=[];successful_q=[];successful_records=[];start=time.monotonic()
    reward_semantics=('bootstrap retained-rotation shaping' if interface['kind']=='bootstrap_joint_increment'
                      else 'paper task reward')
    try:
        for episode in range(args.episodes):
            obs,_=env.reset(seed=args.seed+episode);trace=[env.data.qpos[env.q_ids].copy()];total=0.;paper_total=0.
            for step in range(ec.horizon_steps):
                obs,reward,terminated,truncated,info=env.step(agent.act(obs,deterministic=True));total+=reward
                paper_total+=info.get('paper_reward',reward)
                trace.append(env.data.qpos[env.q_ids].copy())
                if terminated or truncated:break
            result=trajectory_episode(info,step+1,len(trace),episode=episode,
                return_value=total,paper_return=paper_total,reward_semantics=reward_semantics,
                rotation_degrees_per_second=np.degrees(info['angle'])/info['sim_time'])
            result['return']=total
            results.append(result);log(run,result)
            if info['success']:
                successful_q.extend(trace);successful_records.append(result)
        successes=sum(r['success'] for r in results)
        state.update(status='complete',episodes=args.episodes,successes=successes,
            drops=sum(r['drop'] for r in results),success_rate=successes/args.episodes,
            wall_seconds=time.monotonic()-start,environment=asdict(ec),mpc_used=False,
            policy_interface=interface,reward_semantics=reward_semantics,results=results)
        write_json(run/'run.json',state)
        if successful_q:
            np.savez_compressed(run/'successful_joint_trajectories.npz',joint_positions=np.array(successful_q),
                metadata=np.asarray(json.dumps({'source':'deterministic SAC policy in free-object MuJoCo simulation',
                    'successful_episodes':successes,'task':ec.task,'environment':asdict(ec),'checkpoint_sha256':hash_file(args.checkpoint),
                    'episodes':successful_records,'policy_interface':interface,'reward_semantics':reward_semantics,
                    'rotation_validation':rotation_validation(ec),
                    'environment_contract':environment_contract(ec)})))
        return 0 if successes==args.episodes else 2
    finally:env.close()


def render(args):
    from PIL import Image
    config,ec=read_config(args.config)
    env=AllegroRotationEnv(ec)
    try:
        env.reset(seed=args.seed)
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(env.render()).save(args.output)
        print(json.dumps({'image':args.output,'environment':asdict(ec)}))
        return 0
    finally:env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name,func in [('collect',collect),('pretrain',pretrain),('train',train),('evaluate',evaluate),('render',render)]:
        p=sub.add_parser(name);p.set_defaults(function=func)
        p.add_argument('--config',default=str(ROOT/'configs/local_sim.json'))
        p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=0)
        if name in ('pretrain','train','evaluate'):p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
        if name in ('collect','train'):p.add_argument('--primitive',required=True)
        if name in ('pretrain','train'):
            p.add_argument('--replay',required=True);p.add_argument('--batch-size',type=int,default=256)
        if name in ('train','evaluate'):p.add_argument('--checkpoint',required=True)
        if name in ('collect','evaluate'):p.add_argument('--episodes',type=int,default=20 if name=='collect' else 5)
        if name=='collect':p.add_argument('--max-attempts',type=int,default=100)
        if name=='pretrain':
            p.add_argument('--bc-steps',type=int,default=12000);p.add_argument('--critic-steps',type=int,default=50000)
        if name=='train':
            p.add_argument('--steps',type=int,default=12000);p.add_argument('--utd',type=int,default=8)
    p=sub.add_parser('extract-primitive');p.set_defaults(function=extract)
    p.add_argument('--trajectory',required=True);p.add_argument('--output',required=True)
    args=parser.parse_args()
    for key in ('episodes','max_attempts','bc_steps','critic_steps','steps','utd','batch_size'):
        if hasattr(args,key) and getattr(args,key)<=0:parser.error(f'{key} must be positive')
    torch.set_num_threads(4)
    torch.manual_seed(getattr(args,'seed',0));np.random.seed(getattr(args,'seed',0))
    return args.function(args)
