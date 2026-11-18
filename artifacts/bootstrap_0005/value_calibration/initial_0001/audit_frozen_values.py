"""Read-only frozen SAC value calibration against actual strict contact episodes."""
from pathlib import Path
import argparse,copy,hashlib,json,random,shutil,sys,math
from datetime import datetime,timezone


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def main():
    parser=argparse.ArgumentParser()
    for key in ('checkpoint','config','source','output','label','baseline','reference-transitions'):
        parser.add_argument('--'+key,required=key not in ('baseline','reference-transitions'))
    args=parser.parse_args();out=Path(args.output).resolve();source=Path(args.source).resolve()
    out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(__file__,out/'audit_frozen_values.py')
    source_hashes={str(p.relative_to(source)):digest(p) for p in sorted(source.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    shutil.copytree(source,out/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    assert all(digest(source/p)==h==digest(out/'source'/p) for p,h in source_hashes.items())
    write(out/'source_hashes.json',source_hashes)
    bindings={}
    def capture(original,name):
        original=Path(original).resolve();before=digest(original);destination=out/name
        shutil.copyfile(original,destination)
        assert digest(destination)==before==digest(original)
        bindings[name]=dict(source_path=str(original),sha256=before)
        return destination
    checkpoint=capture(args.checkpoint,'checkpoint.pt')
    capture(str(args.checkpoint)+'.metadata.json','checkpoint.pt.metadata.json')
    original_config=capture(args.config,'source_config.json')
    original_state=Path(args.checkpoint).parent/'state.json'
    if original_state.exists():capture(original_state,'source_runner_state.json')
    original_trajectories=Path(args.checkpoint).parent/'successful_joint_trajectories.npz'
    if original_trajectories.exists():capture(original_trajectories,'source_successful_joint_trajectories.npz')
    baseline=capture(args.baseline,'baseline.pt') if args.baseline else None
    if baseline:capture(str(args.baseline)+'.metadata.json','baseline.pt.metadata.json')
    references=capture(args.reference_transitions,'reference_transitions.npz') if args.reference_transitions else None
    config=json.loads(original_config.read_text())
    grasp=Path(config['environment']['initial_grasp'])
    if not grasp.is_absolute():grasp=source/grasp
    copied_grasp=capture(grasp,'initial_grasp.npz')
    config['environment']['initial_grasp']=str(copied_grasp)
    write(out/'config.json',config);write(out/'input_bindings.json',bindings)
    sys.path.insert(0,str(out/'source'))
    import numpy as np,torch,mujoco
    from mpc_dex.env import EnvConfig
    from mpc_dex.learning import SACAgent
    from mpc_dex.bootstrap_env import environment_for_policy
    from mpc_dex.contracts import validate_checkpoint,environment_contract
    from scripts import audit_policy
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.is_available()
    ec=EnvConfig(**config['environment'])
    assert ec.task=='rotation' and ec.terminate_on_success and ec.horizon_steps==100
    assert not config.get('bootstrap',{}).get('control')
    metadata=validate_checkpoint(checkpoint,ec)
    assert metadata['policy_interface']=={'kind':'absolute_joint_target'}
    agent=SACAgent.load(checkpoint,'cuda');alpha=float(agent.alpha.detach());gamma=agent.config.discount
    assert agent.config.ensemble_size==2
    def frozen_fingerprint(learner):
        states={name:{k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in getattr(learner,name).state_dict().items()} for name in ('actor','critics','target_critics')}
        states['alpha']=float(learner.log_alpha.detach());states['counters']=[learner.bc_updates,learner.pretrain_updates,learner.online_updates]
        return states
    before=frozen_fingerprint(agent)
    if (out/'source_runner_state.json').exists():
        runner_state=json.loads((out/'source_runner_state.json').read_text())
        assert runner_state['online_updates']==agent.online_updates
        assert runner_state['bc_updates']==agent.bc_updates and runner_state['pretrain_updates']==agent.pretrain_updates
    success_export=dict(present=False,episodes=0,sac_updated_episodes=0,bc_origin_episodes=0)
    if (out/'source_successful_joint_trajectories.npz').exists():
        with np.load(out/'source_successful_joint_trajectories.npz',allow_pickle=False) as z:
            trajectory_metadata=json.loads(str(z['metadata']))
        episodes=trajectory_metadata['episodes']
        assert all(type(ep.get('online_updates')) is int and ep['online_updates']>=0 for ep in episodes)
        success_export=dict(present=True,episodes=len(episodes),sac_updated_episodes=sum(ep['online_updates']>0 for ep in episodes),bc_origin_episodes=sum(ep['online_updates']==0 for ep in episodes),records=episodes)
    write(out/'source_success_export_review.json',success_export)
    runtime=dict(created_at=datetime.now(timezone.utc).isoformat(),label=args.label,device='cuda',
        torch_version=str(torch.__version__),cuda_version=torch.version.cuda,numpy_version=np.__version__,mujoco_version=mujoco.__version__,
        gpu=torch.cuda.get_device_name(),gpu_capability=list(torch.cuda.get_device_capability()),torch_threads=torch.get_num_threads(),
        torch_interop_threads=torch.get_num_interop_threads(),cuda_matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        float32_matmul_precision=torch.get_float32_matmul_precision(),deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,
        source_success_export_review=success_export,checkpoint_sha256=digest(checkpoint),metadata_sha256=digest(out/'checkpoint.pt.metadata.json'),
        bc_updates=agent.bc_updates,pretrain_updates=agent.pretrain_updates,online_updates=agent.online_updates,
        alpha=alpha,gamma=gamma,policy_interface=metadata['policy_interface'],environment_contract=environment_contract(ec),
        frozen_state_fingerprint=before,new_learning_updates=0,
        actor_call='singleton float32 CUDA actor.sample; actual sampled log_prob returned directly',
        deterministic_call='singleton float32 CUDA tanh(mean), density evaluated at raw mean under frozen stochastic policy')
    write(out/'runtime.json',runtime)
    base_agent=SACAgent.load(baseline,'cuda') if baseline else None
    if baseline:validate_checkpoint(baseline,ec)
    probe_env=environment_for_policy(ec,metadata['policy_interface'])
    try:initial_obs,_=probe_env.reset(seed=42)
    finally:probe_env.close()
    probe_observations=[initial_obs]
    if references:
        with np.load(references,allow_pickle=False) as z:probe_observations.extend(z['obs'])
    probes=np.asarray(probe_observations,dtype=np.float32);np.save(out/'probe_observations.npy',probes)
    def probe(learner):
        result={k:[] for k in ('raw_mean','mean_action','latent_std')}
        with torch.no_grad():
            for ob in probes:
                mean,logstd=learner.actor(torch.as_tensor(ob,device=learner.device,dtype=torch.float32))
                for k,v in [('raw_mean',mean),('mean_action',mean.tanh()),('latent_std',logstd.exp())]:result[k].append(v.cpu().numpy())
        return {k:np.asarray(v) for k,v in result.items()}
    current_probes=probe(agent);np.savez_compressed(out/'policy_probes.npz',**current_probes)
    movement=dict(latent_std_min=float(current_probes['latent_std'].min()),latent_std_max=float(current_probes['latent_std'].max()),
                  cold_reset_std_min=float(current_probes['latent_std'][0].min()),cold_reset_std_max=float(current_probes['latent_std'][0].max()))
    if base_agent:
        base_probes=probe(base_agent);np.savez_compressed(out/'baseline_policy_probes.npz',**base_probes)
        movement['baseline_checkpoint_sha256']=digest(baseline)
        for key in current_probes:
            delta=current_probes[key].astype(np.float64)-base_probes[key]
            movement[key+'_max_abs_change']=float(np.abs(delta).max());movement[key+'_rms_change']=float(np.sqrt(np.mean(delta**2)))
        movement['actor_parameter_max_abs_change']={key:float(torch.max(torch.abs(value-base_agent.actor.state_dict()[key]))) for key,value in agent.actor.state_dict().items()}
    write(out/'policy_movement.json',movement)
    report=dict(status='running',label=args.label,checkpoint_sha256=digest(checkpoint),runtime=runtime,
        policy_movement=movement,prespecified_stochastic_seeds=list(range(17041,17046)),deterministic_seed=42,strict_control_limit=100,
        interpretation='Fixed-seed frozen-policy sample diagnostics; not a robustness-rate estimate. No new learning.',
        q_convention='Q(s0,a0) excludes the current action entropy; compare each critic on the actually executed first action.',
        soft_q_return_formula='sum(t=0..T-1,gamma**t*r[t]) - alpha*sum(t=1..T-1,gamma**t*logpi[t])',
        soft_v_return_formula='soft_q_return-alpha*logpi[0]',
        truncation_caveat='Without genuine terminal flags the unobserved gamma**T*V_soft(s_T) tail remains; prefix bias is not full-return calibration.',
        ensemble_max_definition='Maximum over the two critics for the same executed action; never an action maximization.',
        deterministic_caveat='Deterministic future behavior is not a sample from the frozen stochastic policy; its returns are behavior diagnostics.',
        episodes=[])
    write(out/'report.json',report)
    def run_episode(seed,deterministic):
        directory=out/('deterministic_strict' if deterministic else f'stochastic_seed_{seed}');directory.mkdir()
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed);random.seed(seed)
        torch.save(dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()),directory/'torch_rng_initial.pt')
        env=environment_for_policy(ec,metadata['policy_interface'])
        records={k:[] for k in ('obs','actions','raw_mean','mean_actions','latent_std','log_probs','critic_q','target_critic_q','next_obs','rewards','terminated','truncated','qpos','qvel','targets','integration_state')}
        initial={};physics_states=[];physics_integration=[]
        spec=mujoco.mjtState.mjSTATE_INTEGRATION;fullspec=mujoco.mjtState.mjSTATE_FULLPHYSICS
        def state(specification):
            values=np.empty(mujoco.mj_stateSize(env.model,specification));mujoco.mj_getState(env.model,env.data,values,specification);return values
        original_reset,original_step=env.reset,env.step
        original_capture=audit_policy.PhysicsObserver.capture
        def capture_physics(observer):
            original_capture(observer)
            assert observer.env is env
            physics_states.append(state(fullspec));physics_integration.append(state(spec))
        def reset(*a,**kw):
            result=original_reset(*a,**kw)
            initial.update(obs=result[0].copy(),qpos=env.data.qpos.copy(),qvel=env.data.qvel.copy(),ctrl=env.data.ctrl.copy(),integration_state=state(spec))
            return result
        def step(action):
            result=original_step(action);ob,reward,terminated,truncated,info=result
            for k,v in dict(next_obs=ob.copy(),rewards=reward,terminated=terminated,truncated=truncated,qpos=env.data.qpos.copy(),qvel=env.data.qvel.copy(),targets=env.data.ctrl.copy(),integration_state=state(spec)).items():records[k].append(v)
            return result
        class Policy:
            @torch.no_grad()
            def act(self,ob,deterministic=True):
                tensor=torch.as_tensor(ob,device=agent.device,dtype=torch.float32)
                mean,logstd=agent.actor(tensor)
                if deterministic:
                    action=mean.tanh();logp=agent.actor._log_prob(mean,logstd,mean)
                else:action,logp=agent.actor.sample(tensor)
                q=agent.critics(tensor,action);tq=agent.target_critics(tensor,action)
                for k,v in [('obs',ob.copy()),('actions',action.cpu().numpy()),('raw_mean',mean.cpu().numpy()),('mean_actions',mean.tanh().cpu().numpy()),('latent_std',logstd.exp().cpu().numpy()),('log_probs',float(logp)),('critic_q',q.cpu().numpy()),('target_critic_q',tq.cpu().numpy())]:records[k].append(v)
                return action.cpu().numpy()
        env.reset,env.step=reset,step;audit_policy.PhysicsObserver.capture=capture_physics
        try:result,physics,commands=audit_policy.audit_episode(env,Policy(),seed,100,deterministic)
        finally:
            audit_policy.PhysicsObserver.capture=original_capture;env.close()
        arrays={k:np.asarray(v) for k,v in records.items()};T=len(arrays['rewards'])
        assert len(physics_states)==len(physics)==T*50
        discounts=np.power(gamma,np.arange(T,dtype=np.float64));rewards=arrays['rewards'].astype(np.float64);logp=arrays['log_probs'].astype(np.float64)
        raw_return=float(discounts@rewards);soft_q_return=float(raw_return-alpha*(discounts[1:]@logp[1:]));soft_v_return=float(soft_q_return-alpha*logp[0])
        q0=arrays['critic_q'][0].astype(np.float64);terminal=bool(arrays['terminated'][-1]);complete=terminal
        value=dict(critic_estimates_for_executed_initial_action=q0.tolist(),target_critic_estimates_for_executed_initial_action=arrays['target_critic_q'][0].tolist(),
            critic_ensemble_min_for_same_action=float(q0.min()),critic_ensemble_mean_for_same_action=float(q0.mean()),critic_ensemble_max_for_same_action=float(q0.max()),
            realized_discounted_reward=raw_return,realized_soft_q_return=soft_q_return,realized_soft_v_return=soft_v_return,
            critic_minus_discounted_reward=(q0-raw_return).tolist(),critic_minus_soft_q_return=(q0-soft_q_return).tolist(),
            initial_action_log_prob=float(logp[0]),initial_action=arrays['actions'][0].tolist(),alpha=alpha,gamma=gamma,
            terminal_complete=complete,discounted_tail_coefficient=0. if complete else float(gamma**T),
            missing_continuation_tail=not complete,stochastic_q_sample=not deterministic and complete,
            first_action_q_vs_trajectory_max_q_distinct=True,max_critic_q_across_executed_trajectory=float(arrays['critic_q'].max()))
        result.update(value_calibration=value,deterministic=deterministic,checkpoint_online_updates=agent.online_updates,
            source_bc_updates=agent.bc_updates,new_learning_updates=0,
            sac_updated_policy_success=bool(result['finger_supported_rotation_success'] and agent.online_updates>0),
            bc_origin_success=bool(result['finger_supported_rotation_success'] and agent.online_updates==0),
            latent_std_min=float(arrays['latent_std'].min()),latent_std_max=float(arrays['latent_std'].max()))
        np.savez_compressed(directory/'transitions.npz',**arrays,**{'initial_'+k:v for k,v in initial.items()})
        np.savez_compressed(directory/'physics.npz',samples=physics,columns=np.asarray(audit_policy.PHYSICS_COLUMNS),fullphysics=np.asarray(physics_states),integration_states=np.asarray(physics_integration))
        write(directory/'report.json',result);write(directory/'commands.json',commands)
        torch.save(dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()),directory/'torch_rng_final.pt')
        assert frozen_fingerprint(agent)==before
        print(json.dumps(dict(label=args.label,seed=seed,deterministic=deterministic,steps=T,success=result['finger_supported_rotation_success'],failure=result['environment_failure_reason'],max_force_qualified=result['max_force_qualified_angle'],q0=q0.tolist(),reward_return=raw_return,soft_q_return=soft_q_return)),flush=True)
        return result
    for seed,deterministic in [(42,True)]+[(s,False) for s in range(17041,17046)]:
        report['episodes'].append(run_episode(seed,deterministic));write(out/'report.json',report)
    samples=report['episodes'][1:]
    report.update(status='complete',frozen_weights_verified=True,stochastic_sample_count=5,
        stochastic_sample_successes=sum(x['finger_supported_rotation_success'] for x in samples),
        stochastic_sample_palm_violations=sum(x['independent_palm_support_violation'] for x in samples),
        stochastic_sample_drops=sum(x['independent_drop'] for x in samples),
        mean_matched_action_critic_bias_vs_soft_q=np.mean([x['value_calibration']['critic_minus_soft_q_return'] for x in samples],axis=0).tolist())
    assert digest(checkpoint)==runtime['checkpoint_sha256']
    assert all(digest(source/p)==h==digest(out/'source'/p) for p,h in source_hashes.items())
    assert all(digest(Path(record['source_path']))==record['sha256'] for record in bindings.values())
    write(out/'report.json',report)
    write(out/'manifest.json',{str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p!=out/'manifest.json'})

if __name__=='__main__':main()
