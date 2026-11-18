"""Independent saved-data/inference audit; zero physics or optimizer updates."""
from pathlib import Path
import hashlib, json, random, sys
import numpy as np
import torch

D=Path(__file__).resolve().parent
S=D.parent/'bootstrap_suffix_sac_0001'
sys.path.insert(0,str(S/'source'))
from mpc_dex.learning import SACAgent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b,equal_nan=a.dtype.kind in 'fc')
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
def rng():return dict(torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state_all(),numpy=np.random.get_state(),python=random.getstate())
def restore(x):
    torch.set_rng_state(x['torch_cpu']);torch.cuda.set_rng_state_all(x['torch_cuda'])
    np.random.set_state(x['numpy']);random.setstate(x['python'])
def rng_load(p):return torch.load(p,map_location='cpu',weights_only=False)
def verify_manifest(base,name):
    x=read(base/name)
    for k,v in x.items():assert sha(base/k)==v,(name,k)
    return dict(path=str(base/name),sha256=sha(base/name),files=len(x))

def episode(path, source, source_phys, config):
    t=load(path/'transitions.npz');p=load(path/'physics.npz');r=read(path/'report.json')
    n=len(t['actions']);prefix=r['prefix_controls'];x=p['samples'];c={k:x[:,i] for i,k in enumerate(p['columns'])}
    assert n==r['steps'] and len(x)==50*n==r['physics_steps']
    assert equal(t['policy_controlled'],np.arange(n)>=prefix)
    assert equal(t['log_prob_valid'],t['policy_controlled'])
    assert np.isnan(t['log_prob'][:prefix]).all() and np.isfinite(t['log_prob'][prefix:]).all()
    assert equal(t['next_obs'][:-1],t['obs'][1:])
    assert not (t['terminated'][:-1]|t['truncated'][:-1]).any()
    assert bool(t['terminated'][-1])==r['terminated'] and bool(t['truncated'][-1])==r['truncated']
    assert not t['next_action_valid'][-1] and not t['next_actions'][-1].any()
    assert t['next_action_valid'][:-1].all() and equal(t['next_actions'][:-1],t['actions'][1:])
    assert equal(t['normalized_rewards'],t['rewards']/200)
    for k in ['obs','actions','next_obs','rewards','terminated','truncated','qpos','qvel','targets','integration_state']:
        assert equal(t[k][:prefix],source[k][:prefix]),(path,k)
    assert equal(x[:prefix*50],source_phys[:prefix*50])
    assert equal(t['initial_integration_state'],source['initial_integration_state'])
    if prefix:
        assert equal(t['handoff_integration_state'],source['integration_state'][prefix-1])
        assert equal(t['handoff_obs'],source['next_obs'][prefix-1])
    assert equal(t['integration_state'][:,1:24],t['qpos']) and equal(t['integration_state'][:,24:46],t['qvel'])
    assert equal(t['time'],t['integration_state'][:,0]) and equal(c['time'][49::50],t['time'])
    assert np.allclose(c['time'],np.arange(1,len(x)+1)*.002,atol=1e-12,rtol=0)
    palm=np.maximum.accumulate(c['palm_normal_force']>config['palm_support_force_tolerance'])
    assert equal(palm,c['palm_support_violation'].astype(bool))
    retained=(c['finger_contacts']>=2)&(c['axis_tilt_radians']<=np.deg2rad(config['success_axis_tolerance_degrees']))&~palm
    loaded=retained&(c['force_bearing_fingers']>=2)
    assert equal(retained,c['retained'].astype(bool)) and equal(loaded,c['force_retained'].astype(bool))
    for kind,keep in [('raw_angle',np.ones(len(x),bool)),('qualified_angle',retained),('force_qualified_angle',loaded)]:
        y=np.cumsum(np.where(keep|(c['twist_increment']<0),c['twist_increment'],0.))
        assert np.allclose(y,c[kind],rtol=0,atol=1e-12),(path,kind)
        assert abs(y[-1]-r['final_'+kind])<1e-12
    drop=bool(np.any(c['table_contacts']) or np.any(c['object_displacement']>.18) or
              np.any(c['contactless_seconds']>=config['contact_loss_seconds']-1e-10))
    assert drop==r['independent_drop'] and bool(palm[-1])==r['independent_palm_support_violation']
    external=('palm_force_norm','table_force_norm','other_force_norm','table_normal_force','other_normal_force',
              'object_applied_wrench_norm','object_applied_generalized_force_norm')
    no_external=not palm[-1] and all(np.max(c[k],initial=0)<=1e-6 for k in external)
    assert bool(no_external)==r['no_external_support']
    success=bool(r['terminated'] and not r['truncated'] and r['environment_success'] and not r['environment_drop'] and
        all(c[k][-1]>=2*np.pi for k in ('raw_angle','qualified_angle','force_qualified_angle')) and loaded[-1] and
        not drop and no_external and r['model_equality_constraints']==0)
    assert success==r['finger_supported_rotation_success']
    assert r['fixed_prefix_assisted']==bool(prefix) and not r['eligible_for_primitive'] and not r['trajectory_exported']
    assert r['cold_start_policy_success']==(prefix==0 and success)
    assert r['raw_suffix_return']==float(t['rewards'][prefix:].sum())
    assert r['raw_full_return']==float(t['rewards'].sum())
    rewards=20*np.diff(np.r_[0.,c['raw_angle'][49::50]])-.3
    if r['environment_success']:rewards[-1]+=200
    if r['environment_drop'] or r['independent_palm_support_violation']:rewards[-1]=-50.
    assert np.allclose(rewards,t['rewards'],rtol=0,atol=1e-10),(path,'actual physics reward')
    return t,r

def empirical(t,r,alpha):
    if not r['terminated'] or r['truncated']:return None
    p=r['prefix_controls'];rew=t['rewards'][p:]/200;lp=t['log_prob'][p:]
    # Independent forward sum, excluding current-action entropy.
    return np.asarray([sum(.995**(j-i)*rew[j] for j in range(i,len(rew)))-
                       alpha*sum(.995**(j-i)*lp[j] for j in range(i+1,len(rew))) for i in range(len(rew))])

def main():
    assert read(S/'summary.json')['status']=='complete' and not (S/'failure.json').exists()
    manifests=[verify_manifest(S,'execution_manifest.json'),verify_manifest(S,'manifest.json')]
    protocol=read(S/'protocol.json');config=read(S/'config.json')['environment'];summary=read(S/'summary.json')
    for k,v in protocol['source_bindings'].items():assert sha(S/k)==v['sha256']==sha(v['source'])
    checkpoint_names=['initialized','warmup_complete','pretrained','online_500','online_1000','online_2000']
    checkpoints={};checkpoint_checks=[];metrics_bytes=(S/'metrics.jsonl').read_bytes()
    metrics=[json.loads(x) for x in metrics_bytes.splitlines()]
    for name in checkpoint_names:
        p=S/'checkpoints'/name;manifests.append(verify_manifest(p,'manifest.json'))
        a=torch.load(p/'agent.pt',map_location='cpu',weights_only=True);checkpoints[name]=a
        state=read(p/'state.json');n=int(name.split('_')[1]) if name.startswith('online_') else 0
        pre=500 if name in ('pretrained','online_500','online_1000','online_2000') else 0
        assert (a['bc_updates'],a['pretrain_updates'],a['online_updates'])==(12000,pre,n)
        opt={}
        for k,expected,param_count in [('critic_optimizer',pre+n,20),('actor_optimizer',n,12),('alpha_optimizer',n,1),('bc_optimizer',0,0)]:
            states=a[k]['state'];assert len(states)==(param_count if expected else 0),(name,k)
            steps=sorted(set(float(v['step']) for v in states.values()))
            assert steps==([float(expected)] if expected else []),(name,k,steps)
            opt[k]=dict(parameter_states=len(states),steps=steps)
        if n:assert a['actor_optimizer']['param_groups'][0]['lr']==.0003*(n/5000)
        if name in ('warmup_complete','pretrained'):
            assert equal(a['actor'],checkpoints['initialized']['actor']) and equal(a['log_alpha'],checkpoints['initialized']['log_alpha'])
        if name=='warmup_complete':
            assert equal(a,checkpoints['initialized'])
        assert hashlib.sha256(metrics_bytes[:state['metrics_bytes']]).hexdigest()==state['metrics_sha256']
        assert not state['automatic_resume_supported'] and state['complete_update_boundary']
        checkpoint_checks.append(dict(checkpoint=name,agent_sha256=sha(p/'agent.pt'),bc_updates=12000,pretrain_updates=pre,online_updates=n,optimizers=opt))
    original=torch.load(S/'initial.pt',map_location='cpu',weights_only=True)
    for key in ('actor','critics','target_critics','bc_optimizer','actor_optimizer','critic_optimizer','alpha_optimizer','bc_updates','pretrain_updates','online_updates','std_initialized'):
        assert equal(original[key],checkpoints['initialized'][key]),key
    source=load(S/'prefix_transitions.npz');source_phys=load(S/'prefix_physics.npz')['samples']
    training=[];stage_summary={};replay_expected={k:[] for k in ['obs','actions','rewards','next_obs','terminated','truncated','next_actions','next_action_valid','episode_id','episode_return','episode_complete']}
    for stage in ('warmup','online'):
        paths=sorted((S/'episodes'/stage).iterdir())
        if stage=='online' and (S/'unfinished_suffix').exists():paths.append(S/'unfinished_suffix')
        stage_reports=[]
        for p in paths:
            t,r=episode(p,source,source_phys,config);assert r['prefix_controls']==7 and r['phase']==stage
            assert len(t['actions'])>7
            for k in ['learner_bc_updates','learner_pretrain_updates','learner_online_updates']:assert (t[k][:7]==-1).all()
            assert (t['learner_bc_updates'][7:]==12000).all()
            assert (t['learner_pretrain_updates'][7:]==(0 if stage=='warmup' else 500)).all()
            if stage=='warmup':assert (t['learner_online_updates'][7:]==0).all()
            assert t['density_is_stochastic_sample'][7:].all()
            ep_id=len(training);training.append((p,t,r));stage_reports.append(r);length=len(t['actions'])-7
            for k in ['obs','actions','rewards','next_obs','terminated','truncated','next_actions','next_action_valid']:replay_expected[k].append(t[k][7:])
            replay_expected['episode_id'].append(np.full(length,ep_id,np.int64))
            complete=r['terminated'] or r['truncated']
            replay_expected['episode_complete'].append(np.full(length,complete,bool))
            replay_expected['episode_return'].append(np.full(length,r['raw_suffix_return'] if complete else 0,np.float32))
        stage_summary[stage]=dict(episodes_started=len(paths),completed=sum(r['environment_episode_complete'] for r in stage_reports),
            suffix_controls=sum(r['policy_suffix_controls'] for r in stage_reports),strict_force_successes=sum(r['finger_supported_rotation_success'] for r in stage_reports),
            environment_successes=sum(r['environment_success'] for r in stage_reports),independent_drops=sum(r['independent_drop'] for r in stage_reports),
            palm_violations=sum(r['independent_palm_support_violation'] for r in stage_reports),truncated=sum(r['truncated'] for r in stage_reports),
            raw_suffix_reward_sum=sum(r['raw_suffix_return'] for r in stage_reports))
    replay=load(S/'checkpoints/online_2000/replay.npz');meta=json.loads(str(replay['metadata']))
    for k,v in replay_expected.items():
        expected=np.concatenate(v).astype(replay[k].dtype)
        assert equal(expected,replay[k]),k
    assert len(replay['actions'])==2808 and meta['episode_id']==1009 and meta['open_episode'] and meta['position']==2808
    assert not replay['terminated'][-1] and not replay['truncated'][-1] and not replay['next_action_valid'][-1]
    assert equal(np.concatenate([t['learner_online_updates'][7:] for _,t,r in training if r['phase']=='online']),np.arange(2000))
    assert stage_summary['warmup']['completed']==256 and stage_summary['online']['completed']==753
    assert stage_summary['warmup']['suffix_controls']==808 and stage_summary['online']['suffix_controls']==2000
    assert 7*len(training)==summary['counts']['physical_prefix_controls']==7070
    for stage in ['warmup','online']:
        for field,key in [('completed','assisted_completed_episodes'),('strict_force_successes','assisted_force_successes'),('suffix_controls','policy_interactions')]:
            assert stage_summary[stage][field]==summary['counts'][stage+'_'+key]
        assert abs(stage_summary[stage]['raw_suffix_reward_sum']-summary['counts'][stage+'_raw_reward_sum'])<1e-8
    print('Training replay, all episode physics records, and checkpoint optimizers verified.',flush=True)

    # Independently reconstruct every minibatch from the replay's private RNG.
    replay_rng=np.random.default_rng(protocol['train_seed']);indices=[]
    for event,total,count in [('critic_pretrain_update',500,808),('online_update',2000,None)]:
        rows=[x for x in metrics if x['event']==event];assert len(rows)==total
        for i,row in enumerate(rows,1):
            size=count if count else 808+i;idx=replay_rng.integers(size,size=256);indices.append(idx)
            rewards=replay['rewards'][idx]
            assert float(rewards.mean())==row['batch_raw_reward_mean']
            assert float((rewards/np.float32(200)).mean())==row['batch_normalized_reward_mean']
            if event=='critic_pretrain_update':assert row['pretrain_updates']==i and row['pretrain_samples']==256
            else:
                assert row['online_updates']==i and row['actor_lr']==.0003*min(i/5000,1)
                assert row['raw_transition_reward']==float(np.concatenate([t['rewards'][7:] for _,t,r in training if r['phase']=='online'])[i-1])
                assert row['normalized_transition_reward']==row['raw_transition_reward']/200
        label='pretrained' if count else 'online_2000'
        assert equal(replay_rng.bit_generator.state,json.loads(str(load(S/'checkpoints'/label/'replay.npz')['metadata']))['rng_state'])
    indices=np.asarray(indices);np.savez_compressed(D/'reconstructed_minibatch_indices.npz',indices=indices)

    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    agents={name:SACAgent.load(S/'checkpoints'/name/'agent.pt','cuda') for name in ('initialized','online_500','online_1000','online_2000')}
    # Replay all training RNG draws, not gradient updates. Saved means/std reproduce actual actions.
    restore(rng_load(S/'checkpoints/initialized/rng.pt'));row_cursor=0;online_step=0;rng_matches=[]
    for p,t,r in training:
        assert equal(rng(),rng_load(p/'rng_initial.pt')),(p,'episode initial RNG')
        for i in range(7,len(t['actions'])):
            with torch.no_grad():
                mean=torch.as_tensor(t['raw_mean'][i][None],device='cuda');std=torch.as_tensor(t['latent_std'][i][None],device='cuda')
                sampled=torch.distributions.Normal(mean,std).rsample().tanh()[0].cpu().numpy()
            assert equal(sampled,t['actions'][i]),(p,i,'training sample')
            row_cursor+=1
            if r['phase']=='online':
                online_step+=1;idx=indices[499+online_step];bootstrap=int((~replay['terminated'][idx]).sum())
                with torch.no_grad():
                    if bootstrap:
                        torch.distributions.Normal(torch.zeros((bootstrap,16),device='cuda'),torch.ones((bootstrap,16),device='cuda')).rsample()
                        torch.randperm(2,device='cuda')
                    torch.distributions.Normal(torch.zeros((256,16),device='cuda'),torch.ones((256,16),device='cuda')).rsample()
                if online_step in (500,1000,2000):
                    label=f'online_{online_step}';assert equal(rng(),rng_load(S/'checkpoints'/label/'rng.pt')),(label,'global RNG')
                    rng_matches.append(label)
            elif row_cursor==808:
                for label in ('warmup_complete','pretrained'):
                    assert equal(rng(),rng_load(S/'checkpoints'/label/'rng.pt')),(label,'global RNG')
                    rng_matches.append(label)
    assert (row_cursor,online_step)==(2808,2000)
    print('All 2,500 minibatches, 2,808 training actions, episode/checkpoint global RNG states verified.',flush=True)

    evaluation_checks=[];evaluation_episodes=[]
    for label in ('pretraining_baseline','online_500','online_1000','online_2000'):
        name='initialized' if label=='pretraining_baseline' else label;agent=agents[name]
        for p in sorted((S/'evaluations'/label).iterdir()):
            if not p.is_dir():continue
            t,r=episode(p,source,source_phys,config);evaluation_episodes.append((p,t,r));prefix=r['prefix_controls']
            seed=r['policy_sampling_seed'];torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed);random.seed(seed)
            assert equal(rng(),rng_load(p/'rng_initial.pt')),(p,'declared seed')
            for i in range(prefix,len(t['actions'])):
                value=torch.as_tensor(t['obs'][i][None],dtype=torch.float32,device='cuda');assert value.shape==(1,60)
                with torch.no_grad():
                    mean,ls=agent.actor(value)
                    assert equal(mean[0].cpu().numpy(),t['raw_mean'][i])
                    assert equal(ls.exp()[0].cpu().numpy(),t['latent_std'][i])
                    if r['deterministic']:action=mean.tanh();lp=agent.actor._log_prob(mean,ls,mean)
                    else:action,lp=agent.actor.sample(value)
                assert equal(action[0].cpu().numpy(),t['actions'][i]) and float(lp[0])==float(t['log_prob'][i]),(p,i)
            assert equal(rng(),rng_load(p/'rng_final.pt')),(p,'final RNG')
            assert r['learner_counters']==dict(bc_updates=12000,pretrain_updates=agent.pretrain_updates,online_updates=agent.online_updates)
            evaluation_checks.append(dict(label=label,trial=p.name,prefix=prefix,controls=len(t['actions']),suffix_controls=r['policy_suffix_controls'],
                strict_force_success=r['finger_supported_rotation_success'],environment_success=r['environment_success'],drop=r['independent_drop'],
                palm=r['independent_palm_support_violation'],failure=r['environment_failure_reason'],final_force_angle=r['final_force_qualified_angle'],
                sampling_seed=seed,mean_std_action_direct_logp_bitexact=True,initial_and_final_all_rng_bitexact=True))
    assert len(evaluation_checks)==45
    print('All 45 saved evaluations: exact singleton actions, direct log probabilities, and RNG verified.',flush=True)

    calibration=[]
    for label,eps in [('warmup',training[:256]),('heldout',[e for e in evaluation_episodes if e[0].parent.name=='pretraining_baseline' and e[0].name.startswith('phase_stochastic')])]:
        terminal=[e for e in eps if e[2]['terminated'] and not e[2]['truncated']]
        targets=np.concatenate([empirical(t,r,.05/200) for _,t,r in terminal]);pretargets=np.concatenate([empirical(t,r,.02/200) for _,t,r in terminal])
        for stage in ('before','after'):
            z=load(S/f'{label}_calibration_{stage}.npz')
            for k in ('obs','actions','log_prob'):
                assert equal(z[k],np.concatenate([t[k][r['prefix_controls']:] for _,t,r in terminal]))
            assert equal(z['episode_number'],np.concatenate([np.full(r['policy_suffix_controls'],r['episode_number'],np.int64) for _,t,r in terminal]))
            assert np.allclose(targets,z['soft_q_normalized'],atol=1e-14,rtol=0)
            assert np.allclose(pretargets,z['soft_q_pretrain_normalized'],atol=1e-14,rtol=0)
            assert equal(z['soft_q_raw_equivalent'],z['soft_q_normalized']*200)
            assert equal(z['soft_q_pretrain_raw_equivalent'],z['soft_q_pretrain_normalized']*200)
            assert equal(z['q_raw_equivalent'],z['q_normalized']*200)
        calibration.append(dict(role=label,episodes=len(terminal),rows=len(targets),excluded_truncations=sum(e[2]['truncated'] for e in eps),
            forward_sum_current_entropy_excluded=True,alpha_units_verified=[.05/200,.02/200],heldout_never_fit=(label=='heldout')))
    # The final unfinished transition and live environment are preserved without a fabricated terminal.
    final=S/'checkpoints/online_2000';state=read(final/'state.json');env=load(final/'environment.npz');last_t=training[-1][1]
    for k in ['integration_state','qpos','qvel','targets','time','env_initial_position','env_initial_orientation','env_last_action','env_last_q','env_last_p','env__last_object_quaternion','env_scalars_json']:
        assert equal(env[k],last_t[k][-1]),k
    assert equal(env['observation'],last_t['next_obs'][-1])
    assert state['replay_pending_slot']==2807 and state['unfinished_episode'] and not state['environment_scalars']['_episode_over']
    assert state['environment_scalars']==json.loads(str(last_t['env_scalars_json'][-1]))
    for f in ['transitions.npz','physics.npz','rng_initial.pt']:
        assert sha(final/'current_episode'/f)==sha(S/'unfinished_suffix'/f)
    assert not summary['eligible_for_primitive'] and not summary['main_mpc_dataset_eligible'] and not summary['trajectory_exported']
    write(D/'checks.json',dict(status='complete',new_physics_steps=0,new_optimizer_steps=0,verified_manifests=manifests,
        frozen_execution_manifest_sha256=sha(S/'execution_manifest.json'),training=stage_summary,checkpoints=checkpoint_checks,
        replay=dict(rows=2808,completed_episodes=1009,raw_rewards_and_only_executed_suffix_rows_exact=True,all_actual_next_actions_exact=True,
            rewards_independently_recomputed_from_physics=True,
            prefixes_and_evaluations_excluded=True,all_2500_minibatch_indices_and_raw_scaled_reward_means_exact=True,private_rng_final_exact=True),
        training_rng=dict(training_actions_bitexact=2808,episode_initial_states_bitexact=len(training),checkpoint_global_states_exact=rng_matches,
            evaluation_restoration_verified_from_subsequent_training=['pretraining_baseline','online_500','online_1000'],
            final_evaluation_restoration='Source finally block restores saved state; no post-final training/RNG snapshot exists for an independent state comparison.',
            method='Replay saved mean/std singleton Normal draws plus shape-identical update draws; no learning'),
        calibration=calibration,evaluations=evaluation_checks,
        unfinished=dict(actual_controls=8,prefix_controls=7,suffix_controls=1,terminal=False,truncated=False,pending_slot=2807,
            integration_python_environment_observation_exact=True,environment_binary_hash=sha(final/'environment.bin'),automatic_resume_supported=False),
        runtime=dict(device=torch.cuda.get_device_name(),dtype='float32',shape=[1,60],torch=str(torch.__version__),threads=torch.get_num_threads(),
            interop_threads=torch.get_num_interop_threads(),tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,matmul_precision=torch.get_float32_matmul_precision()),
        caveats=['Training successes are assisted and can span policy updates.','Saved final evaluations establish prescribed assisted samples, not cold skill or a robustness rate.',
                 'Checkpoint is complete evidence at an optimizer boundary, not a supported resumable runner.','No new physics or learning was performed by this review.']))
    write(D/'source_bindings.json',dict(execution_manifest=read(S/'execution_manifest.json'),summary_sha256=sha(S/'summary.json'),
        checkpoint_manifests={n:sha(S/'checkpoints'/n/'manifest.json') for n in checkpoint_names},review_script_sha256=sha(__file__)))
    print(json.dumps(dict(status='complete',training_episodes=len(training),evaluations=len(evaluation_checks),new_physics_steps=0,new_optimizer_steps=0)),flush=True)

if __name__=='__main__':main()
