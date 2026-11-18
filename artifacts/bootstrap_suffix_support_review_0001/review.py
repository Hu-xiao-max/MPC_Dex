"""Independent frozen inference/RNG and saved-physics review; no dynamics."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import torch

D=Path(__file__).resolve().parent
S=D.parent/'bootstrap_suffix_support_0001'
sys.path.insert(0,str(S/'source'))
from mpc_dex.learning import SACAgent


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def rng_equal(a,b):
    return torch.equal(a['cpu'],b['cpu']) and len(a['cuda'])==len(b['cuda']) and all(torch.equal(x,y) for x,y in zip(a['cuda'],b['cuda']))


def main():
    manifest=read(S/'manifest.json')
    assert all(sha(S/p)==h for p,h in manifest.items())
    inputs=read(S/'inputs_manifest.json')
    assert all(sha(S/p)==h for p,h in inputs.items())
    protocol=read(S/'protocol.json')
    for target,binding in protocol['source_bindings'].items():
        assert sha(S/target)==sha(binding['source'])==binding['sha256']
    assert sha(S/'checkpoint.pt')=='66681b0f9651b21edb39ef94cb9827e8dcb1ab08249ac7e82010f81c114cf937'
    config=read(S/'config.json');source_config=read(S/'source_config.json')
    assert sha(config['environment']['initial_grasp'])==sha(source_config['environment']['initial_grasp'])
    config['environment']['initial_grasp']=source_config['environment']['initial_grasp']
    assert config==source_config
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    agent=SACAgent.load(S/'checkpoint.pt','cuda')
    assert (agent.obs_dim,agent.action_dim,agent.bc_updates,agent.pretrain_updates,agent.online_updates)==(60,16,12000,0,0)
    assert agent.std_initialized
    checkpoint=torch.load(S/'checkpoint.pt',map_location='cpu',weights_only=True)
    assert all(not checkpoint[name]['state'] for name in ['actor_optimizer','critic_optimizer','alpha_optimizer'])
    before={name:{k:v.detach().cpu().clone() for k,v in getattr(agent,name).state_dict().items()}
            for name in ['actor','critics','target_critics']}
    alpha=agent.log_alpha.detach().cpu().clone()
    source=load(S/'prefix_transitions.npz');source_physics=load(S/'prefix_physics.npz')
    tolerance=config['environment']['palm_support_force_tolerance']
    tilt_limit=np.deg2rad(config['environment']['success_axis_tolerance_degrees'])
    results=[];reconstructed={}
    for experiment in protocol['experiments']:
        deterministic=experiment['deterministic'];seed=experiment['seed']
        label='deterministic' if deterministic else f'stochastic_{seed}'
        p=S/label;trace=load(p/'transitions.npz');phys=load(p/'physics.npz');report=read(p/'report.json')
        initial_rng=torch.load(p/'rng_initial.pt',map_location='cpu',weights_only=True)
        final_rng=torch.load(p/'rng_final.pt',map_location='cpu',weights_only=True)
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        assert rng_equal(initial_rng,{'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()})
        torch.set_rng_state(initial_rng['cpu']);torch.cuda.set_rng_state_all(initial_rng['cuda'])
        predicted_actions=[];predicted_logp=[]
        for i,ob in enumerate(trace['obs']):
            t=torch.as_tensor(ob[None],dtype=torch.float32,device='cuda')
            assert t.shape==(1,60)
            with torch.no_grad():
                mean,log_std=agent.actor(t)
                assert np.array_equal(mean[0].cpu().numpy(),trace['raw_mean'][i])
                assert np.array_equal(log_std[0].exp().cpu().numpy(),trace['latent_std'][i])
                if i<8:
                    action=source['actions'][i].copy();lp=None
                    assert np.isnan(trace['log_prob'][i])
                    assert rng_equal(initial_rng,{'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()})
                elif deterministic:
                    action=mean.tanh()[0].cpu().numpy()
                    lp=float(agent.actor._log_prob(mean,log_std,mean)[0])
                else:
                    action_t,density=agent.actor.sample(t)
                    action=action_t[0].cpu().numpy();lp=float(density[0])
            assert np.array_equal(action,trace['actions'][i])
            if i>=8:assert lp==trace['log_prob'][i]
            predicted_actions.append(action);predicted_logp.append(np.nan if lp is None else lp)
        reconstructed[label+'_actions']=np.asarray(predicted_actions)
        reconstructed[label+'_log_prob']=np.asarray(predicted_logp)
        assert rng_equal(final_rng,{'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()})
        assert np.array_equal(trace['policy_controlled'],np.arange(len(trace['actions']))>=8)
        assert len(trace['actions'])==9 and len(phys['samples'])==450
        assert not trace['terminated'][:-1].any() and trace['terminated'][-1] and not trace['truncated'].any()
        assert np.array_equal(trace['next_obs'][:-1],trace['obs'][1:])
        for key in ['obs','actions','next_obs','rewards','terminated','truncated','qpos','qvel','targets','integration_state']:
            assert np.array_equal(trace[key][:8],source[key][:8]),(label,key)
            if deterministic:assert np.array_equal(trace[key],source[key]),key
        assert np.array_equal(trace['initial_integration_state'],source['initial_integration_state'])
        assert np.array_equal(trace['integration_state'][:,1:24],trace['qpos'])
        assert np.array_equal(trace['integration_state'][:,24:46],trace['qvel'])
        assert np.array_equal(phys['samples'][:400],source_physics['samples'][:400])
        if deterministic:assert np.array_equal(phys['samples'],source_physics['samples'])
        columns={n:i for i,n in enumerate(phys['columns'])};x=phys['samples'];col=lambda n:x[:,columns[n]]
        assert len(x)==50*len(trace['actions'])
        assert np.allclose(col('time'),np.arange(1,451)*.002,rtol=0,atol=2e-15)
        assert np.array_equal(col('time')[49::50],trace['integration_state'][:,0])
        retained=(col('finger_contacts')>=2)&(col('force_bearing_fingers')>=2)&(
            col('axis_tilt_radians')<=tilt_limit)&(col('palm_support_violation')==0)
        force_angle=np.cumsum(np.where((col('twist_increment')<0)|retained,col('twist_increment'),0.))
        assert np.allclose(force_angle,col('force_qualified_angle'),rtol=0,atol=1e-12)
        assert force_angle[-1]>=2*np.pi and retained[-1]
        assert col('raw_angle')[-1]>=2*np.pi and np.max(col('palm_normal_force'))<=tolerance
        assert not col('table_contacts').any() and not col('palm_support_violation').any()
        assert not col('object_applied_wrench_norm').any() and not col('object_applied_generalized_force_norm').any()
        assert report['finger_supported_rotation_success'] and not report['independent_drop']
        assert report['fixed_prefix_assisted'] and not report['cold_start_policy_success'] and not report['sac_policy_success']
        assert report['learner_online_updates']==0
        delta=np.diff(np.r_[0.,col('raw_angle')[49::50]])
        rewards=20*delta-.3;rewards[-1]+=200
        assert np.allclose(rewards,trace['rewards'],rtol=0,atol=1e-10)
        assert len(trace['rewards'][8:])==1
        assert trace['rewards'][-1]==report['suffix_discounted_return']==report['suffix_soft_q_return']
        assert np.allclose(trace['latent_std'][8:],.05,rtol=0,atol=1e-8)
        results.append(dict(label=label,seed=seed,deterministic=deterministic,controls=9,
            physically_recorded_prefix_controls=8,sampled_suffix_controls=0 if deterministic else 1,
            deterministic_suffix_controls=1 if deterministic else 0,physics_steps=450,
            prefix_physics_bitexact_rows=400,full_deterministic_source_trace_bitexact=deterministic,
            sampling_shape=[1,60],direct_sampled_actions_and_log_prob_bitexact=True,
            raw_mean_and_latent_std_bitexact=True,prefix_consumes_no_policy_rng=True,
            initial_seed_rng_and_final_rng_bitexact=True,
            suffix_terminal=True,suffix_actual_reward=float(trace['rewards'][-1]),
            first_suffix_log_prob=float(trace['log_prob'][-1]),
            handoff_force_qualified_angle=float(force_angle[399]),
            final_force_qualified_angle=float(force_angle[-1]),
            final_loaded_fingers=int(col('force_bearing_fingers')[-1]),
            final_axis_tilt=float(col('axis_tilt_radians')[-1]),
            peak_palm_normal_force=float(col('palm_normal_force').max()),
            independently_recomputed_force_success=True,fixed_prefix_assisted_only=True))
    assert all(torch.equal(before[name][k],v.detach().cpu()) for name in before for k,v in getattr(agent,name).state_dict().items())
    assert torch.equal(alpha,agent.log_alpha.detach().cpu())
    assert all(sha(S/p)==h for p,h in manifest.items())
    np.savez_compressed(D/'reconstructed_inference.npz',**reconstructed)
    write(D/'checks.json',dict(status='complete',source_artifact=str(S),source_manifest_sha256=sha(S/'manifest.json'),
        source_manifest_files=len(manifest),input_source_bindings_verified=True,
        actual_checkpoint_sha256=sha(S/'checkpoint.pt'),checkpoint_kind='bootstrap0005 initial BC-origin actor',
        config_difference_only_byte_identical_grasp_path=True,obs_dim=60,bc_updates=12000,
        pretrain_updates=0,online_updates=0,initial_actor_critic_alpha_optimizer_states_empty=True,
        actor_critics_targets_logalpha_unchanged_by_review=True,new_physics_steps=0,new_optimizer_steps=0,
        fixed_seeds=[17041,17042,17043,17044,17045],runtime=dict(device=str(agent.device),
            device_name=torch.cuda.get_device_name(),torch=str(torch.__version__),dtype='float32',
            inference_shape=[1,60],threads=torch.get_num_threads(),interop=torch.get_num_interop_threads(),
            matmul_precision=torch.get_float32_matmul_precision(),matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_tf32=torch.backends.cudnn.allow_tf32),results=results,
        interpretation='Five prescribed assisted suffix samples succeeded; not a robustness rate, cold-reset actor success, or SAC success.'))
    write(D/'source_inputs_manifest.json',{'source_manifest':sha(S/'manifest.json'),
        'source_files':manifest,'review_script':sha(__file__)})
    print(json.dumps(dict(status='complete',manifest_files=len(manifest),trials=len(results),
        assisted_stochastic_successes=5,new_physics_steps=0)))


if __name__=='__main__':main()
