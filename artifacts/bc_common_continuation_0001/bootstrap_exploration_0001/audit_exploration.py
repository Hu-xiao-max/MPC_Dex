"""Bounded independent validation and five prespecified stochastic audits.

No optimizer steps, mean changes, action clipping, or standard-deviation sweep.
The source policy is BC plus one diagnostic mean-head solve, not trained SAC.
"""
from dataclasses import asdict,replace
import copy
from datetime import datetime,timezone
import hashlib,json,math,os
from pathlib import Path
import random,shutil,sys

F=Path('/home/autel001/code/mpc_dex/artifacts/bc_common_continuation_0001')
D=F/'bootstrap_exploration_0001'
ROOT=F.parents[1]


def hash_file(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source_manifest=json.loads((F/'manifest.json').read_text())
    source_inputs=json.loads((F/'inputs_manifest.json').read_text())
    # The finalized parent manifest lists relative files and their hashes.
    for manifest in (source_manifest,source_inputs):
        assert all(hash_file(F/path)==digest for path,digest in manifest.items())
    polished_report=json.loads((F/'head_polish/report.json').read_text())
    assert polished_report['status']=='complete'
    assert hash_file(F/'head_polish/polished.pt')==polished_report['polished_sha256']
    assert hash_file(F/'polish.py')==polished_report['source_script_sha256']
    current={}
    for relative in ('mpc_dex/env.py','mpc_dex/learning.py','mpc_dex/bootstrap_env.py',
                     'mpc_dex/contracts.py','scripts/audit_policy.py'):
        current[relative]=hash_file(ROOT/relative)
        assert current[relative]==hash_file(F/'source'/relative),relative
    D.mkdir(exist_ok=False)
    shutil.copyfile(__file__,D/'audit_exploration.py')
    shutil.copytree(F/'source',D/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for relative,target in [('head_polish/polished.pt','source_policy.pt'),
            ('head_polish/polished.pt.metadata.json','source_policy.pt.metadata.json'),
            ('head_polish/baseline.pt','source_baseline.pt'),
            ('head_polish/report.json','source_polish_report.json'),
            ('strict_config.json','source_strict_config.json'),
            ('manifest.json','source_manifest.json'),('inputs_manifest.json','source_inputs_manifest.json')]:
        shutil.copyfile(F/relative,D/target)
    sys.path.insert(0,str(D/'source'))
    import numpy as np,torch,mujoco
    from mpc_dex.env import EnvConfig,AllegroRotationEnv
    from mpc_dex.learning import SACAgent
    from mpc_dex.contracts import environment_contract,validate_checkpoint
    from scripts.audit_policy import audit_episode,PHYSICS_COLUMNS,write_json
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.is_available()
    configuration=json.loads((D/'source_strict_config.json').read_text())
    original_grasp=configuration['environment']['initial_grasp']
    configuration['environment']['initial_grasp']=str(D/'source/assets/initial_grasp_40mm.npz')
    assert hash_file(original_grasp)==hash_file(configuration['environment']['initial_grasp'])
    write_json(D/'config.json',configuration)
    ec=EnvConfig(**configuration['environment'])
    assert ec.terminate_on_success and ec.object_diameter==.04 and ec.horizon_steps==100
    assert configuration.get('bootstrap',{}).get('control') is None
    checkpoint_record=validate_checkpoint(D/'source_policy.pt',ec)
    assert checkpoint_record['policy_interface']['kind']=='absolute_joint_target'
    source_agent=SACAgent.load(D/'source_policy.pt','cuda')
    assert source_agent.bc_updates==12000 and source_agent.pretrain_updates==source_agent.online_updates==0
    def differences(a,b,path='checkpoint'):
        if type(a) is not type(b):return [path+'[type]']
        if isinstance(a,torch.Tensor):
            return [] if a.dtype==b.dtype and a.shape==b.shape and a.cpu().numpy().tobytes()==b.cpu().numpy().tobytes() else [path]
        if isinstance(a,dict):
            assert a.keys()==b.keys();return [p for k in a for p in differences(a[k],b[k],path+'/'+str(k))]
        if isinstance(a,(list,tuple)):
            assert len(a)==len(b);return [p for i,(x,y) in enumerate(zip(a,b)) for p in differences(x,y,path+'/'+str(i))]
        return [] if a==b else [path]
    original_payload=torch.load(D/'source_policy.pt',map_location='cpu',weights_only=True)
    baseline_payload=torch.load(D/'source_baseline.pt',map_location='cpu',weights_only=True)
    source_changes=differences(baseline_payload,original_payload)
    assert set(source_changes)=={'checkpoint/actor/mean_head.weight','checkpoint/actor/mean_head.bias'}
    runtime=dict(created_at=datetime.now(timezone.utc).isoformat(),device='cuda',torch_version=str(torch.__version__),
        numpy_version=np.__version__,mujoco_version=mujoco.__version__,cuda_version=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),gpu_capability=list(torch.cuda.get_device_capability()),
        torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads(),
        cuda_matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        actor_inference='singleton float32 CUDA SACAgent.act, no extra clipping or smoothing',
        current_numerical_source_sha256=current,source_checkpoint_sha256=hash_file(D/'source_policy.pt'),
        source_checkpoint_stage=checkpoint_record['stage'],source_payload_changed_from_baseline=source_changes,
        verified_parent_manifest_files=len(source_manifest),verified_parent_input_files=len(source_inputs),
        config_transformation=dict(initial_grasp_path_only=dict(original=original_grasp,copy=ec.initial_grasp),
                                   initial_grasp_sha256=hash_file(ec.initial_grasp)),
        environment_contract=environment_contract(ec))
    write_json(D/'runtime.json',runtime)
    def read_npz(path):
        with np.load(path,allow_pickle=False) as archive:return {k:archive[k].copy() for k in archive.files}
    def run_audit(agent,config,output,limit,seed,deterministic):
        output.mkdir()
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed);random.seed(seed)
        torch.save(dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()),output/'torch_rng_initial.pt')
        env=AllegroRotationEnv(config)
        initial={};records={key:[] for key in ['obs','actions','conditional_mean_actions','latent_std','next_obs','rewards','terminated','truncated','qpos','qvel','targets','integration_state']}
        original_reset,original_step=env.reset,env.step
        def integration():
            spec=mujoco.mjtState.mjSTATE_INTEGRATION
            values=np.empty(mujoco.mj_stateSize(env.model,spec));mujoco.mj_getState(env.model,env.data,values,spec)
            return values
        def reset(*args,**kwargs):
            result=original_reset(*args,**kwargs)
            initial.update(qpos=env.data.qpos.copy(),qvel=env.data.qvel.copy(),ctrl=env.data.ctrl.copy(),
                joint_positions=env.data.qpos[env.q_ids].copy(),integration_state=integration())
            return result
        def step(action):
            result=original_step(action);ob,reward,done,truncated,info=result
            for key,value in dict(next_obs=ob.copy(),rewards=reward,terminated=done,truncated=truncated,
                qpos=env.data.qpos.copy(),qvel=env.data.qvel.copy(),targets=env.data.ctrl.copy(),integration_state=integration()).items():records[key].append(value)
            return result
        class RecordingPolicy:
            def act(self,obs,deterministic=True):
                with torch.no_grad():
                    mean,log_std=agent.actor(torch.as_tensor(obs,device=agent.device,dtype=torch.float32))
                action=agent.act(obs,deterministic=deterministic)
                records['obs'].append(obs.copy());records['actions'].append(action.copy())
                records['conditional_mean_actions'].append(mean.tanh().cpu().numpy().copy())
                records['latent_std'].append(log_std.exp().cpu().numpy().copy())
                return action
        env.reset,env.step=reset,step
        try:
            result,physics,commands=audit_episode(env,RecordingPolicy(),seed,limit,deterministic)
            columns={name:physics[:,i] for i,name in enumerate(PHYSICS_COLUMNS)}
            result.update(policy_kind='diagnostic BC plus one mean-head correction; no SAC updates',
                production_eligible=False,sac_success=False,stochastic=not deterministic,
                environment=asdict(config),environment_contract=environment_contract(config),
                requested_control_limit=limit,model_equality_constraints=int(env.model.neq),
                max_sampled_action_abs=float(np.abs(records['actions']).max()),
                latent_std_min=float(np.min(records['latent_std'])),latent_std_max=float(np.max(records['latent_std'])),
                failure_checks=dict(max_contactless_seconds=float(columns['contactless_seconds'].max()),
                    max_displacement=float(columns['object_displacement'].max()),max_table_contacts=int(columns['table_contacts'].max())))
            np.savez_compressed(output/'transitions.npz',**{k:np.asarray(v) for k,v in records.items()},**{'initial_'+k:v for k,v in initial.items()})
            np.savez_compressed(output/'physics.npz',samples=physics,columns=np.asarray(PHYSICS_COLUMNS))
            write_json(output/'commands.json',commands);write_json(output/'report.json',result)
            torch.save(dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()),output/'torch_rng_final.pt')
            return result
        finally:env.close()
    strict=run_audit(source_agent,ec,D/'deterministic_strict',100,42,True)
    assert strict['finger_supported_rotation_success']
    strict_original=F/'head_polish/independent_audit'
    continuous_original=F/'head_polish/independent_audit_continuous_policy_2s'
    continuation=run_audit(source_agent,replace(ec,terminate_on_success=False),D/'deterministic_continued_policy',strict['steps']+20,42,True)
    for ours,original in [(D/'deterministic_strict',strict_original),(D/'deterministic_continued_policy',continuous_original)]:
        new,old=read_npz(ours/'transitions.npz'),read_npz(original/'transitions.npz')
        for key in old:np.testing.assert_array_equal(new[key],old[key],err_msg=key)
        new,old=read_npz(ours/'physics.npz'),read_npz(original/'physics.npz')
        for key in old:np.testing.assert_array_equal(new[key],old[key],err_msg=key)
    assert continuation['independent_drop'] and continuation['steps']==12
    # All changes to the exploration policy are confined to this in-memory copy.
    exploratory=copy.deepcopy(source_agent)
    before={k:v.detach().cpu().clone() for k,v in exploratory.actor.state_dict().items()}
    with torch.no_grad():
        exploratory.actor.log_std_head.weight.zero_()
        exploratory.actor.log_std_head.bias.fill_(math.log(.05))
    after={k:v.detach().cpu().clone() for k,v in exploratory.actor.state_dict().items()}
    changed=[key for key in before if not torch.equal(before[key],after[key])]
    assert changed==['log_std_head.bias'],changed
    assert all(torch.equal(v,source_agent.actor.state_dict()[key].detach().cpu()) for key,v in before.items())
    report=dict(status='running',purpose='Five prespecified stochastic exploration samples from independently verified BC-plus-refined-head mean; no learning',
        production_eligible=False,sac_success_claim=False,robustness_rate_claim=False,
        requested_latent_std=.05,std_source='Independent bootstrap runner default, not a published companion setting',
        std_override_in_memory_only=True,modified_checkpoint_saved=False,changed_actor_parameters=changed,
        mean_and_trunk_bitwise_unchanged=True,original_source_unchanged=True,
        prespecified_seeds=[17021,17022,17023,17024,17025],strict_control_limit=100,
        source_bc_updates=source_agent.bc_updates,source_critic_updates=0,source_online_updates=0,
        updates_this_diagnostic=0,deterministic_strict_bitexact=True,deterministic_continuation_bitexact=True,
        deterministic_strict=strict,deterministic_continued_policy=continuation,stochastic_results=[])
    write_json(D/'report.json',report)
    for seed in report['prespecified_seeds']:
        result=run_audit(exploratory,ec,D/f'stochastic_seed_{seed}',100,seed,False)
        report['stochastic_results'].append(result);write_json(D/'report.json',report)
        print(json.dumps(dict(seed=seed,steps=result['steps'],success=result['finger_supported_rotation_success'],
            drop=result['independent_drop'],palm=result['independent_palm_support_violation'],
            max_force_qualified_angle=result['max_force_qualified_angle'])),flush=True)
    assert exploratory.bc_updates==12000 and exploratory.pretrain_updates==exploratory.online_updates==0
    for key,value in exploratory.actor.state_dict().items():assert torch.equal(value.detach().cpu(),after[key]),key
    for key,value in source_agent.actor.state_dict().items():assert torch.equal(value.detach().cpu(),before[key]),key
    assert not differences(source_agent.critics.state_dict(),exploratory.critics.state_dict(),'critics')
    assert not differences(source_agent.target_critics.state_dict(),exploratory.target_critics.state_dict(),'target_critics')
    assert torch.equal(source_agent.log_alpha,exploratory.log_alpha)
    assert hash_file(F/'head_polish/polished.pt')==runtime['source_checkpoint_sha256']
    assert all(hash_file(F/path)==digest for path,digest in source_manifest.items())
    results=report['stochastic_results']
    report.update(status='complete',sample_count=len(results),sample_full_turns=sum(x['finger_supported_rotation_success'] for x in results),
        sample_drops=sum(x['independent_drop'] for x in results),sample_palm_violations=sum(x['independent_palm_support_violation'] for x in results),
        interpretation='Five fixed-seed diagnostic samples, not a robustness estimate or learned SAC success')
    write_json(D/'report.json',report)
    write_json(D/'manifest.json',{str(p.relative_to(D)):hash_file(p) for p in sorted(D.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p.name not in ('manifest.json','run.log')})


if __name__=='__main__':main()
