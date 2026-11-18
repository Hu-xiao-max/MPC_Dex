"""Freeze authentic corrective observation/action pairs; no physical execution."""
from pathlib import Path
import hashlib,json,shutil,sys
import numpy as np
import torch

D=Path(__file__).resolve().parent
ROOT=D.parents[1]
COMMON=ROOT/'artifacts/bc_common_continuation_0001'
RECOVERY=ROOT/'artifacts/mpc_common_noise_recovery_0001'
SEEDS=list(range(17021,17026))

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def arrays(path):
    with np.load(path,allow_pickle=False) as data:return {k:v.copy() for k,v in data.items()}

def main():
    source_checks={}
    for label,root in [('common',COMMON),('recovery',RECOVERY)]:
        manifest=json.loads((root/'manifest.json').read_text())
        assert all((root/name).is_file() and sha(root/name)==digest for name,digest in manifest.items())
        source_checks[label]=dict(root=str(root),manifest_sha256=sha(root/'manifest.json'),verified_files=len(manifest))
    assert source_checks['recovery']['manifest_sha256']=='be1a96b6445f86d1891159b9723d5cff0831b8ba34d694f4b991afb354d14e46'
    bindings={}
    def copy(path,relative):
        dest=D/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        digest=sha(path);shutil.copyfile(path,dest);assert digest==sha(path)==sha(dest)
        bindings[relative]=dict(source_path=str(path),sha256=digest)
    shutil.copytree(COMMON/'source',D/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for p in sorted((COMMON/'source').rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':
            rel='source/'+str(p.relative_to(COMMON/'source'));assert sha(p)==sha(D/rel)
            bindings[rel]=dict(source_path=str(p),sha256=sha(p))
    for name in ['manifest.json','inputs_manifest.json','dataset_validation.json','supervised_pairs.npz',
                 'train.py','audit.py','strict_config.json','checkpoint_state.json','bc/initial_fresh.pt']:
        copy(COMMON/name,'inputs/common/'+name)
    copy(COMMON/'audit.py','audit.py')
    for branch in ['nominal','recovery']:
        for name in ['transitions.npz','physics.npz','report.json']:
            copy(COMMON/f'inputs/{branch}/strict/{name}',f'inputs/common/{branch}/{name}')
    copy(COMMON/'inputs/recovery_origins.npz','inputs/common/recovery_origins.npz')
    for name in ['manifest.json','final_report.json','summary.json','protocol.json','config.json']:
        copy(RECOVERY/name,'inputs/recovery/'+name)
    for seed in SEEDS:
        for name in ['transitions.npz','physics.npz','control_origins.npz','report.json',
                     'independent_audit/transitions.npz','independent_audit/physics.npz','independent_audit/report.json']:
            copy(RECOVERY/f'seed_{seed}'/name,f'inputs/recovery/seed_{seed}/{name}')
        copy(RECOVERY/f'inputs/noisy_prefixes/seed_{seed}/transitions.npz',f'inputs/noisy_prefix_{seed}.npz')
    sys.path.insert(0,str(D/'source'))
    from mpc_dex.env import EnvConfig
    from mpc_dex.contracts import environment_contract
    config=json.loads((COMMON/'strict_config.json').read_text())
    config['environment']['initial_grasp']=str(D/'source/assets/initial_grasp_40mm.npz')
    write(D/'strict_config.json',config);ec=EnvConfig(**config['environment']);contract=environment_contract(ec)
    common=arrays(D/'inputs/common/supervised_pairs.npz')
    nominal=arrays(D/'inputs/common/nominal/transitions.npz');old_recovery=arrays(D/'inputs/common/recovery/transitions.npz')
    for key in ['obs','actions']:
        np.testing.assert_array_equal(common[key],np.concatenate([nominal[key],old_recovery[key][6:]]))
    np.testing.assert_array_equal(nominal['actions'][6:],old_recovery['actions'][6:])
    old_origins=arrays(D/'inputs/common/recovery_origins.npz')
    np.testing.assert_array_equal(np.flatnonzero(old_origins['actual_mpc_correction_mask']),np.arange(6,9))
    observations=[common['obs']];actions=[common['actions']]
    provenance=[];source_episodes=[];excluded=[]
    def provenance_rows(label,path,rows,groups):
        for row,group in zip(rows,groups):
            provenance.append(dict(dataset_row_one_based=len(provenance)+1,source_label=label,
                source_path=path,source_sha256=sha(D/path),source_row_one_based=int(row)+1,origin=group))
    provenance_rows('common_nominal','inputs/common/nominal/transitions.npz',range(9),
                    ['original_teacher_prefix']*6+['nominal_mpc_correction']*3)
    provenance_rows('common_actor_recovery','inputs/common/recovery/transitions.npz',range(6,9),
                    ['actual_actor_state_mpc_correction']*3)
    for seed in SEEDS:
        base=D/f'inputs/recovery/seed_{seed}';data=arrays(base/'transitions.npz')
        audit=arrays(base/'independent_audit/transitions.npz');n=len(data['actions'])
        assert set(data)==set(audit)
        for key in data:np.testing.assert_array_equal(data[key],audit[key])
        for key,value in arrays(base/'physics.npz').items():
            np.testing.assert_array_equal(value,arrays(base/'independent_audit/physics.npz')[key])
        prefix=arrays(D/f'inputs/noisy_prefix_{seed}.npz')
        for key in ['obs','actions','next_obs','rewards','qpos','qvel','terminated','truncated','targets']:
            np.testing.assert_array_equal(data[key][:9],prefix[key])
        np.testing.assert_array_equal(data['obs'][0],nominal['obs'][0])
        np.testing.assert_array_equal(data['obs'][1:],data['next_obs'][:-1])
        np.testing.assert_array_equal(data['terminated'],np.arange(n)==n-1)
        assert not data['truncated'].any()
        report=json.loads((base/'report.json').read_text());r=json.loads((base/'independent_audit/report.json').read_text())
        assert report['environment_contract']==contract
        assert all(r[name] is True for name in ['finger_supported_rotation_success','strict_force_success',
                                              'environment_success','no_external_support','terminated'])
        assert not any(r[name] for name in ['independent_drop','environment_drop','independent_palm_support_violation','truncated'])
        assert r['steps']==n and r['final_force_qualified_angle']>=2*np.pi
        origins=arrays(base/'control_origins.npz')
        np.testing.assert_array_equal(origins['expert_correction_mask'],np.arange(n)>=9)
        assert np.all(origins['origin'][:9]=='recorded_fixed_mean_noisy_prefix')
        assert np.all(origins['origin'][9:]=='actual_corrected_native_mpc')
        observations.append(data['obs'][1:]);actions.append(data['actions'][1:])
        provenance_rows(f'recovery_{seed}',f'inputs/recovery/seed_{seed}/transitions.npz',range(1,n),origins['origin'][1:].tolist())
        excluded.append(dict(seed=seed,row_one_based=1,reason='Identical cold-reset observation with random perturbed first-action label; original nominal hold retained once',
            action_max_difference_from_nominal=float(abs(data['actions'][0]-nominal['actions'][0]).max())))
        source_episodes.append(dict(seed=seed,full_recorded_rows=n,included_prefix_rows=8,included_mpc_rows=n-9,
            final_force_qualified_angle=r['final_force_qualified_angle'],strict_force_success=True,
            independent_transition_and_physics_bitexact=True,post_success_hold_executed=False))
    obs=np.concatenate(observations);actions=np.concatenate(actions)
    assert obs.shape==(73,60) and actions.shape==(73,16) and len(provenance)==73
    t=torch.as_tensor(actions);old_clamped=t.clamp(-.98,.98);new_clamped=t.clamp(-.99,.99)
    assert torch.equal(t,new_clamped)
    raw_latent=torch.atanh(t).numpy();old_latent=torch.atanh(old_clamped).numpy()
    delta=obs.astype(float)[:,None]-obs.astype(float)[None,:]
    distances=np.linalg.norm(delta,axis=2)
    action_delta=actions.astype(float)[:,None]-actions.astype(float)[None,:]
    latent_delta=raw_latent.astype(float)[:,None]-raw_latent.astype(float)[None,:]
    pairs=np.array([(i,j) for i in range(73) for j in range(i+1,73)])
    duplicate=pairs[distances[pairs[:,0],pairs[:,1]]==0]
    conflicts=[(int(i),int(j)) for i,j in duplicate if not np.array_equal(actions[i],actions[j])]
    assert not len(duplicate) and not conflicts
    slices=[('q',0,16),('position',16,19),('quaternion',19,23),('angles',23,25),('qvel',25,41),('object_velocity',41,44),('previous_action',44,60)]
    def pair_record(i,j):
        d=delta[i,j];a=action_delta[i,j];z=latent_delta[i,j]
        return dict(first=int(i)+1,second=int(j)+1,first_source=provenance[i],second_source=provenance[j],
            observation_l2=float(np.linalg.norm(d)),observation_max_abs=float(abs(d).max()),
            action_l2=float(np.linalg.norm(a)),action_max_abs=float(abs(a).max()),
            latent_l2=float(np.linalg.norm(z)),latent_max_abs=float(abs(z).max()),
            latent_l2_per_raw_observation_l2=float(np.linalg.norm(z)/np.linalg.norm(d)),
            observation_groups={name:dict(l2=float(np.linalg.norm(d[start:end])),max_abs=float(abs(d[start:end]).max())) for name,start,end in slices})
    pair_dist=distances[pairs[:,0],pairs[:,1]]
    slopes=np.linalg.norm(latent_delta[pairs[:,0],pairs[:,1]],axis=1)/pair_dist
    correction=np.array(['mpc' in p['origin'] for p in provenance]);new=np.arange(73)>=12
    cross=pairs[(correction[pairs[:,0]]|correction[pairs[:,1]]) & (new[pairs[:,0]]|new[pairs[:,1]])]
    cross=cross[np.argsort(distances[cross[:,0],cross[:,1]])]
    changed=(t!=old_clamped).numpy()
    write(D/'row_provenance.json',dict(rows=provenance,excluded=excluded,
        no_relabeling=True,no_interpolation=True,no_synthetic_episode_links=True))
    np.savez_compressed(D/'supervised_pairs.npz',obs=obs,actions=actions,
        source_label=np.asarray([p['source_label'] for p in provenance]),
        source_row_one_based=np.asarray([p['source_row_one_based'] for p in provenance]),
        origin=np.asarray([p['origin'] for p in provenance]))
    np.savez_compressed(D/'conditioning_arrays.npz',observation_distances=distances,actions=actions,
        raw_latent_targets=raw_latent,epsilon002_latent_targets=old_latent,
        epsilon002_clamped_actions=old_clamped.numpy(),epsilon002_changed=changed)
    report=dict(status='complete',kind='diagnostic-supervised-corrective-pairs',pairs=73,
        original_common_pairs=12,new_noisy_prefix_pairs=40,new_actual_mpc_correction_pairs=21,
        excluded_identical_reset_noisy_labels=5,exact_duplicate_observation_pairs=0,exact_label_conflicts=0,
        labels_are_original_float32=True,deduplicated_rows=0,production_eligible=False,main_demonstrations=0,
        sac_updates=0,policy_only_success_claim=False,source_manifest_checks=source_checks,
        sources=source_episodes,source_rows_bitexact=True,environment_contract=contract,
        bc_clipping=dict(source_main_epsilon=.02,diagnostic_epsilon=.01,action_abs_max=float(abs(actions).max()),
            source_changed_scalars=int(changed.sum()),source_changed_rows=int(changed.any(1).sum()),
            source_max_action_change=float(abs(t-old_clamped).max()),diagnostic_changed_scalars=0,
            diagnostic_identity_verified=True,reason='Preserve all authentic noisy labels; source .02 clamp alters recorded commands'),
        conditioning=dict(distance_convention='Unscaled mixed-unit observation norms are descriptive, not causal physical sensitivities',
            closest_pairs=[pair_record(*pair) for pair in pairs[np.argsort(pair_dist)[:12]]],
            largest_latent_slopes=[pair_record(*pair) for pair in pairs[np.argsort(slopes)[-5:][::-1]]],
            closest_new_correction_pairs=[pair_record(*pair) for pair in cross[:8]],
            inherited_startup_pair=pair_record(0,1)),
        remaining_issue='Inherited near-stationary startup hold-versus-motion labels remain; no exact contradiction. New noisy feedback states are authentic supervised coverage, not a robustness guarantee.')
    write(D/'dataset_validation.json',report)
    protocol=dict(status='prepared',purpose='One fresh mean-only BC fit on authentic feedback-state corrections; no SAC/primitive claim',
        seed=17001,updates=12000,bc_lr=.003,bc_action_epsilon=.01,source_main_bc_action_epsilon=.02,
        batch_size=256,sampling='Uniform with replacement over all73 pairs',hidden_sizes=[256,256],
        diagnostic_unused_critics=2,critic_updates=0,online_updates=0,initial_std=.3,
        evaluation_updates=[3000,6000,12000],evaluation_seed=42,evaluation_max_controls=100,
        evaluation_if_success='One same-checkpoint deterministic closed-loop continuation for2s using fresh policy actions',
        head_polish_allowed=False,additional_epochs_allowed=False,device='cuda',torch_threads=1,torch_interop_threads=1,
        matmul_tf32=False,cudnn_tf32=False,source_main_parameters_unchanged=True,
        dataset_sha256=sha(D/'supervised_pairs.npz'),dataset_validation_sha256=sha(D/'dataset_validation.json'))
    write(D/'protocol.json',protocol)
    for relative,binding in bindings.items():assert sha(D/relative)==binding['sha256']==sha(binding['source_path'])
    write(D/'input_bindings.json',bindings)
    write(D/'inputs_manifest.json',{str(p.relative_to(D)):sha(p) for p in sorted(D.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p.name!='inputs_manifest.json'})
    print(json.dumps(dict(pairs=73,clipping=report['bc_clipping'],largest_latent_slopes=report['conditioning']['largest_latent_slopes'][:2]),indent=2))

if __name__=='__main__':main()
