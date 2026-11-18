"""Read-only checks and final report for the single prescribed corrective BC fit."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
D=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def main():
    inputs=json.loads((D/'inputs_manifest.json').read_text())
    assert all(sha(D/path)==digest for path,digest in inputs.items())
    bindings=json.loads((D/'input_bindings.json').read_text())
    assert all(sha(D/path)==item['sha256']==sha(item['source_path']) for path,item in bindings.items())
    report=json.loads((D/'bc/report.json').read_text());assert report['status']=='complete'
    initial=torch.load(D/'bc/initial_fresh.pt',map_location='cpu',weights_only=True)
    summary=[]
    for update in (3000,6000,12000):
        path=D/f'bc/checkpoint_{update:05d}.pt'
        payload=torch.load(path,map_location='cpu',weights_only=True)
        assert payload['bc_updates']==update and payload['pretrain_updates']==payload['online_updates']==0
        for group in ['critics','target_critics']:
            assert all(torch.equal(v,payload[group][name]) for name,v in initial[group].items())
        assert torch.equal(payload['log_alpha'],initial['log_alpha'])
        assert {int(v['step']) for v in payload['bc_optimizer']['state'].values()}=={update}
        for name in ['critic_optimizer','actor_optimizer','alpha_optimizer']:assert not payload[name]['state']
        assert not payload['actor']['log_std_head.weight'].any()
        np.testing.assert_allclose(payload['actor']['log_std_head.bias'].exp().numpy(),.3,rtol=0,atol=2e-8)
        result=next(r for r in report['results'] if r['updates']==update)
        strict=result['strict']
        assert not strict['finger_supported_rotation_success'] and strict['terminated'] and not strict['truncated']
        with np.load(D/f'bc/audit_{update:05d}/transitions.npz') as z:
            n=len(z['actions']);np.testing.assert_array_equal(z['obs'][1:],z['next_obs'][:-1])
            np.testing.assert_array_equal(z['terminated'],np.arange(n)==n-1)
            assert not z['truncated'].any() and n==strict['steps']
        with np.load(D/f'bc/audit_{update:05d}/physics.npz') as z:
            assert z['samples'].shape==(n*50,len(z['columns']))
        assert sha(path)==result['checkpoint_sha256']
        summary.append(dict(bc_updates=update,checkpoint=str(path.relative_to(D)),checkpoint_sha256=sha(path),
            steps=strict['steps'],failure_reason=strict['environment_failure_reason'],
            final_force_qualified_angle=strict['final_force_qualified_angle'],
            peak_force_qualified_angle=strict['max_force_qualified_angle'],
            independent_drop=strict['independent_drop'],palm_support_violation=strict['independent_palm_support_violation'],
            fit={group:{k:v for k,v in metrics.items() if k!='action_max_abs_error_by_row'} for group,metrics in result['fit'].items()}))
    with np.load(D/'bc/sampled_batch_indices.npz') as z:
        batches=z['indices'];assert batches.shape==(12000,256)
        np.testing.assert_array_equal(batches,np.random.default_rng(17001).integers(73,size=(12000,256)))
    with np.load(D/'supervised_pairs.npz') as z:obs=z['obs'].copy();actions=z['actions'].copy();origin=z['origin'].copy()
    with np.load(D/'bc/fit_12000.npz') as z:
        latent=z['latent_targets'].copy();pred=z['predicted_actions'].copy();means=z['means'].copy()
        np.testing.assert_array_equal(obs,z['observations']);np.testing.assert_array_equal(actions,z['actions'])
    cluster=np.array([0,1,12,27,36,50,61]);target_average=latent[cluster].astype(float).mean(0)
    startup=dict(dataset_rows_one_based=(cluster+1).tolist(),label_counts=dict(initial_hold=1,nearby_move=6),
        exact_duplicate_observations=False,
        reset_action_error_max=float(abs(pred[0].astype(float)-actions[0]).max()),
        reset_prediction_to_nominal_move_action_max=float(abs(pred[0].astype(float)-actions[1]).max()),
        reset_to_nominal_move_prediction_max=float(abs(pred[0].astype(float)-pred[1]).max()),
        reset_to_nominal_move_latent_prediction_l2=float(np.linalg.norm(means[0].astype(float)-means[1])),
        reset_latent_error_l2=float(np.linalg.norm(means[0].astype(float)-latent[0])),
        reset_prediction_to_cluster_mean_latent_l2=float(np.linalg.norm(means[0].astype(float)-target_average)),
        per_row_action_error_max=abs(pred[cluster].astype(float)-actions[cluster]).max(1).tolist(),
        interpretation='The one startup hold is poorly fit while six nearby move labels receive similar predictions. Mixed-unit distance and this training outcome support a conditioning/weighting explanation, not a proof of architectural impossibility.')
    # Actual new MPC corrections can have different labels at separated states;
    # preserve this subset separately from inherited correction/noisy-prefix pairs.
    correction=origin=='actual_corrected_native_mpc'
    distance=np.linalg.norm(obs.astype(float)[:,None]-obs.astype(float)[None,:],axis=2)
    pairs=sorted((distance[i,j],i,j) for i in range(73) for j in range(i+1,73) if correction[i] or correction[j])
    nearest=[dict(first_row_one_based=i+1,second_row_one_based=j+1,raw_observation_l2=float(d),
        action_max_abs=float(abs(actions[i].astype(float)-actions[j]).max()),
        latent_l2=float(np.linalg.norm(latent[i].astype(float)-latent[j]))) for d,i,j in pairs[:8]]
    write(D/'startup_fit_diagnosis.json',dict(startup=startup,closest_pairs_involving_actual_new_mpc=nearest))
    result=dict(status='complete',diagnostic_kind='authentic-corrective-feedback-pair-mean-only-BC',
        source_dataset_pairs=73,source_common_pairs=12,noisy_prefix_pairs=40,actual_new_mpc_pairs=21,
        omitted_noisy_reset_labels=5,bc_updates=12000,critic_updates=0,online_updates=0,
        fresh_seed=17001,batch_size=256,bc_lr=.003,diagnostic_bc_epsilon=.01,main_bc_epsilon=.02,
        diagnostic_clipping_identity=True,main_clipping_changed_scalars=99,
        sampled_batches_verified=True,critics_targets_alpha_unchanged=True,
        fresh_initial_actor_matches_common_seed_bitexact=True,std_at_checkpoints=.30000001192092896,
        source_inputs_unchanged=True,head_polish_updates=0,extra_epochs=0,
        strict_checkpoint_evaluations=3,successful_strict_evaluations=0,conditional_continuations=0,
        main_demonstrations=0,production_eligible=False,sac_success_claim=False,
        actual_wall_seconds=report['wall_seconds'],results=summary,startup_diagnosis=startup,
        dataset_sha256=sha(D/'supervised_pairs.npz'),protocol_sha256=sha(D/'protocol.json'),
        final_checkpoint_sha256=sha(D/'bc/checkpoint_12000.pt'),source_training_script_sha256=sha(D/'train.py'))
    write(D/'summary.json',result)
    findings='''# One fixed corrective BC diagnostic

The fresh mean-only BC fit used 73 authentic observation/action pairs: the original common-continuation 12, forty noisy prefix rows and twenty-one actual MPC correction rows from five independently successful physical recoveries. Each noisy first action was excluded because it shares the same reset observation with a different randomized label. The original nominal hold remains once. All retained labels are unchanged float32 records with source row identities; all source manifests and independent transition/physics replays verify. There are no exact duplicate observations or exact conflicting labels, and no invented replay links.

The local diagnostic epsilon is 0.01, explicitly authorized to preserve every measured action. The main method's 0.02 epsilon is unchanged; applying it here would alter 99 scalars in 31 rows by up to 1.490116e-6. No hidden clipping or relabeling was used.

One fresh seed17001 run used two 256-unit hidden layers, mean-only Adam at0.003, batch256 uniformly with replacement, and exactly12,000 updates. Actual CUDA singleton audits at3,000/6,000/12,000 used one thread, TF32 off, the unchanged strict40mm task and latent std0.3. The fresh initial actor exactly matches the earlier common-run initialization. Critic, target-critic and temperature tensors are unchanged; all critic and online update counts are zero. Sampled batches and optimizer step counters verify.

All three physical checkpoint audits failed: palm support at control4 (peak force-qualified angle4.174268823rad), palm support at control3 (peak1.290391309rad), and drop at control5 (peak1.392583691rad). No conditional post-success continuation ran. The final action fit MSE is0.00249010886, with maximum error0.841766983 on the original reset hold. Nearby step2 labels have about0.068 maximum action error. The inherited startup hold-versus-motion distinction remains hard for this fit, and adding five nearby move labels gives six nearby move records for one hold record. This diagnostic supplies authentic feedback data but does not establish a robust learned policy.

No head correction, extra epochs, SAC, new successful learned trajectory, or main-stage demonstration claim was made. Per-group fit, startup predictions, all actual failed transitions/2ms physics, optimizer checkpoints and exact input/source hashes are retained.
'''
    (D/'findings.md').write_text(findings)
    write(D/'manifest.json',{str(p.relative_to(D)):sha(p) for p in sorted(D.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p.name!='manifest.json'})
    print(json.dumps(dict(summary_sha256=sha(D/'summary.json'),checkpoint_sha256=result['final_checkpoint_sha256'],startup=startup),indent=2))

if __name__=='__main__':main()
