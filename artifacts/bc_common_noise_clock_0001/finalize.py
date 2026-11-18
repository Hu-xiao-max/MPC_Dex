"""Read-only final checks for the single prescribed61D elapsed-time BC fit."""
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
    previous=D.parent/'bc_common_noise_corrective_0001'
    previous_manifest=json.loads((D/'inputs/previous73/manifest.json').read_text())
    assert all(sha(previous/path)==digest for path,digest in previous_manifest.items())
    report=json.loads((D/'bc/report.json').read_text());assert report['status']=='complete'
    initial=torch.load(D/'bc/initial_fresh.pt',map_location='cpu',weights_only=True)
    summary=[]
    for update in (3000,6000,12000):
        path=D/f'bc/checkpoint_{update:05d}.pt'
        payload=torch.load(path,map_location='cpu',weights_only=True)
        assert payload['obs_dim']==61 and payload['action_dim']==16
        metadata=json.loads(Path(str(path)+'.metadata.json').read_text())
        assert metadata['policy_interface']==json.loads((D/'policy_interface.json').read_text())
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
        assert strict['policy_kind']=='diagnostic-common-noise-clock-bc-only'
        final_audit=json.loads((D/f'bc/audit_{update:05d}/report.json').read_text())
        assert final_audit['policy_kind']==strict['policy_kind']
        assert not strict['finger_supported_rotation_success'] and strict['terminated'] and not strict['truncated']
        with np.load(D/f'bc/audit_{update:05d}/transitions.npz') as z:
            n=len(z['actions']);np.testing.assert_array_equal(z['obs'][1:],z['next_obs'][:-1])
            np.testing.assert_array_equal(z['terminated'],np.arange(n)==n-1)
            assert not z['truncated'].any() and n==strict['steps']
            assert z['obs'].shape==(n,61) and z['next_obs'].shape==(n,61)
            np.testing.assert_array_equal(z['obs'][:,:60],z['environment_obs'])
            np.testing.assert_array_equal(z['next_obs'][:,:60],z['environment_next_obs'])
            np.testing.assert_array_equal(z['obs'][:,60],z['decision_times'].astype(np.float32))
            np.testing.assert_array_equal(z['next_obs'][:,60],z['control_times'][1:].astype(np.float32))
            np.testing.assert_array_equal(z['decision_times'],z['control_times'][:-1])
            np.testing.assert_allclose(z['control_times'],np.arange(n+1)*.1,rtol=0,atol=1e-12)
            assert z['obs'][0,60]==0.
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
        interpretation='Clock input permits separate startup hold and move predictions, sharply improving the inherited gross fit error. Remaining milliscale imitation errors and failed physical rollouts prevent a policy-success or robustness claim.')
    # Actual new MPC corrections can have different labels at separated states;
    # preserve this subset separately from inherited correction/noisy-prefix pairs.
    correction=origin=='actual_corrected_native_mpc'
    distance=np.linalg.norm(obs.astype(float)[:,None]-obs.astype(float)[None,:],axis=2)
    pairs=sorted((distance[i,j],i,j) for i in range(73) for j in range(i+1,73) if correction[i] or correction[j])
    nearest=[dict(first_row_one_based=i+1,second_row_one_based=j+1,raw_observation_l2=float(d),
        action_max_abs=float(abs(actions[i].astype(float)-actions[j]).max()),
        latent_l2=float(np.linalg.norm(latent[i].astype(float)-latent[j]))) for d,i,j in pairs[:8]]
    write(D/'startup_fit_diagnosis.json',dict(startup=startup,closest_pairs_involving_actual_new_mpc=nearest))
    result=dict(status='complete',diagnostic_kind='authentic-corrective-feedback-pair-clock-mean-only-BC',
        policy_interface=json.loads((D/'policy_interface.json').read_text()),observation_dimension=61,
        actual_clock_verified=True,architecture_initialization_changed=True,pure_causal_estimate=False,
        source_dataset_pairs=73,source_common_pairs=12,noisy_prefix_pairs=40,actual_new_mpc_pairs=21,
        omitted_noisy_reset_labels=5,bc_updates=12000,critic_updates=0,online_updates=0,
        fresh_seed=17001,batch_size=256,bc_lr=.003,diagnostic_bc_epsilon=.01,main_bc_epsilon=.02,
        diagnostic_clipping_identity=True,main_clipping_changed_scalars=99,
        sampled_batches_verified=True,critics_targets_alpha_unchanged=True,
        fresh_initial_actor=True,imported60D_weights=False,std_at_checkpoints=.30000001192092896,
        source_inputs_unchanged=True,head_polish_updates=0,extra_epochs=0,
        strict_checkpoint_evaluations=3,successful_strict_evaluations=0,conditional_continuations=0,
        main_demonstrations=0,production_eligible=False,sac_success_claim=False,
        actual_wall_seconds=report['wall_seconds'],results=summary,startup_diagnosis=startup,
        dataset_sha256=sha(D/'supervised_pairs.npz'),protocol_sha256=sha(D/'protocol.json'),
        final_checkpoint_sha256=sha(D/'bc/checkpoint_12000.pt'),source_training_script_sha256=sha(D/'train.py'))
    write(D/'summary.json',result)
    earlier=json.loads((previous/'summary.json').read_text())
    result['comparison_to_60D']=dict(source_summary_sha256=sha(previous/'summary.json'),
        action_mse_60D=earlier['results'][-1]['fit']['all73']['action_mse'],
        action_mse_61D=summary[-1]['fit']['all73']['action_mse'],
        reset_action_error_60D=earlier['startup_diagnosis']['reset_action_error_max'],
        reset_action_error_61D=startup['reset_action_error_max'],
        same_actions_and_rows=True,same_sampled_batch_indices=True,
        extra_input_changes_network_initialization=True,no_pure_causal_effect_claim=True)
    write(D/'summary.json',result)
    write(D/'metadata_label_review.json',dict(executed_helper_sha256=sha(D/'audit.py'),
        executed_training_driver_sha256=sha(D/'train.py'),
        inherited_temporary_helper_policy_kind='recorded-control-replay',
        final_policy_kind='diagnostic-common-noise-clock-bc-only',
        all_three_final_audit_reports_correct=True,source_files_changed_after_execution=False,
        explanation='evaluate() already overwrites the inherited helper default with the actual learned-policy diagnostic kind before every final report is saved. Review found no mislabeled final report; executed source is preserved unchanged.',
        final_audit_hashes={str((D/f'bc/audit_{update:05d}/report.json').relative_to(D)):sha(D/f'bc/audit_{update:05d}/report.json') for update in (3000,6000,12000)}))
    findings="""# One fixed elapsed-time BC diagnostic
+
+This separate artifact trains a fresh61-input actor on the exact same73 authentic labels and original60 observation values as the completed60D run. The added feature is decision elapsed seconds. Every appended index*0.1 value agrees with the measured source control boundary after float32 conversion (maximum float64 discrepancy1.11e-15s). Physical evaluation appends actual simulator time minus reset time and saves both original60 and augmented61 observations. The interface is explicitly diagnostic_bootstrap_elapsed_time_absolute_joint_target for rotation, not a reorientation goal or a compatible main60D policy.
+
+The fixed protocol remains seed17001, hidden256/256, mean-only Adam0.003, uniform batch256,12,000 updates, diagnostic epsilon0.01 and latent std0.3 at checkpoints. Every measured label survives clipping unchanged. No old60D weights are imported. Adding an input changes architecture and RNG consumption even under the same seed; the comparison is not a pure causal estimate. Shared environment/learning/main interfaces and the main epsilon0.02 are unchanged.
+
+All three strict physical audits failed:3k dropped after7 controls (peak force-qualified angle2.375909122rad),6k hit palm support after4 (peak3.197079780rad), and12k dropped after7 (peak1.230795701rad, final−1.495143283rad). No conditional2s continuation ran. Final actionMSE improved from0.00249010886 in60D to1.57270695e-6, maximum action error is0.00694829226, and reset hold error improved from0.841766983 to0.00635620952. Clock information supports much better fitting of the inherited startup distinction, but these actual rollouts do not establish a successful or robust policy.
+
+Exactly12,000 BC optimizer steps and the same sampled batch indices verify. Critics, target critics and temperature are unchanged within this run; critic/online counters remainzero. No head correction, extra epochs, noise trials, SAC updates, successful learned trajectory or main-stage demonstration claim is made.
+
+The inherited audit helper initially uses a generic recorded-control label, but evaluate() replaces it with diagnostic-common-noise-clock-bc-only before saving each final report. All three final records are correctly labeled; executed code remains unchanged. metadata_label_review.json binds that check to the executed helper and driver hashes.
+""".replace('\n+','\n')
    (D/'findings.md').write_text(findings)
    write(D/'manifest.json',{str(p.relative_to(D)):sha(p) for p in sorted(D.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p.name!='manifest.json'})
    print(json.dumps(dict(summary_sha256=sha(D/'summary.json'),checkpoint_sha256=result['final_checkpoint_sha256'],startup=startup),indent=2))

if __name__=='__main__':main()
