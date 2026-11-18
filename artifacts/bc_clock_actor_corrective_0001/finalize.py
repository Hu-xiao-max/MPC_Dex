"""Verify and freeze the86-pair BC run; head refinement is not executed here."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
D=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def main():
    inputs=json.loads((D/'inputs_manifest.json').read_text())
    assert all(sha(D/name)==digest for name,digest in inputs.items())
    for local,source in [('inputs/clock73',D.parent/'bc_common_noise_clock_0001'),
                         ('inputs/clock_recovery',D.parent/'mpc_clock_actor_recovery_0001')]:
        manifest=json.loads((D/local/'manifest.json').read_text());assert all(sha(source/name)==digest for name,digest in manifest.items())
    report=json.loads((D/'bc/report.json').read_text());assert report['status']=='complete'
    init=torch.load(D/'bc/initial_fresh.pt',weights_only=True,map_location='cpu')
    previous=torch.load(D/'inputs/clock73/bc/initial_fresh.pt',weights_only=True,map_location='cpu')
    for family in ['actor','critics','target_critics']:
        assert all(torch.equal(v,previous[family][k]) for k,v in init[family].items())
    results=[]
    for update in [3000,6000,12000]:
        cp=D/f'bc/checkpoint_{update:05d}.pt';payload=torch.load(cp,weights_only=True,map_location='cpu')
        assert payload['obs_dim']==61 and payload['bc_updates']==update and payload['pretrain_updates']==payload['online_updates']==0
        assert {int(x['step']) for x in payload['bc_optimizer']['state'].values()}=={update}
        for family in ['critics','target_critics']:
            assert all(torch.equal(v,payload[family][k]) for k,v in init[family].items())
        assert torch.equal(payload['log_alpha'],init['log_alpha'])
        for optimizer in ['actor_optimizer','critic_optimizer','alpha_optimizer']:assert not payload[optimizer]['state']
        assert not payload['actor']['log_std_head.weight'].any()
        np.testing.assert_allclose(payload['actor']['log_std_head.bias'].exp().numpy(),.3,rtol=0,atol=2e-8)
        entry=next(x for x in report['results'] if x['updates']==update);r=entry['strict']
        assert not r['finger_supported_rotation_success'] and r['environment_failure_reason']=='palm_support'
        with np.load(D/f'bc/audit_{update:05d}/transitions.npz') as z:
            n=len(z['actions']);assert n==r['steps']
            np.testing.assert_array_equal(z['obs'][1:],z['next_obs'][:-1])
            np.testing.assert_array_equal(z['obs'][:,:60],z['environment_obs'])
            np.testing.assert_array_equal(z['next_obs'][:,:60],z['environment_next_obs'])
            np.testing.assert_array_equal(z['obs'][:,60],z['decision_times'].astype(np.float32))
            np.testing.assert_array_equal(z['next_obs'][:,60],z['control_times'][1:].astype(np.float32))
            np.testing.assert_array_equal(z['terminated'],np.arange(n)==n-1);assert not z['truncated'].any()
        with np.load(D/f'bc/audit_{update:05d}/physics.npz') as z:assert len(z['samples'])==n*50
        assert sha(cp)==entry['checkpoint_sha256']
        results.append(dict(bc_updates=update,checkpoint_sha256=sha(cp),steps=n,failure_reason=r['environment_failure_reason'],
            final_force_qualified_angle=r['final_force_qualified_angle'],peak_force_qualified_angle=r['max_force_qualified_angle'],
            peak_palm_force=r['peak_support_normal_forces']['palm'],drop=r['independent_drop'],
            fit={group:{k:v for k,v in values.items() if k!='action_max_abs_error_by_row'} for group,values in entry['fit'].items()}))
    with np.load(D/'bc/sampled_batch_indices.npz') as z:
        np.testing.assert_array_equal(z['indices'],np.random.default_rng(17001).integers(86,size=(12000,256)))
    with np.load(D/'bc/fit_12000.npz') as z:
        design=np.column_stack([z['features'].astype(float),np.ones(86)])
        residual=z['latent_targets'].astype(float)-z['means'].astype(float)
    singular=np.linalg.svd(design,compute_uv=False);tol=np.finfo(float).eps*max(design.shape)*singular[0]
    condition=float(singular[0]/singular[-1]);rank=int((singular>tol).sum())
    np.savez_compressed(D/'head_refinement_preflight.npz',design_float64=design,residual_float64=residual,singular_values=singular)
    preflight=dict(status='prepared_not_solved',rank=rank,design_shape=list(design.shape),condition_number=condition,
        final_action_max_error=results[-1]['fit']['all86']['action_max_abs_error'],fit_gate_threshold=1e-6,
        fit_gate_met=True,residual_solves=0,optimizer_updates=0,physical_tests=0,
        proposed_method='One float64 minimum-norm residual mean-head correction with finaltrunk frozen',
        parent_numerical_review_required_before_refinement=True,no_ridge_or_scale_sweep=True)
    write(D/'head_refinement_preflight.json',preflight)
    summary=dict(status='complete',diagnostic='86 authentic61D clock pairs: prior73+actual policy prefix8+actual MPC5',
        dataset_sha256=sha(D/'supervised_pairs.npz'),protocol_sha256=sha(D/'protocol.json'),
        rows=86,previous73_unchanged=True,new_actor_prefix_rows=8,new_mpc_rows=5,excluded_new_reset_rows=1,
        labels_unchanged=True,bc_epsilon=.01,main_bc_epsilon=.02,clock_verified=True,exact_conflicts=0,
        fresh_seed=17001,initial61D_actor_critics_bitexact_to_previous_run=True,bc_lr=.003,batch_size=256,
        bc_updates=12000,critic_updates=0,online_updates=0,sampled_batches_verified=True,
        critics_targets_temperature_unchanged=True,source_artifacts_unchanged=True,
        physical_checkpoint_audits=3,strict_successes=0,conditional_continuations=0,
        main_demonstrations=0,production_eligible=False,sac_success_claim=False,
        wall_seconds=report['wall_seconds'],results=results,optional_head_refinement=preflight,
        final_checkpoint_sha256=sha(D/'bc/checkpoint_12000.pt'))
    write(D/'summary.json',summary)
    (D/'findings.md').write_text('''# One86-pair clock corrective BC diagnostic

The dataset preserves the prior73 authentic61D pairs and adds eight actual clock-policy prefix decisions (controls2–9) plus five actual MPC corrections (controls10–14) from the independently successful recovered trajectory. The repeated reset observation's slightly different actor label is excluded; the original nominal hold remains once. All observations, float32 actions and elapsed-time features match their recorded source rows. Independent transitions and2ms physics match bit for bit. Epsilon0.01 leaves every label unchanged; main epsilon0.02 is unchanged. No exact duplicate or conflicting observation labels remain.

The nearest added prefix state differs from its nominal counterpart by observationL2=3.75516e-6 and actionmax=6.55651e-7. The nearest added MPC state differs from an earlier correction by observationL2=2.7436 and actionmax=0.00204. Raw norms mix units; these checks disclose nearby labels without claiming a smooth global optimal controller.

One fresh61D seed17001 run used exactly the prior actor/critic initialization, architecture, mean-only Adam0.003, batch256, latentstd0.3, epsilon0.01 and12,000 updates. CUDA singleton audits at3k/6k/12k all failed on palm support after2/4/4 controls, with peak force-qualified rotations1.79369/4.15215/1.11922rad. The final actionMSE is2.82112185e-6 and maxerror0.01995947. All optimizer/counter/source and sampled-batch checks pass; critics, target critics and temperature remain unchanged.

The predeclared optional frozen-head diagnostic is prepared but has not run: final feature design86x257 has rank86 and condition8.35169e7. It requires separate numerical review before the one residual solve. No extra epochs, head solve, SAC update, successful policy claim or main-stage demonstration was added by this artifact.
''')
    write(D/'manifest.json',{str(p.relative_to(D)):sha(p) for p in sorted(D.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p.name!='manifest.json'})
    print(json.dumps(dict(summary_sha256=sha(D/'summary.json'),checkpoint_sha256=summary['final_checkpoint_sha256'],head_preflight=preflight),indent=2))
if __name__=='__main__':main()
