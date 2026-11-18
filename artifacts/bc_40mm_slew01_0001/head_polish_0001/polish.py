"""One frozen-trunk least-squares residual correction; explicitly not paper BC."""
from pathlib import Path
from dataclasses import replace
import json
import shutil
import sys

D=Path(__file__).resolve().parent
F=D.parent
sys.path.insert(0,str(F));sys.path.insert(0,str(F/'source'))
import numpy as np
import torch
from mpc_dex.learning import SACAgent
from mpc_dex.env import EnvConfig
from mpc_dex.contracts import sha256,write_checkpoint_metadata
from audit import run_audit
from train_bc import invariant_hashes,module_hash,fit_metrics
from scripts.audit_policy import write_json


@torch.no_grad()
def singleton_values(agent,observations):
    features=[];means=[]
    for observation in observations:
        x=torch.as_tensor(observation,device=agent.device,dtype=torch.float32)
        feature=agent.actor.trunk(x)
        mean=agent.actor.mean_head(feature)
        features.append(feature.cpu().numpy());means.append(mean.cpu().numpy())
    return np.asarray(features),np.asarray(means)


def metrics(means,targets,actions):
    predicted=np.tanh(means)
    return dict(latent_mse=float(np.mean((means.astype(np.float64)-targets.astype(np.float64))**2)),
        latent_max_abs_error=float(abs(means-targets).max()),
        action_mse=float(np.mean((predicted.astype(np.float64)-actions.astype(np.float64))**2)),
        action_max_abs_error=float(abs(predicted-actions).max()),
        first_action_max_abs_error=float(abs(predicted[0]-actions[0]).max()),
        action_max_abs_error_by_step=abs(predicted-actions).max(1).tolist())


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    source=F/'bc_paper/checkpoint_12000.pt';baseline_source_hash=sha256(source)
    shutil.copyfile(source,D/'baseline.pt')
    shutil.copyfile(str(source)+'.metadata.json',D/'baseline.pt.metadata.json')
    agent=SACAgent.load(D/'baseline.pt','cuda')
    ec=EnvConfig(**json.loads((F/'strict_config.json').read_text())['environment'])
    with np.load(F/'strict_first_success/transitions.npz') as a:
        obs,actions=a['obs'].copy(),a['actions'].copy()
    frozen=dict(**invariant_hashes(agent),trunk=module_hash(agent.actor.trunk),
                log_std_head=module_hash(agent.actor.log_std_head))
    with torch.no_grad():
        target_tensor=torch.atanh(torch.as_tensor(actions,device=agent.device).clamp(-1+.02,1-.02))
        targets=target_tensor.cpu().numpy()
        weight_before=agent.actor.mean_head.weight.detach().cpu().numpy().copy()
        bias_before=agent.actor.mean_head.bias.detach().cpu().numpy().copy()
        batched_features=agent.actor.trunk(torch.as_tensor(obs,device=agent.device)).cpu().numpy()
    features,means_before=singleton_values(agent,obs)
    design=np.column_stack((features.astype(np.float64),np.ones(len(features))))
    # Fit the residual of the actually executed float32 singleton mean route.
    residual=targets.astype(np.float64)-means_before.astype(np.float64)
    correction,residual_sums,rank,singular_values=np.linalg.lstsq(design,residual,rcond=None)
    projected_residual=design@correction-residual
    with torch.no_grad():
        agent.actor.mean_head.weight.add_(torch.as_tensor(correction[:-1].T.copy(),device=agent.device,dtype=torch.float32))
        agent.actor.mean_head.bias.add_(torch.as_tensor(correction[-1].copy(),device=agent.device,dtype=torch.float32))
    features_after,means_after=singleton_values(agent,obs)
    frozen_after=dict(**invariant_hashes(agent),trunk=module_hash(agent.actor.trunk),
                log_std_head=module_hash(agent.actor.log_std_head))
    assert frozen==frozen_after and np.array_equal(features,features_after)
    assert agent.bc_updates==12000 and agent.pretrain_updates==agent.online_updates==0
    checkpoint=D/'polished.pt';agent.save(checkpoint)
    metadata=write_checkpoint_metadata(checkpoint,ec,agent.config,stage='diagnostic-fixed-trunk-head-refinement',
        offline_replay_sha256=sha256(F/'strict_first_success/transitions.npz'))
    metadata.update(production_eligible=False,actor_kind='BC plus one numerical mean-head correction; not paper BC or SAC',
        baseline_checkpoint_sha256=baseline_source_hash,refinement_solves=1,optimizer_updates=0,
        frozen_numerical_source=str(F/'source'),refinement_script_sha256=sha256(__file__))
    write_json(str(checkpoint)+'.metadata.json',metadata)
    with torch.no_grad():
        weight_after=agent.actor.mean_head.weight.cpu().numpy().copy()
        bias_after=agent.actor.mean_head.bias.cpu().numpy().copy()
    np.savez_compressed(D/'solve_inputs_and_result.npz',observations=obs,actions=actions,
        target_latents_float32=targets,singleton_features_float32=features,batched_features_float32=batched_features,
        design_float64=design,residual_float64=residual,correction_float64=correction,
        projected_residual_float64=projected_residual,singular_values=singular_values,
        baseline_mean_float32=means_before,polished_mean_float32=means_after,
        baseline_weight=weight_before,baseline_bias=bias_before,polished_weight=weight_after,polished_bias=bias_after)
    report=dict(purpose='One bounded numerical precision diagnostic, not paper BC, SAC, or a deployed policy',
        production_eligible=False,baseline_sha256=baseline_source_hash,baseline_copy_sha256=sha256(D/'baseline.pt'),
        polished_sha256=sha256(checkpoint),source_script_sha256=sha256(__file__),
        input_sha256=sha256(F/'strict_first_success/transitions.npz'),source_manifest_sha256=sha256(F/'inputs_manifest.json'),
        numpy_version=np.__version__,torch_version=str(torch.__version__),
        solve=dict(method='numpy.linalg.lstsq minimum-norm residual correction',dtype='float64',rcond=None,
            feature_route='Singleton float32 actor trunk, matching actual deterministic policy inference',
            targets='Existing float32 torch.atanh(clamp(actual_action,-.98,.98)); unchanged BC target definition',
            application='Cast correction to float32, add to original float32 mean-head weight and bias exactly once',
            design_shape=list(design.shape),rank=int(rank),singular_values=singular_values.tolist(),
            condition_number=float(singular_values[0]/singular_values[-1]) if singular_values[-1]>0 else None,
            correction_frobenius_norm=float(np.linalg.norm(correction)),correction_max_abs=float(abs(correction).max()),
            ideal_residual_mse=float(np.mean(projected_residual**2)),
            actual_weight_max_change=float(abs(weight_after-weight_before).max()),
            actual_bias_max_change=float(abs(bias_after-bias_before).max()),
            batch_vs_singleton_feature_max_difference=float(abs(batched_features-features).max())),
        baseline_fit=metrics(means_before,targets,actions),polished_fit=metrics(means_after,targets,actions),
        frozen_before=frozen,frozen_after=frozen_after,frozen_components_bitwise_unchanged=frozen==frozen_after,
        critic_updates=0,online_updates=0,extra_optimizer_updates=0,status='auditing')
    write_json(D/'report.json',report)
    evaluation,executed,_=run_audit(agent,ec,D/'independent_audit',100)
    evaluation['policy_kind']='diagnostic-fixed-trunk-head-refinement'
    write_json(D/'independent_audit/report.json',evaluation)
    report['evaluation']=evaluation
    if evaluation['finger_supported_rotation_success']:
        total=len(executed)+20;continuous_ec=replace(ec,terminate_on_success=False,horizon_steps=max(100,total))
        aftermath,_,_=run_audit(agent,continuous_ec,D/'continuous_policy_aftermath',total)
        with np.load(D/'independent_audit/physics.npz') as a,np.load(D/'continuous_policy_aftermath/physics.npz') as b:
            assert np.array_equal(a['samples'],b['samples'][:len(a['samples'])])
        aftermath.update(policy_kind='diagnostic-fixed-trunk-head-refinement',requested_post_success_seconds=2.,
            completed_post_success_seconds=max(0.,aftermath['simulated_seconds']-.1*len(executed)),
            completed_two_seconds_without_failure=bool(aftermath['steps']==total and not aftermath['independent_drop'] and
                not aftermath['independent_palm_support_violation']))
        report['continuous_policy_aftermath']=aftermath
        write_json(D/'continuous_policy_aftermath/report.json',aftermath)
    assert sha256(source)==baseline_source_hash
    report.update(status='complete',baseline_source_unchanged=True)
    write_json(D/'report.json',report)
    manifest={str(p.relative_to(D)):sha256(p) for p in sorted(D.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p.name not in ('manifest.json','polish.log')}
    write_json(D/'manifest.json',manifest)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
