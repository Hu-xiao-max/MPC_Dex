"""One condition-reported float64 residual solve, only after the prescribed fit gate."""
from pathlib import Path
import json,shutil,sys
F=Path(__file__).resolve().parent
sys.path.insert(0,str(F/'source'))
import numpy as np
import torch
from mpc_dex.learning import SACAgent
from mpc_dex.env import EnvConfig
from mpc_dex.contracts import sha256,write_checkpoint_metadata
from scripts.audit_policy import write_json
from train import singleton_values,grouped_metrics,invariant_hashes,module_hash,evaluate


def differences(a,b,path='checkpoint'):
    if type(a) is not type(b):return [path+'[type]']
    if isinstance(a,torch.Tensor):
        return [] if a.dtype==b.dtype and a.shape==b.shape and a.numpy().tobytes()==b.numpy().tobytes() else [path]
    if isinstance(a,dict):
        assert set(a)==set(b)
        return [v for k in a for v in differences(a[k],b[k],path+'/'+str(k))]
    if isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        return [v for i,(x,y) in enumerate(zip(a,b)) for v in differences(x,y,path+'/'+str(i))]
    return [] if a==b else [path]


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    bc=json.loads((F/'bc/report.json').read_text())
    assert bc['status']=='complete'
    final_fit=bc['results'][-1]['fit']['all12']['action_max_abs_error']
    assert final_fit>1e-6,'Head solve not authorized when final fit is already within1e-6'
    inputs=json.loads((F/'inputs_manifest.json').read_text())
    assert all(sha256(F/p)==v for p,v in inputs.items())
    D=F/'head_polish';D.mkdir()
    source=F/'bc/checkpoint_12000.pt';baseline_hash=sha256(source)
    shutil.copyfile(source,D/'baseline.pt')
    shutil.copyfile(str(source)+'.metadata.json',D/'baseline.pt.metadata.json')
    agent=SACAgent.load(D/'baseline.pt','cuda')
    assert agent.bc_updates==12000 and agent.pretrain_updates==agent.online_updates==0
    ec=EnvConfig(**json.loads((F/'strict_config.json').read_text())['environment'])
    with np.load(F/'supervised_pairs.npz') as a:obs,actions=a['obs'].copy(),a['actions'].copy()
    before=dict(**invariant_hashes(agent),trunk=module_hash(agent.actor.trunk),log_std_head=module_hash(agent.actor.log_std_head))
    with torch.no_grad():
        targets=torch.atanh(torch.as_tensor(actions,device=agent.device).clamp(-.98,.98)).cpu().numpy()
        w0=agent.actor.mean_head.weight.cpu().numpy().copy();b0=agent.actor.mean_head.bias.cpu().numpy().copy()
    features,means0,pred0=singleton_values(agent,obs)
    design=np.column_stack([features.astype(np.float64),np.ones(len(obs))])
    residual=targets.astype(np.float64)-means0.astype(np.float64)
    singular=np.linalg.svd(design,compute_uv=False)
    tolerance=np.finfo(np.float64).eps*max(design.shape)*singular[0]
    rank=int((singular>tolerance).sum());condition=float(singular[0]/singular[-1])
    assert rank==12 and np.isfinite(condition),(rank,condition)
    report=dict(status='rank_verified',purpose='One bounded diagnostic fixed-trunk head correction after fresh common-continuation BC',
        production_eligible=False,fit_gate=dict(final_max_action_error=final_fit,threshold=1e-6,permitted=True),
        design_shape=list(design.shape),rank=rank,condition_number=condition,singular_values=singular.tolist(),
        baseline_sha256=baseline_hash,source_script_sha256=sha256(__file__),
        input_data_sha256=sha256(F/'supervised_pairs.npz'),rank_checked_before_solve=True,
        baseline_fit=grouped_metrics(means0,pred0,targets,actions),frozen_before=before,
        optimizer_updates=0,critic_updates=0,online_updates=0)
    write_json(D/'report.json',report)
    correction,_,solved_rank,_=np.linalg.lstsq(design,residual,rcond=None)
    assert solved_rank==rank and np.isfinite(correction).all()
    with torch.no_grad():
        agent.actor.mean_head.weight.add_(torch.as_tensor(correction[:-1].T.copy(),device=agent.device,dtype=torch.float32))
        agent.actor.mean_head.bias.add_(torch.as_tensor(correction[-1].copy(),device=agent.device,dtype=torch.float32))
        w1=agent.actor.mean_head.weight.cpu().numpy().copy();b1=agent.actor.mean_head.bias.cpu().numpy().copy()
    features1,means1,pred1=singleton_values(agent,obs)
    after=dict(**invariant_hashes(agent),trunk=module_hash(agent.actor.trunk),log_std_head=module_hash(agent.actor.log_std_head))
    assert before==after and np.array_equal(features,features1)
    checkpoint=D/'polished.pt';agent.save(checkpoint)
    metadata=write_checkpoint_metadata(checkpoint,ec,agent.config,stage='diagnostic-common-continuation-head-refinement',
        offline_replay_sha256=report['input_data_sha256'])
    metadata.update(production_eligible=False,actor_kind='Fresh BC plus one numerical mean-head correction; no SAC',
        baseline_sha256=baseline_hash,refinement_solves=1,optimizer_updates=0)
    write_json(str(checkpoint)+'.metadata.json',metadata)
    changed=differences(torch.load(D/'baseline.pt',map_location='cpu',weights_only=True),
                        torch.load(checkpoint,map_location='cpu',weights_only=True))
    assert set(changed)=={'checkpoint/actor/mean_head.weight','checkpoint/actor/mean_head.bias'}
    projected=design@correction-residual
    np.savez_compressed(D/'solve_inputs_and_result.npz',observations=obs,actions=actions,
        features_float32=features,latent_targets_float32=targets,design_float64=design,residual_float64=residual,
        correction_float64=correction,ideal_residual_float64=projected,singular_values=singular,
        baseline_means=means0,polished_means=means1,baseline_predictions=pred0,polished_predictions=pred1,
        baseline_weight=w0,baseline_bias=b0,polished_weight=w1,polished_bias=b1)
    report.update(polished_fit=grouped_metrics(means1,pred1,targets,actions),frozen_after=after,
        frozen_components_bitwise_unchanged=True,changed_checkpoint_paths=changed,
        all_other_checkpoint_payload_values_bitwise_unchanged=True,refinement_solves=1,
        correction_frobenius_norm=float(np.linalg.norm(correction)),correction_max_abs=float(abs(correction).max()),
        ideal_residual_mse=float(np.mean(projected**2)),baseline_weight_norm=float(np.linalg.norm(w0.astype(np.float64))),
        polished_weight_norm=float(np.linalg.norm(w1.astype(np.float64))),polished_sha256=sha256(checkpoint),
        actual_weight_max_change=float(abs(w1-w0).max()),actual_bias_max_change=float(abs(b1-b0).max()),
        method='One float64 minimum-norm residual solve; cast correction to original float32 head once',
        targets='Existing float32 Torch atanh(clamp(actions,-.98,.98)); exact singleton Torch fit reported',status='auditing')
    write_json(D/'report.json',report)
    outcome=evaluate(agent,ec,D/'independent_audit','diagnostic-common-continuation-bc-plus-head-correction')
    assert sha256(source)==baseline_hash and all(sha256(F/p)==v for p,v in inputs.items())
    report.update(**outcome,status='complete',baseline_source_unchanged=True,inputs_unchanged=True)
    write_json(D/'report.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
