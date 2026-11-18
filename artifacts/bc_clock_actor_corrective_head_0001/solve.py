"""One float64 frozen-feature residual solve; no physics or optimizer update."""
from pathlib import Path
import json,sys
D=Path(__file__).resolve().parent
sys.path.insert(0,str(D/'source'))
import numpy as np
import torch
from mpc_dex.learning import SACAgent
from mpc_dex.env import EnvConfig
from mpc_dex.contracts import sha256,write_checkpoint_metadata
from scripts.audit_policy import write_json
import train

def differences(a,b,path='checkpoint'):
    if type(a) is not type(b):return [path+'[type]']
    if isinstance(a,torch.Tensor):
        return [] if a.dtype==b.dtype and a.shape==b.shape and a.numpy().tobytes()==b.numpy().tobytes() else [path]
    if isinstance(a,dict):
        assert set(a)==set(b)
        return [v for key in a for v in differences(a[key],b[key],path+'/'+str(key))]
    if isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        return [v for i,(x,y) in enumerate(zip(a,b)) for v in differences(x,y,path+'/'+str(i))]
    return [] if a==b else [path]

def main():
    assert not (D/'candidate.pt').exists() and not (D/'solve_inputs_and_result.npz').exists()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    inputs=json.loads((D/'inputs_manifest.json').read_text())
    assert all(sha256(D/name)==digest for name,digest in inputs.items())
    agent=SACAgent.load(D/'baseline.pt','cuda')
    assert agent.obs_dim==61 and agent.bc_updates==12000 and agent.pretrain_updates==agent.online_updates==0
    ec=EnvConfig(**json.loads((D/'strict_config.json').read_text())['environment'])
    with np.load(D/'supervised_pairs.npz') as z:obs=z['obs'].copy();actions=z['actions'].copy();origins=z['origin'].copy()
    assert obs.shape==(86,61) and actions.shape==(86,16)
    train.GROUP_MASKS={'prefix':origins=='recorded_fixed_mean_noisy_prefix','mpc':origins=='actual_corrected_native_mpc',
        'actor_prefix':origins=='recorded_actual_clock_actor_prefix','actor_mpc':origins=='actual_clock_actor_state_mpc_correction'}
    before=dict(**train.invariant_hashes(agent),trunk=train.module_hash(agent.actor.trunk),log_std_head=train.module_hash(agent.actor.log_std_head))
    with torch.no_grad():
        tensor=torch.as_tensor(actions,device=agent.device);assert torch.equal(tensor,tensor.clamp(-.99,.99))
        targets=torch.atanh(tensor.clamp(-.99,.99)).cpu().numpy()
        w0=agent.actor.mean_head.weight.cpu().numpy().copy();b0=agent.actor.mean_head.bias.cpu().numpy().copy()
    features,means0,pred0=train.singleton_values(agent,obs)
    design=np.column_stack([features.astype(np.float64),np.ones(len(obs))])
    residual=targets.astype(np.float64)-means0.astype(np.float64)
    left,singular,right=np.linalg.svd(design,full_matrices=False)
    tolerance=np.finfo(np.float64).eps*max(design.shape)*singular[0]
    rank=int((singular>tolerance).sum());condition=float(singular[0]/singular[-1])
    report=dict(status='rank_checked',design_shape=list(design.shape),feature_rank=rank,
        condition_number=condition,singular_values=singular.tolist(),rank_tolerance=tolerance,
        baseline_checkpoint_sha256=sha256(D/'baseline.pt'),dataset_sha256=sha256(D/'supervised_pairs.npz'),
        source_script_sha256=sha256(__file__),baseline_fit=train.grouped_metrics(means0,pred0,targets,actions),
        physical_evaluations=0,bc_updates=12000,new_bc_updates=0,pretrain_updates=0,online_updates=0,
        optimizer_updates=0,production_eligible=False,policy_interface=json.loads((D/'policy_interface.json').read_text()))
    write_json(D/'report.json',report)
    assert rank==86 and np.isfinite(condition),(rank,condition)
    correction,_,solved_rank,_=np.linalg.lstsq(design,residual,rcond=None)
    assert solved_rank==rank and np.isfinite(correction).all()
    with torch.no_grad():
        agent.actor.mean_head.weight.add_(torch.as_tensor(correction[:-1].T.copy(),device=agent.device,dtype=torch.float32))
        agent.actor.mean_head.bias.add_(torch.as_tensor(correction[-1].copy(),device=agent.device,dtype=torch.float32))
        w1=agent.actor.mean_head.weight.cpu().numpy().copy();b1=agent.actor.mean_head.bias.cpu().numpy().copy()
    features1,means1,pred1=train.singleton_values(agent,obs)
    after=dict(**train.invariant_hashes(agent),trunk=train.module_hash(agent.actor.trunk),log_std_head=train.module_hash(agent.actor.log_std_head))
    assert before==after and np.array_equal(features,features1)
    checkpoint=D/'candidate.pt';agent.save(checkpoint)
    metadata=write_checkpoint_metadata(checkpoint,ec,agent.config,stage='diagnostic-clock86-frozen-mean-head-correction',
        offline_replay_sha256=report['dataset_sha256'],policy_interface=report['policy_interface'])
    metadata.update(production_eligible=False,primitive_eligible=False,main_mpc_dataset_eligible=False,
        baseline_sha256=report['baseline_checkpoint_sha256'],residual_solves=1,optimizer_updates=0,
        bc_updates=12000,pretrain_updates=0,online_updates=0,physical_evaluations=0)
    write_json(str(checkpoint)+'.metadata.json',metadata)
    changed=differences(torch.load(D/'baseline.pt',map_location='cpu',weights_only=True),
        torch.load(checkpoint,map_location='cpu',weights_only=True))
    assert set(changed)=={'checkpoint/actor/mean_head.weight','checkpoint/actor/mean_head.bias'}
    projected=design@correction-residual
    np.savez_compressed(D/'solve_inputs_and_result.npz',observations=obs,actions=actions,
        features_float32=features,latent_targets_float32=targets,design_float64=design,residual_float64=residual,
        correction_float64=correction,ideal_residual_float64=projected,singular_values=singular,
        residual_left_singular_projection=left.T@residual,ideal_remaining_left_projection=left.T@projected,
        baseline_means=means0,candidate_means=means1,baseline_predictions=pred0,candidate_predictions=pred1,
        baseline_weight=w0,baseline_bias=b0,candidate_weight=w1,candidate_bias=b1)
    report.update(status='candidate_ready_physics_review_pending',residual_solves=1,
        candidate_fit=train.grouped_metrics(means1,pred1,targets,actions),
        reset_action_error_max=float(abs(pred1[0].astype(float)-actions[0]).max()),
        reset_latent_error_max=float(abs(means1[0].astype(float)-targets[0]).max()),
        frozen_before=before,frozen_after=after,changed_checkpoint_paths=changed,
        all_other_checkpoint_values_bitexact=True,correction_frobenius_norm=float(np.linalg.norm(correction)),
        correction_max_abs=float(abs(correction).max()),ideal_residual_max_abs=float(abs(projected).max()),
        ideal_residual_mse=float(np.mean(projected**2)),baseline_weight_norm=float(np.linalg.norm(w0.astype(np.float64))),
        candidate_weight_norm=float(np.linalg.norm(w1.astype(np.float64))),candidate_weight_max_abs=float(abs(w1).max()),
        candidate_bias_norm=float(np.linalg.norm(b1.astype(np.float64))),candidate_bias_max_abs=float(abs(b1).max()),
        actual_weight_max_change=float(abs(w1-w0).max()),actual_bias_max_change=float(abs(b1-b0).max()),
        candidate_checkpoint_sha256=sha256(checkpoint),actual_singleton_dtype='float32 CUDA one thread TF32off',
        target='CUDA float32 Torch atanh(float32 actions clamped±.99); clamp isexactidentity')
    assert all(sha256(D/name)==digest for name,digest in inputs.items())
    write_json(D/'report.json',report)
    write_json(D/'manifest.json',{str(p.relative_to(D)):sha256(p) for p in sorted(D.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p.name not in ['manifest.json','solve.log']})
    print(json.dumps({key:report[key] for key in ['status','feature_rank','condition_number','correction_frobenius_norm',
        'correction_max_abs','baseline_weight_norm','candidate_weight_norm','candidate_weight_max_abs',
        'candidate_bias_max_abs','candidate_checkpoint_sha256']}),flush=True)
    print(json.dumps(report['candidate_fit']['all86']),flush=True)

if __name__=='__main__':main()
