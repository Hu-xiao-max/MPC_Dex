"""Saved-observation singleton inference and nominal-prefix comparison; no physics."""
from pathlib import Path
import hashlib,json,sys,shutil
D=Path(__file__).resolve().parent
sys.path.insert(0,str(D/'source'))
import numpy as np
import torch
from mpc_dex.learning import SACAgent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    agent=SACAgent.load(D/'candidate.pt','cuda')
    with np.load(D/'independent_audit/transitions.npz') as z:actual={k:v.copy() for k,v in z.items()}
    nominal_source=D.parent/'bc_common_noise_clock_0001/inputs/common/nominal/transitions.npz'
    source_digest=sha(nominal_source)
    assert source_digest=='29a0e448db77a04b0916fa961673131fef6cf8d03b7f74a8ba9f37a39ebbdcbb'
    shutil.copyfile(nominal_source,D/'inputs/nominal_transitions.npz')
    with np.load(nominal_source) as z:nominal={k:v.copy() for k,v in z.items()}
    with np.load(D/'supervised_pairs.npz') as z:
        nominal_obs=z['obs'][:9].copy();nominal_actions=z['actions'][:9].copy()
    np.testing.assert_array_equal(nominal_obs[:,:60],nominal['obs'])
    np.testing.assert_array_equal(nominal_actions,nominal['actions'])
    means=[];pred=[]
    with torch.no_grad():
        for ob in actual['obs']:
            mean,_=agent.actor(torch.as_tensor(ob,device='cuda',dtype=torch.float32))
            means.append(mean.cpu().numpy());pred.append(agent.act(ob,deterministic=True))
        targets=torch.atanh(torch.as_tensor(nominal_actions,device='cuda')).cpu().numpy()
    means=np.asarray(means);pred=np.asarray(pred)
    np.testing.assert_array_equal(pred,actual['actions'])
    count=min(9,len(pred));delta=actual['obs'][:count].astype(float)-nominal_obs[:count]
    action_delta=actual['actions'][:count].astype(float)-nominal_actions[:count]
    latent_delta=means[:count].astype(float)-targets[:count]
    qdelta=actual['qpos'][:count]-nominal['qpos'][:count];vdelta=actual['qvel'][:count]-nominal['qvel'][:count]
    groups=[('q',0,16),('position',16,19),('quaternion',19,23),('angles',23,25),('qvel',25,41),('object_velocity',41,44),('previous_action',44,60),('clock_seconds',60,61)]
    rows=[]
    for i in range(count):
        rows.append(dict(control_one_based=i+1,elapsed_seconds=float(actual['decision_times'][i]),
            observation_l2=float(np.linalg.norm(delta[i,:60])),observation_max_abs=float(abs(delta[i,:60]).max()),
            clock_difference=float(delta[i,60]),action_l2=float(np.linalg.norm(action_delta[i])),
            action_max_abs=float(abs(action_delta[i]).max()),latent_l2=float(np.linalg.norm(latent_delta[i])),
            latent_max_abs=float(abs(latent_delta[i]).max()),post_control_qpos_max_abs=float(abs(qdelta[i]).max()),
            post_control_qvel_max_abs=float(abs(vdelta[i]).max()),
            observation_groups={name:dict(l2=float(np.linalg.norm(delta[i,start:end])),max_abs=float(abs(delta[i,start:end]).max())) for name,start,end in groups}))
    step2=dict(**rows[1],actual_raw_mean=means[1].tolist(),actual_action=actual['actions'][1].tolist(),
        nominal_latent_target=targets[1].tolist(),nominal_action=nominal_actions[1].tolist())
    result=dict(status='complete',candidate_sha256=sha(D/'candidate.pt'),nominal_source_path=str(nominal_source),
        nominal_sha256=source_digest,actual_transition_sha256=sha(D/'independent_audit/transitions.npz'),
        saved_actions_recomputed_bitexact=True,new_physics_steps=0,new_optimizer_steps=0,
        actual_inference='CUDA float32 singleton, one thread, TF32off',step2=step2,first9_rows=rows,
        interpretation='Comparison uses actual decisions at matching elapsed times. Zero clock differences do not imply equal physical states. Small initial command errors and later state-dependent command divergence remain; no robustness inference.')
    np.savez_compressed(D/'execution_comparison.npz',actual_raw_means=means,actual_actions=actual['actions'],
        actual_observations=actual['obs'],nominal_observations=nominal_obs,nominal_actions=nominal_actions,
        nominal_latent_targets=targets,observation_delta=delta,action_delta=action_delta,
        latent_delta=latent_delta,qpos_delta=qdelta,qvel_delta=vdelta)
    write(D/'execution_comparison.json',result)
    report=json.loads((D/'report.json').read_text());r=report['strict']
    summary=dict(status='complete',bc_updates=12000,new_bc_updates=0,critic_updates=0,online_updates=0,
        frozen_head_residual_solves=1,extra_solves=0,strict_physical_audits=1,conditional_continuations=0,
        force_qualified_success=False,steps=r['steps'],failure_reason=r['environment_failure_reason'],
        peak_force_qualified_angle=r['max_force_qualified_angle'],final_force_qualified_angle=r['final_force_qualified_angle'],
        palm_support=r['independent_palm_support_violation'],drop=r['independent_drop'],
        feature_rank=report['feature_rank'],condition_number=report['condition_number'],
        correction_frobenius_norm=report['correction_frobenius_norm'],head_weight_norm=report['candidate_weight_norm'],
        singleton_fit_action_max_abs=report['candidate_fit']['all73']['action_max_abs_error'],
        candidate_checkpoint_sha256=report['candidate_checkpoint_sha256'],
        step2={k:v for k,v in step2.items() if not isinstance(v,list)},
        policy_interface=report['policy_interface'],production_eligible=False,sac_success_claim=False,
        source_bc_run_unchanged=True)
    assert not r['finger_supported_rotation_success'] and r['independent_drop']
    write(D/'summary.json',summary)
    findings=f'''# One frozen-head clock-policy correction

One float64 minimum-norm residual solve used the73 authentic float32 latent targets with the final61D BC trunk fixed. The73x257 feature design has rank73 and condition3.13046e6. Correction Frobenius norm is4.41821, maximum coefficient0.400166, and head weight norm grows from3.19472 to5.45770. Only mean-head weight/bias changed; all other serialized learner and optimizer values are bit-exact. Actual float32 CUDA singleton fit has maximum action error4.47034836e-7. The original BC artifact remains unchanged.

After numerical review, exactly one strict physical audit used the actual61D clock policy. It reached peak force-qualified rotation6.060199270rad (347.22degrees) and dropped at control19; final qualified angle3.826087702rad. It never met the full-turn success criterion and had no palm-support violation. No conditional2s continuation was run. There were no additional solves, BC epochs, SAC updates or success claims.

At control2 the clock matches the nominal0.1s decision exactly after float32 conversion. Actual observation deviation from the nominal physical state has L2{step2['observation_l2']:.12g} and maximum{step2['observation_max_abs']:.12g}; actual action deviation is{step2['action_max_abs']:.12g} and latent-mean deviation is{step2['latent_max_abs']:.12g}. Full step2 mean/action vectors and semantic observation-group differences are in execution_comparison.json; first-nine-control arrays are in execution_comparison.npz. These are actual visited observations, without interpolated states or a robustness claim.

Counters remain BC12000/critic0/SAC0. This is an artifact-local61D rotation diagnostic with a distinct elapsed-time interface, not a successful SAC checkpoint, a main60D policy, or eligible primitive evidence.
'''
    (D/'findings.md').write_text(findings)
    write(D/'manifest.json',{str(p.relative_to(D)):sha(p) for p in sorted(D.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p.name!='manifest.json'})
    print(json.dumps(dict(summary_sha256=sha(D/'summary.json'),step2=step2),indent=2))
if __name__=='__main__':main()
