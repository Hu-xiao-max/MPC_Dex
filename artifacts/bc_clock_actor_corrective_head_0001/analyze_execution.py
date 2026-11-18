"""Verify saved full states and singleton inference; compare actual nominal clocks."""
from pathlib import Path
import hashlib,json,sys,shutil
D=Path(__file__).resolve().parent
sys.path.insert(0,str(D/'source'))
import numpy as np
import torch
import mujoco
from mpc_dex.learning import SACAgent
from mpc_dex.env import EnvConfig,make_model
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def read(path):
    with np.load(path,allow_pickle=False) as z:return {k:v.copy() for k,v in z.items()}
def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    agent=SACAgent.load(D/'candidate.pt','cuda');actual=read(D/'independent_audit/transitions.npz')
    nominal_source=D.parent/'bc_clock_actor_corrective_0001/inputs/common/nominal/transitions.npz'
    recovered_source=D.parent/'mpc_clock_actor_recovery_0001/recovery/transitions.npz'
    for label,source in [('nominal',nominal_source),('successful_recovery',recovered_source)]:
        shutil.copyfile(source,D/f'inputs/{label}_transitions.npz')
    nominal=read(nominal_source);recovered=read(recovered_source)
    with np.load(D/'supervised_pairs.npz') as z:nominal_obs=z['obs'][:9].copy()
    np.testing.assert_array_equal(nominal_obs[:,:60],nominal['obs'])
    ec=EnvConfig(**json.loads((D/'strict_config.json').read_text())['environment']);model=make_model(ec);data=mujoco.MjData(model)
    spec=mujoco.mjtState.mjSTATE_INTEGRATION;size=mujoco.mj_stateSize(model,spec)
    assert actual['integration_state'].shape==(19,size) and actual['initial_integration_state'].shape==(size,)
    for row in range(20):
        state=actual['initial_integration_state'] if row==0 else actual['integration_state'][row-1]
        mujoco.mj_setState(model,data,state,spec)
        for name,field in [('qpos','qpos'),('qvel','qvel'),('ctrl','targets')]:
            expected=actual['initial_'+name] if row==0 else actual[field][row-1]
            np.testing.assert_array_equal(getattr(data,name),expected)
        assert data.time-float(actual['initial_time'])==actual['control_times'][row]
    np.testing.assert_array_equal(actual['obs'][:,:60],actual['environment_obs'])
    np.testing.assert_array_equal(actual['next_obs'][:,:60],actual['environment_next_obs'])
    np.testing.assert_array_equal(actual['obs'][:,60],actual['decision_times'].astype(np.float32))
    np.testing.assert_array_equal(actual['next_obs'][:,60],actual['control_times'][1:].astype(np.float32))
    @torch.no_grad()
    def infer(observations):
        means=[];actions=[]
        for observation in observations:
            mean,_=agent.actor(torch.as_tensor(observation,device='cuda',dtype=torch.float32))
            means.append(mean.cpu().numpy());actions.append(agent.act(observation,deterministic=True))
        return np.asarray(means),np.asarray(actions)
    means,pred=infer(actual['obs']);np.testing.assert_array_equal(pred,actual['actions'])
    references={};allarrays=dict(actual_means=means,actual_actions=actual['actions'],actual_observations=actual['obs'])
    groups=[('q',0,16),('position',16,19),('quaternion',19,23),('angles',23,25),('qvel',25,41),('object_velocity',41,44),('previous_action',44,60),('clock',60,61)]
    for label,source,obs in [('nominal9',nominal,nominal_obs),('recovered14',recovered,recovered['obs'])]:
        n=min(len(obs),len(pred));refmeans,refpred=infer(obs[:n])
        od=actual['obs'][:n].astype(float)-obs[:n];ad=pred[:n].astype(float)-source['actions'][:n]
        feedback=pred[:n].astype(float)-refpred;fit=refpred.astype(float)-source['actions'][:n]
        rows=[]
        for i in range(n):
            rows.append(dict(control_one_based=i+1,elapsed_seconds=float(actual['decision_times'][i]),
                obs_l2=float(np.linalg.norm(od[i,:60])),obs_max_abs=float(abs(od[i,:60]).max()),
                action_max_abs=float(abs(ad[i]).max()),feedback_component_action_max_abs=float(abs(feedback[i]).max()),
                reference_fit_action_max_abs=float(abs(fit[i]).max()),clock_difference=float(od[i,60]),
                post_qpos_max_abs=float(abs(actual['qpos'][i]-source['qpos'][i]).max()),
                post_qvel_max_abs=float(abs(actual['qvel'][i]-source['qvel'][i]).max()),
                groups={name:dict(l2=float(np.linalg.norm(od[i,start:end])),max_abs=float(abs(od[i,start:end]).max())) for name,start,end in groups}))
        first={}
        for threshold in [1e-6,1e-4,1e-3,1e-2]:
            candidates=[r for r in rows if r['action_max_abs']>threshold]
            first[str(threshold)]=candidates[0] if candidates else None
        references[label]=dict(rows=rows,first_action_difference_above_declared_thresholds=first,
            threshold_note='Descriptive normalized-action error thresholds, not a physical safety or success definition')
        for name,value in [('observations',obs[:n]),('actions',source['actions'][:n]),('means',refmeans),
                           ('policy_predictions',refpred),('observation_delta',od),('action_delta',ad),
                           ('feedback_action_delta',feedback),('reference_fit_action_delta',fit)]:allarrays[label+'_'+name]=value
    np.savez_compressed(D/'execution_comparison.npz',**allarrays)
    result=dict(status='complete',candidate_sha256=sha(D/'candidate.pt'),actual_actions_reconstructed_bitexact=True,
        full_integration_states_verified=20,source_physics_reexecuted=False,new_optimizer_steps=0,
        nominal_source_sha256=sha(nominal_source),recovered_source_sha256=sha(recovered_source),references=references)
    write(D/'execution_comparison.json',result)
    report=json.loads((D/'report.json').read_text());r=report['strict']
    summary=dict(status='complete',dataset_rows=86,observation_dimension=61,bc_updates=12000,critic_updates=0,online_updates=0,
        head_residual_solves=1,strict_actual_policy_audits=1,stochastic_trials=0,post_success_continuations=0,
        strict_force_success=r['finger_supported_rotation_success'],steps=r['steps'],failure_reason=r['environment_failure_reason'],
        peak_force_angle=r['max_force_qualified_angle'],peak_time_seconds=r['max_qualified_angle_time_seconds'],
        final_force_angle=r['final_force_qualified_angle'],independent_drop=r['independent_drop'],palm=r['independent_palm_support_violation'],
        full_integration_states_verified=20,exact_candidate_inference_reconstruction=True,
        first_nominal_action_difference_gt_1e_3=references['nominal9']['first_action_difference_above_declared_thresholds']['0.001'],
        head_correction_norm=report['correction_frobenius_norm'],head_weight_norm=report['candidate_weight_norm'],
        feature_condition=report['condition_number'],training_action_max_error=report['candidate_fit']['all86']['action_max_abs_error'],
        candidate_sha256=report['candidate_checkpoint_sha256'],production_eligible=False,main_demonstrations=0,sac_success_claim=False)
    write(D/'summary.json',summary)
    divergence=summary['first_nominal_action_difference_gt_1e_3']
    (D/'findings.md').write_text(f'''# One86-pair frozen-head policy audit

One float64 minimum-norm residual solve preserved the final86-pair61D BC trunk and all non-head learner/optimizer values. The86x257 design has full row rank and condition8.35169e7; correction norm7.07181, maximum coefficient0.756758, and final headweight norm7.70553. The ideal maximum residual is4.33e-15. Actual CUDA singleton float32 label fit has maximum actionerror4.76837e-7, including all original/noisy-prefix/actual-policy/MPC groups. There were no alternate solves, regularization sweeps or BC epochs.

After numerical review, exactly one strict deterministic actual-policy audit ran with unchanged physics and the artifact-local actual elapsed-time feature. It dropped at control19, without a qualified full turn or palm support. Per2ms peak force-qualified angle was5.644944168rad at1.370s; final angle was−0.738627121rad. No stochastic test or post-success continuation ran.

The reset and all19control boundaries preserve complete MuJoCo integration states; every saved qpos/qvel/ctrl/time agrees with those states. Candidate inference on every saved visited observation reproduces the executed float32 actions bit for bit. Comparison against nominal9 and the successful recovered14 records is saved without rerunning physics. Using the explicitly descriptive threshold of normalized action error>0.001, the first nominal divergence is control{divergence['control_one_based']} at{divergence['elapsed_seconds']:.3f}s: maxactionerror{divergence['action_max_abs']:.9g}, observationL2{divergence['obs_l2']:.9g}. Other thresholds and semantic observation groups are retained; elapsed-time equality is checked separately.

All prior artifacts are unchanged. Counters remain BC12000/critic0/SAC0. Neither the successful teacher recovery nor this failed learned execution supplies a SAC/main60D primitive or robustness claim.
''')
    write(D/'manifest.json',{str(p.relative_to(D)):sha(p) for p in sorted(D.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p.name!='manifest.json'})
    print(json.dumps(dict(summary_sha256=sha(D/'summary.json'),first_nominal_divergence=divergence,
        step2=references['nominal9']['rows'][1]),indent=2))
if __name__=='__main__':main()
