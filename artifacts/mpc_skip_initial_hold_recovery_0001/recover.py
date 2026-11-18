"""One cold-reset skip-hold eight-control prefix plus bounded diagnostic MPC."""
from pathlib import Path
from dataclasses import asdict,replace
from collections import Counter
import json,sys,time

sys.dont_write_bytecode=True
D=Path(__file__).resolve().parent
sys.path.insert(0,str(D/'source'))
import mujoco
import numpy as np
from mpc_dex.env import AllegroRotationEnv,EnvConfig
from mpc_dex.mpc import PrimitiveCEM,CEMConfig,NoFeasiblePlan
from mpc_dex.primitives import PrimitiveBasis
from mpc_dex.rollouts import RolloutCostConfig
from mpc_dex.native_rollouts import NativeMujocoRolloutEvaluator
from mpc_dex.contracts import environment_contract,sha256
from scripts.probe_mpc import action_search_bounds
from scripts.audit_policy import PhysicsObserver,PHYSICS_COLUMNS,write_json
from scripts.audit_recorded_controls import RecordedPolicy
from audit_helpers import run_audit


class ProjectedReferenceCEM(PrimitiveCEM):
    """Local reference-only change: preserve the true live execution start."""
    def __init__(self,*args,**kwargs):
        self.execution_start=None;self._reference_batch=None;self.reference_records=[]
        super().__init__(*args,**kwargs)

    def _sample(self,mean,std):
        samples=super()._sample(mean,std)
        live=np.asarray(self.execution_start,dtype=np.float64)
        if live.shape!=(16,) or not np.isfinite(live).all():
            raise ValueError('Invalid actual execution start')
        self._reference_target=np.clip(live,self.action_low,self.action_high)
        projected=not np.array_equal(live,self._reference_target)
        coefficients=self.primitive.project(self._reference_target)
        encoded=np.r_[coefficients,self._reference_target-self.primitive.reconstruct(coefficients)]
        self._reference_knots=np.broadcast_to(encoded,self.nominal_knots.shape).copy()
        samples[-2]=self.nominal_knots
        samples[-1]=self._reference_knots
        self._reference_batch=samples
        self.reference_records.append(dict(reference_kind=('projected_bounded_target_reference' if projected else 'exact_current_target_hold'),
            true_live_execution_start=live.tolist(),bounded_reference_target=self._reference_target.tolist(),
            projected=projected,projected_dimensions=np.flatnonzero(live!=self._reference_target).tolist(),
            projection_max_radians=float(abs(live-self._reference_target).max())))
        return samples

    def _decode(self,knots):
        commands=super()._decode(knots)
        if knots is self._reference_batch:
            commands[-1]=self._reference_target
            if np.array_equal(knots[-2],self._reference_knots):commands[-2]=self._reference_target
        if np.any(commands<self.action_low) or np.any(commands>self.action_high):
            raise RuntimeError('A proposal escaped the fixed search bounds')
        return commands


def integration_state(env):
    spec=mujoco.mjtState.mjSTATE_INTEGRATION
    state=np.empty(mujoco.mj_stateSize(env.model,spec))
    mujoco.mj_getState(env.model,env.data,state,spec)
    return state


def main():
    ec=EnvConfig(**json.loads((D/'config.json').read_text())['environment'])
    assert ec.terminate_on_success and ec.object_diameter==.04
    with np.load(D/'prefix_transitions.npz') as a:actor={k:a[k].copy() for k in a.files}
    with np.load(D/'prefix_physics.npz') as a:actor_physics=a['samples'].copy();assert np.array_equal(a['columns'],PHYSICS_COLUMNS)
    env=AllegroRotationEnv(ec);obs,info=env.reset(seed=42)
    initial=dict(qpos=env.data.qpos.copy(),qvel=env.data.qvel.copy(),ctrl=env.data.ctrl.copy(),
                 joints=env.data.qpos[env.q_ids].copy(),integration=integration_state(env))
    assert np.array_equal(initial['qpos'],actor['initial_qpos'])
    assert np.array_equal(initial['qvel'],actor['initial_qvel'])
    assert np.array_equal(initial['ctrl'],actor['initial_ctrl'])
    # Construct immutable search bounds while the unmodified reset target is in bounds.
    search_low,search_high,search_record=action_search_bounds(env,.98)
    with np.load(D/'prior_transitions.npz') as a:frames=np.vstack((initial['joints'],a['qpos'][:,env.q_ids]))
    prior=PrimitiveBasis.from_trajectories(frames,provenance=dict(source=str(D/'prior_transitions.npz'),
        source_sha256=sha256(D/'prior_transitions.npz'),purpose='Same measured 40 mm diagnostic PCA prior',successful_policy=False))
    reference_report=json.loads((D/'reference_planner_report.json').read_text())
    for key in ['mean','basis','coefficient_min','coefficient_max','explained_variance_ratio']:
        assert np.array_equal(getattr(prior,key),reference_report['diagnostic_basis'][key])
    prior.save(D/'diagnostic_prior.npz')
    observer=PhysicsObserver(env);data={k:[] for k in ['obs','actions','rewards','next_obs','terminated','truncated','qpos','qvel','targets']}
    commands=[];evaluator=None;planner=None;began=time.monotonic()
    report=dict(purpose='One prescribed recovery test from the cold-reset skip-initial-hold branch: eight recorded controls then at most twenty corrected MPC controls',
        production_eligible=False,policy_only_success=False,critic_updates=0,online_updates=0,
        source_prefix_sha256=sha256(D/'prefix_transitions.npz'),source_prefix_physics_sha256=sha256(D/'prefix_physics.npz'),
        config=asdict(ec),environment_contract=environment_contract(ec),prefix_controls=8,max_mpc_controls=20,
        seed=42,action_search=search_record,prior_components=prior.components,
        source_script_sha256=sha256(__file__),corrected_rollout_source_sha256=sha256(D/'source/mpc_dex/rollouts.py'),
        status='replaying_skip_initial_hold_prefix',commands=commands)

    def execute(action,origin,extra=None):
        nonlocal obs,info
        before=obs.copy();previous_target=env.data.ctrl.copy()
        obs,reward,terminated,truncated,info=env.step(action)
        row=dict(control=len(commands)+1,origin=origin,reward=float(reward),raw_angle=observer.raw_angle,
            retained_angle=observer.qualified_angle,force_qualified_angle=observer.force_qualified_angle,
            finger_contacts=int(info['finger_contacts']),axis_tilt_radians=float(info['axis_tilt_radians']),
            drop=bool(info['drop']),success=bool(info['success']),palm_support_violation=bool(info['palm_support_violation']),
            failure_reason=info['failure_reason'],target_delta_max=float(abs(env.data.ctrl-previous_target).max()),
            action_abs_max=float(abs(action).max()),**(extra or {}))
        for key,value in dict(obs=before,actions=action.copy(),rewards=reward,next_obs=obs.copy(),
            terminated=terminated,truncated=truncated,qpos=env.data.qpos.copy(),qvel=env.data.qvel.copy(),
            targets=env.data.ctrl.copy()).items():data[key].append(value)
        commands.append(row)
        return terminated or truncated

    try:
        for i in range(8):
            assert not execute(actor['actions'][i],'recorded_skip_hold_prefix')
            for key in ['obs','actions','rewards','next_obs','terminated','truncated','qpos','qvel','targets']:
                assert np.array_equal(data[key][-1],actor[key][i]),(i,key)
        assert np.array_equal(np.asarray(observer.rows),actor_physics[:8*env.substeps])
        live_start=env.data.ctrl.copy();branch_state=integration_state(env)
        np.savez_compressed(D/'actual_branch_state.npz',qpos=env.data.qpos.copy(),qvel=env.data.qvel.copy(),
            ctrl=live_start,integration_state=branch_state,observation=obs.copy(),
            search_low=search_low,search_high=search_high)
        report.update(prefix_transition_and_physics_bitexact=True,branch_state_time_seconds=float(env.data.time),
            branch_force_qualified_angle=observer.force_qualified_angle,true_live_execution_start=live_start.tolist(),
            live_start_out_of_search_dimensions=np.flatnonzero((live_start<search_low)|(live_start>search_high)).tolist())
        assert environment_contract(ec)==json.loads((D/'prefix_report.json').read_text())['environment_contract']
        write_json(D/'prefix_verification.json',dict(prefix_controls=8,transition_fields=['obs','actions','rewards','next_obs','terminated','truncated','qpos','qvel','targets'],max_error=0.0,physics_rows=8*env.substeps,physics_bitexact=True,time_seconds=float(env.data.time),force_qualified_angle=observer.force_qualified_angle,info={k:info[k] for k in ['retained_angle','finger_contacts','axis_tilt_radians','success','drop','palm_support_violation']}))
        print('PREFIX_VERIFIED ' + json.dumps(dict(controls=8,physics_rows=8*env.substeps,time=float(env.data.time),force_qualified_angle=observer.force_qualified_angle)),flush=True)
        planner=ProjectedReferenceCEM(prior,search_low,search_high,CEMConfig(timestep_seconds=.1),42)
        planner.execution_start=live_start.copy()
        planner.reset(live_start)  # Changes planner latent state only; reset projects its nominal internally.
        assert np.array_equal(branch_state,integration_state(env))
        cost=RolloutCostConfig(workers=8,rotation_progress='retained',perpendicular_weight=100.)
        evaluator=NativeMujocoRolloutEvaluator(env,timestep_seconds=.1,config=cost,require_native=True,
                                               cache_directory=D/'native_cache')
        report.update(cem=asdict(planner.config),cost=asdict(cost),native_build=evaluator.native_provenance,
                      status='mpc_recovery',no_slew_limit=True)
        write_json(D/'report.json',report)
        for recovery_step in range(1,21):
            before=integration_state(env);planner.execution_start=env.data.ctrl.copy()
            reference_begin=len(planner.reference_records);start=time.monotonic()
            try:target,diagnostics=planner.plan(evaluator)
            except NoFeasiblePlan as error:
                assert np.array_equal(before,integration_state(env))
                report.update(status='no_feasible_plan',error=str(error),
                    last_rejection_counts=dict(Counter(evaluator.last_diagnostics.get('rejection_reasons',[]))),
                    last_failed_evaluation=evaluator.last_diagnostics)
                break
            assert np.array_equal(before,integration_state(env))
            selected_cost=float(evaluator(planner.last_sequence[None])[0])
            assert np.isclose(selected_cost,diagnostics['best_cost'],rtol=1e-10,atol=1e-10)
            prediction=evaluator.last_diagnostics.copy()
            assert prediction['joint_target_execution']=='float32_normalized_action'
            assert np.array_equal(before,integration_state(env))
            action=env.to_action(target);executed=env.to_joint_targets(action)
            assert np.all(executed>=search_low) and np.all(executed<=search_high)
            assert float(abs(action).max())<=search_record['effective_bound']
            boundary=execute(action,'corrected_native_mpc',dict(recovery_step=recovery_step,
                planning_seconds=time.monotonic()-start,planner=diagnostics,prediction=prediction,
                reference_candidates=planner.reference_records[reference_begin:],
                true_live_execution_start=planner.execution_start.tolist(),selected_target=target.tolist(),
                executed_target=executed.tolist(),target_mapping_max_radians=float(abs(target-executed).max()),
                proposals_and_execution_within_search_bounds=True,planning_left_live_integration_state_bitexact=True))
            write_json(D/'report.json',report)
            print(json.dumps({k:v for k,v in commands[-1].items() if k not in ['planner','prediction','reference_candidates']}),flush=True)
            if boundary:report['status']='environment_boundary';break
        else:report['status']='mpc_budget_exhausted'
        report['reference_records']=planner.reference_records
    except BaseException as error:
        report.update(status='failed',error=str(error));raise
    finally:
        report.update(wall_seconds=time.monotonic()-began,executed_controls=len(commands),
            executed_mpc_controls=max(0,len(commands)-8),final_info=info)
        # Environment info includes arrays, so preserve only the auditable scalar fields here.
        report['final_info']={k:info[k] for k in ['angle','retained_angle','success','drop','palm_support_violation','failure_reason','finger_contacts','axis_tilt_radians']}
        write_json(D/'report.json',report)
        np.savez_compressed(D/'transitions.npz',**{k:np.asarray(v) for k,v in data.items()},
            initial_qpos=initial['qpos'],initial_qvel=initial['qvel'],initial_ctrl=initial['ctrl'],
            initial_joint_positions=initial['joints'],initial_integration_state=initial['integration'],
            joint_positions=np.vstack((initial['joints'],np.asarray(data['qpos'])[:,env.q_ids])))
        np.savez_compressed(D/'physics.npz',samples=np.asarray(observer.rows),columns=np.asarray(PHYSICS_COLUMNS))
        observer.close()
        if evaluator is not None:evaluator.close()
        env.close()
    # Independent physical replay of every executed command, including the exactly verified skip-initial-hold prefix.
    actions=np.asarray(data['actions'])
    audited,_,_=run_audit(RecordedPolicy(actions),ec,D/'independent_audit',len(actions))
    audited.update(policy_kind='recorded_skip_initial_hold_prefix_then_corrected_mpc_recovery',policy_only_success=False)
    with np.load(D/'independent_audit/physics.npz') as a,np.load(D/'physics.npz') as b:
        audited['full_execution_physics_bitexact']=bool(np.array_equal(a['samples'],b['samples']))
    assert audited['full_execution_physics_bitexact']
    with np.load(D/'independent_audit/transitions.npz') as replayed:
        for key in data:np.testing.assert_array_equal(replayed[key],np.asarray(data[key]))
    audited['full_execution_transition_fields_bitexact']=list(data)
    np.savez_compressed(D/'executed_control_origins.npz',origins=np.asarray([row['origin'] for row in commands]),actual_mpc_correction_mask=np.arange(len(commands))>=8,prefix_source_rows_one_based=np.r_[np.arange(1,9),np.full(max(0,len(commands)-8),-1)])
    write_json(D/'independent_audit/report.json',audited);report['independent_audit']=audited
    if audited['finger_supported_rotation_success']:
        held=np.vstack((actions,np.repeat(actions[-1:],20,axis=0)))
        hold_ec=replace(ec,terminate_on_success=False,horizon_steps=max(ec.horizon_steps,len(held)))
        held_result,_,_=run_audit(RecordedPolicy(held),hold_ec,D/'post_success_hold_2s',len(held))
        held_result.update(policy_kind='skip_initial_hold_prefix_plus_mpc_plus_constant_target_aftermath',
            requested_hold_seconds=2.,completed_hold_seconds=max(0.,held_result['simulated_seconds']-.1*len(actions)),
            completed_two_seconds_without_failure=bool(held_result['steps']==len(held) and not held_result['independent_drop'] and
                not held_result['independent_palm_support_violation']))
        write_json(D/'post_success_hold_2s/report.json',held_result);report['post_success_hold_2s']=held_result
    write_json(D/'report.json',report)
    report['numerical_sources_unchanged']=all(sha256(D/'source'/k)==v for k,v in json.loads((D/'source_hashes.json').read_text()).items())
    assert report['numerical_sources_unchanged']
    write_json(D/'report.json',report)
    inputs=json.loads((D/'inputs_manifest.json').read_text())
    assert all(sha256(D/k)==v for k,v in inputs.items())
    manifest={str(p.relative_to(D)):sha256(p) for p in sorted(D.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p.name not in ('manifest.json','recover.log')}
    write_json(D/'manifest.json',manifest)


if __name__=='__main__':main()
