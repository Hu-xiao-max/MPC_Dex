"""Bounded contact-planning diagnostic using a measured motion prior.

This is an exploratory bootstrap aid. It never exports a production primitive
or labels the source trajectory as a successful simulation policy.
"""
import argparse
from dataclasses import asdict
import json
from collections import Counter
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
from mpc_dex.cli import read_config
from mpc_dex.contracts import environment_contract,sha256
from mpc_dex.env import AllegroRotationEnv
from mpc_dex.mpc import CEMConfig,NoFeasiblePlan,PrimitiveCEM
from mpc_dex.primitives import PrimitiveBasis
from mpc_dex.rollouts import MujocoRolloutEvaluator,RolloutCostConfig
from scripts.audit_policy import PHYSICS_COLUMNS,PhysicsObserver


class SlewLimitedCEM(PrimitiveCEM):
    """Explicit diagnostic references and slew limits; base CEM is unchanged."""
    def __init__(self,*args,reference_candidates=False,**kwargs):
        self.reference_candidates=reference_candidates
        self._reference_batch=None
        super().__init__(*args,**kwargs)
        if self.reference_candidates and self.config.samples//2<2:
            raise ValueError('Reference candidates require at least two refinement slots')

    def _sample(self,mean,std):
        samples=super()._sample(mean,std)
        self._reference_batch=None
        if self.reference_candidates:
            target=np.asarray(self.execution_start,dtype=np.float64)
            if (target.shape!=(16,) or not np.isfinite(target).all()
                or np.any(target<self.action_low) or np.any(target>self.action_high)):
                raise ValueError('Reference hold must be a finite executed target within actuator bounds')
            coefficients=self.primitive.project(target)
            hold=np.r_[coefficients,target-self.primitive.reconstruct(coefficients)]
            self._reference_hold_knots=np.broadcast_to(hold,self.nominal_knots.shape).copy()
            # Keep all exploration samples and replace two refinement samples;
            # the encoded references participate normally in elite updates.
            samples[-2]=self.nominal_knots
            samples[-1]=self._reference_hold_knots
            self._reference_batch=samples
        return samples

    def _decode(self,knots):
        commands=super()._decode(knots)
        if knots is self._reference_batch:
            # PCA/residual cancellation can leave a few ulps of target error.
            # The deterministic hold must execute the original target exactly.
            commands[-1]=self.execution_start
            if np.array_equal(knots[-2],self._reference_hold_knots):
                commands[-2]=self.execution_start
        if self.max_target_delta is None:return commands
        previous=np.broadcast_to(self.execution_start,(len(commands),16)).copy()
        for step in range(commands.shape[1]):
            commands[:,step]=np.clip(commands[:,step],previous-self.max_target_delta,
                                     previous+self.max_target_delta)
            previous=commands[:,step]
        return commands


def action_search_bounds(env,requested_bound):
    """Narrow this diagnostic's search without changing environment actions."""
    if not np.isfinite(requested_bound) or not 0<requested_bound<=1:
        raise ValueError('Action bound must be finite and satisfy 0 < bound <= 1')
    if requested_bound==1:
        # Preserve the original default limits and all sampling numerics.
        effective_bound=1.
        low,high=env.low.copy(),env.high.copy()
        source_choice='original environment actuator limits (diagnostic default)'
    else:
        # to_joint_targets adds 1 while actions still have float32 dtype.
        # Round inward on that offset's representable grid so both endpoints
        # remain within the requested bound after action/target conversion.
        quantum=float(np.finfo(np.float32).eps)
        effective_bound=float(np.float32(np.floor(requested_bound/quantum)*quantum))
        if effective_bound<=0:
            raise ValueError('Action bound is narrower than float32 action normalization resolution')
        low=env.to_joint_targets(np.full(16,-effective_bound,dtype=np.float32))
        high=env.to_joint_targets(np.full(16,effective_bound,dtype=np.float32))
        source_choice='independent diagnostic action restriction; not a paper parameter'
    initial=np.asarray(env.data.ctrl)
    if (initial.shape!=(16,) or not np.isfinite(initial).all()
        or np.any(initial<low) or np.any(initial>high)):
        raise ValueError('Initial executed target is outside the action search bounds; refusing to clip it')
    return low,high,dict(requested_bound=float(requested_bound),effective_bound=effective_bound,
                         low=low.tolist(),high=high.tolist(),source_choice=source_choice,
                         local_diagnostic_change=requested_bound<1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(ROOT/'configs/local_sim.json'))
    parser.add_argument('--trajectory',default=str(ROOT/'artifacts/rotation_probe/orbit_0.8_s8_p1_f0.npz'))
    parser.add_argument('--output',required=True)
    parser.add_argument('--steps',type=int,default=30)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--backend',choices=['python','native'],default='python')
    parser.add_argument('--progress',choices=['raw','retained'],default='retained')
    parser.add_argument('--perpendicular-weight',type=float,default=100.)
    parser.add_argument('--max-target-delta',type=float)
    parser.add_argument('--action-bound',type=float,default=1.,
        help='Diagnostic only: restrict absolute normalized target search to +/- this bound')
    parser.add_argument('--reference-candidates',action='store_true',
        help='Diagnostic only: replace two refinement samples with warm-start nominal and exact hold')
    args=parser.parse_args()
    if min(args.steps,args.workers)<1 or args.perpendicular_weight<0:
        parser.error('Invalid diagnostic limits')
    if args.max_target_delta is not None and args.max_target_delta<=0:
        parser.error('Target slew bound must be positive')
    if not np.isfinite(args.action_bound) or not 0<args.action_bound<=1:
        parser.error('Action bound must be finite and satisfy 0 < bound <= 1')
    output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    cfg,ec=read_config(args.config)
    env=AllegroRotationEnv(ec);obs,_=env.reset(seed=args.seed)
    try:search_low,search_high,action_search=action_search_bounds(env,args.action_bound)
    except ValueError as error:
        env.close();parser.error(str(error))
    with np.load(args.trajectory,allow_pickle=False) as archive:
        frames=np.vstack((env.data.qpos[env.q_ids],archive['qpos'][:,env.q_ids]))
    prior=PrimitiveBasis.from_trajectories(frames,provenance={
        'source':str(Path(args.trajectory).resolve()),'source_sha256':sha256(args.trajectory),
        'purpose':'measured coordination prior for bootstrap diagnostic only','successful_policy':False})
    planner=SlewLimitedCEM(prior,search_low,search_high,CEMConfig(timestep_seconds=ec.control_seconds),args.seed,
                          reference_candidates=args.reference_candidates)
    planner.max_target_delta=args.max_target_delta
    planner.execution_start=env.data.ctrl.copy()
    planner.reset(env.data.ctrl)
    cost=RolloutCostConfig(workers=args.workers,rotation_progress=args.progress,
                           perpendicular_weight=args.perpendicular_weight)
    if args.backend=='native':
        from mpc_dex.native_rollouts import NativeMujocoRolloutEvaluator
        evaluator=NativeMujocoRolloutEvaluator(env,timestep_seconds=ec.control_seconds,
                                             config=cost,require_native=True)
    else:evaluator=MujocoRolloutEvaluator(env,timestep_seconds=ec.control_seconds,config=cost)
    observer=PhysicsObserver(env)
    output.mkdir(parents=True)
    source=output/'source';source.mkdir()
    source_hashes={}
    for relative in ['scripts/probe_mpc.py','scripts/audit_policy.py','mpc_dex/env.py',
                     'mpc_dex/mpc.py','mpc_dex/primitives.py','mpc_dex/rollouts.py',
                     'mpc_dex/native_rollouts.py','mpc_dex/native/exact_rollout.cc','uv.lock']:
        path=ROOT/relative;shutil.copyfile(path,source/path.name)
        source_hashes[relative]=sha256(path)
    shutil.copyfile(args.config,source/'config.json')
    source_hashes['config']=sha256(args.config)
    shutil.copyfile(args.trajectory,source/'input_trajectory.npz')
    source_hashes['input_trajectory']=sha256(args.trajectory)
    steps=[];data={k:[] for k in ['obs','actions','rewards','next_obs','qpos','qvel']}
    report=dict(purpose='exploratory measured-prior MPC diagnostic',production_eligible=False,
                environment_contract=environment_contract(ec),primitive=prior.metadata,
                diagnostic_basis={key:getattr(prior,key).tolist() for key in
                    ['mean','basis','coefficient_min','coefficient_max','explained_variance_ratio']},
                components=prior.components,cem=asdict(planner.config),cost=asdict(cost),
                seed=args.seed,max_target_delta=args.max_target_delta,
                action_search=action_search,
                reference_candidates=args.reference_candidates,
                sampling=dict(exploration=planner.config.samples//2,
                    random_refinement=planner.config.samples//2-(2 if args.reference_candidates else 0),
                    deterministic_references=2 if args.reference_candidates else 0,
                    reference_types=['warm_start_nominal','exact_current_target_hold'] if args.reference_candidates else [],
                    local_diagnostic_change=args.reference_candidates),
                requested_backend=args.backend,native_build=getattr(evaluator,'native_provenance',None),
                source_hashes=source_hashes,
                source_sha256=sha256(__file__),evaluator_sha256=sha256(ROOT/'mpc_dex/rollouts.py'),
                status='running',steps=steps)
    started=time.monotonic()
    try:
        for step in range(args.steps):
            began=time.monotonic();planner.execution_start=env.data.ctrl.copy()
            try:target,diagnostics=planner.plan(evaluator)
            except NoFeasiblePlan as error:
                diagnostic=evaluator.last_diagnostics
                report.update(status='no_feasible_plan',error=str(error),
                    last_failed_evaluation=diagnostic,
                    last_rejection_counts=dict(Counter(diagnostic.get('rejection_reasons',[]))))
                break
            # Verify the selected sequence is scored exactly as during search.
            selected_cost=float(evaluator(planner.last_sequence[None])[0])
            if not np.isclose(selected_cost,diagnostics['best_cost'],rtol=1e-10,atol=1e-10):
                raise RuntimeError('Selected trajectory prediction changed without an environment step')
            prediction=evaluator.last_diagnostics.copy()
            action=env.to_action(target)
            executed_target=env.to_joint_targets(action)
            executed_in_search=bool(np.all(executed_target>=search_low)
                                    and np.all(executed_target<=search_high))
            if args.action_bound<1 and (not executed_in_search
                    or np.max(np.abs(action))>action_search['effective_bound']):
                raise RuntimeError('Selected command leaves the diagnostic action search bounds after conversion')
            next_obs,reward,done,truncated,info=env.step(action)
            row=dict(step=step+1,planning_seconds=time.monotonic()-began,
                angle=info['angle'],retained_angle=info['retained_angle'],
                force_qualified_angle=observer.force_qualified_angle,
                contacts=info['finger_contacts'],tilt=info['axis_tilt_radians'],
                drop=info['drop'],success=info['success'],reward=reward,
                palm_support_violation=info['palm_support_violation'],
                peak_palm_support_force=info['peak_palm_support_force'],
                failure_reason=info['failure_reason'],
                target_delta_max=float(np.max(np.abs(target-planner.execution_start))),
                executed_target_delta_max=float(np.max(np.abs(executed_target-planner.execution_start))),
                selected_target_mapping_delta=float(np.max(np.abs(executed_target-target))),
                action_abs_max=float(np.max(np.abs(action))),
                executed_target_in_search_bounds=executed_in_search,
                planner=diagnostics,prediction=prediction)
            steps.append(row)
            for key,value in dict(obs=obs,actions=action,rewards=reward,next_obs=next_obs,
                                  qpos=env.data.qpos.copy(),qvel=env.data.qvel.copy()).items():
                data[key].append(value)
            obs=next_obs
            print(json.dumps(row),flush=True)
            (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            if done or truncated:
                report['status']='environment_boundary';break
        else:report['status']='diagnostic_budget_exhausted'
    except BaseException:
        report['status']='failed';raise
    finally:
        report['wall_seconds']=time.monotonic()-started
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        np.savez_compressed(output/'transitions.npz',**{k:np.asarray(v) for k,v in data.items()})
        np.savez_compressed(output/'physics.npz',samples=np.asarray(observer.rows),columns=np.asarray(PHYSICS_COLUMNS))
        observer.close();evaluator.close();env.close()


if __name__=='__main__':main()
