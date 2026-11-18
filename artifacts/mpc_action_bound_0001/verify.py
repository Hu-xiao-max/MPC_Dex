"""Bounded interface checks; the sole physical plan is recorded in report.json."""
from pathlib import Path
import importlib.util
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from mpc_dex.cli import read_config
from mpc_dex.contracts import sha256
from mpc_dex.env import AllegroRotationEnv
from mpc_dex.mpc import CEMConfig
from mpc_dex.primitives import PrimitiveBasis
from scripts.probe_mpc import SlewLimitedCEM,action_search_bounds

artifact=Path(__file__).resolve().parent
historical=ROOT/'artifacts/mpc_matched40_reference_0001/source/probe_mpc.py'
spec=importlib.util.spec_from_file_location('historical_probe',historical)
old=importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
_,ec=read_config(artifact/'source/config.json')
env=AllegroRotationEnv(ec)
checks={}
try:
    env.reset(seed=42)
    initial=env.data.ctrl.copy()
    actual_low,actual_high=env.low.copy(),env.high.copy()
    low,high,metadata=action_search_bounds(env,1.)
    checks['default_limits_equal_previous']=np.array_equal(low,actual_low) and np.array_equal(high,actual_high)
    with np.load(artifact/'source/input_trajectory.npz',allow_pickle=False) as archive:
        frames=np.vstack((env.data.qpos[env.q_ids],archive['qpos'][:,env.q_ids]))
    prior=PrimitiveBasis.from_trajectories(frames,provenance={'source':'recorded physical prior; interface verification only'})
    for references in (False,True):
        for slew in (None,.02):
            previous=old.SlewLimitedCEM(prior,actual_low,actual_high,
                CEMConfig(timestep_seconds=.1),42,reference_candidates=references)
            current=SlewLimitedCEM(prior,low,high,
                CEMConfig(timestep_seconds=.1),42,reference_candidates=references)
            for planner in (previous,current):
                planner.execution_start=initial.copy()
                planner.max_target_delta=slew
                planner.reset(initial)
            mean,std=current.nominal_knots.copy(),current.std_knots.copy()
            equal=True
            for _ in range(3):
                a=previous._sample(mean,std)
                b=current._sample(mean,std)
                equal &= np.array_equal(a,b) and np.array_equal(previous._decode(a),current._decode(b))
                mean=a[:10].mean(axis=0)
                std=a[:10].std(axis=0)
            checks[f'default_sampling_decoding_exact_references_{references}_slew_{slew}']=bool(equal)
    low,high,metadata=action_search_bounds(env,.98)
    effective=metadata['effective_bound']
    checks['effective_bound_is_inward_float32']=0<effective<=.98 and float(np.float32(effective))==effective
    checks['endpoint_round_trips_exact']=all(np.array_equal(env.to_joint_targets(env.to_action(x)),x) for x in (low,high))
    checks['endpoint_normalized_actions_within_effective_bound']=all(np.max(np.abs(env.to_action(x)))<=effective for x in (low,high))
    for slew in (None,.02):
        planner=SlewLimitedCEM(prior,low,high,CEMConfig(timestep_seconds=.1),42,reference_candidates=True)
        planner.execution_start=initial.copy()
        planner.max_target_delta=slew
        planner.reset(initial)
        samples=planner._sample(planner.nominal_knots,planner.std_knots)
        targets=planner._decode(samples)
        converted=env.to_joint_targets(env.to_action(targets))
        checks[f'all_candidates_and_round_trips_bounded_slew_{slew}']=bool(
            np.all(targets>=low) and np.all(targets<=high)
            and np.all(converted>=low) and np.all(converted<=high)
            and np.max(np.abs(env.to_action(targets)))<=effective)
        checks[f'exact_hold_preserved_slew_{slew}']=np.array_equal(targets[-1],np.broadcast_to(initial,(5,16)))
    invalid=[]
    for bound in (0.,-1.,1.01,float('inf'),float('-inf'),float('nan')):
        try:action_search_bounds(env,bound)
        except ValueError:invalid.append(True)
        else:invalid.append(False)
    checks['invalid_values_rejected']=all(invalid)
    try:
        env.data.ctrl[0]=env.low[0]
        try:action_search_bounds(env,.98)
        except ValueError as error:
            checks['outside_initial_target_rejected_without_clipping']='Initial executed target' in str(error) and env.data.ctrl[0]==env.low[0]
        else:checks['outside_initial_target_rejected_without_clipping']=False
    finally:env.data.ctrl[:]=initial
    checks['environment_limits_and_initial_target_unchanged']=bool(
        np.array_equal(env.low,actual_low) and np.array_equal(env.high,actual_high)
        and np.array_equal(env.data.ctrl,initial))
    report=json.loads((artifact/'report.json').read_text())
    with np.load(artifact/'transitions.npz',allow_pickle=False) as archive:
        actions=archive['actions']
        executed=env.to_joint_targets(actions)
    checks['single_actual_native_plan_executed']=len(report['steps'])==len(actions)==1 and report['steps'][0]['prediction']['backend']=='exact_native_step_threads'
    checks['selected_executed_target_bounded']=bool(np.max(np.abs(actions))<=effective and np.all(executed>=low) and np.all(executed<=high))
    checks['reference_sampling_budget_unchanged']=report['steps'][0]['planner']['rollouts']==1050 and report['sampling']==dict(exploration=175,random_refinement=173,deterministic_references=2,reference_types=['warm_start_nominal','exact_current_target_hold'],local_diagnostic_change=True)
    checks['physical_step_no_drop_or_palm_support']=not report['steps'][0]['drop'] and not report['steps'][0]['palm_support_violation'] and report['steps'][0]['peak_palm_support_force']==0
    checks['frozen_numerical_sources_unchanged']=all(sha256(ROOT/name)==digest for name,digest in report['source_hashes'].items() if name not in ('config','input_trajectory'))
    checks={key:bool(value) for key,value in checks.items()}
    assert all(checks.values()),checks
    verification=dict(checks=checks,physical_plans=1,additional_verification_plans=0,
        physical_execution_steps=1,requested_bound=.98,effective_bound=effective,
        selected_action_abs_max=float(np.max(np.abs(actions))),
        probe_source_sha256=sha256(ROOT/'scripts/probe_mpc.py'),
        historical_probe_source_sha256=sha256(historical))
    (artifact/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps(verification,indent=2))
finally:env.close()
