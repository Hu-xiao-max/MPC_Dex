"""One fresh CEM plan at a recorded prefix; diagnostic artifacts only."""
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import mujoco
import numpy as np

from mpc_dex.cli import read_config
from mpc_dex.contracts import environment_contract,sha256
from mpc_dex.env import AllegroRotationEnv
from mpc_dex.mpc import CEMConfig,NoFeasiblePlan,PrimitiveCEM
from mpc_dex.native_rollouts import NativeMujocoRolloutEvaluator
from mpc_dex.primitives import PrimitiveBasis
from mpc_dex.rollouts import RolloutCostConfig

OUT=Path(__file__).resolve().parent
SOURCE=ROOT/'artifacts/mpc_partial_probe_0002'
PRIOR=ROOT/'artifacts/rotation_probe/orbit_0.8_s8_p1_f0.npz'


def brief(info):
    return {key:info[key] for key in ('angle','retained_angle','finger_contacts',
        'axis_tilt_radians','drop','success','palm_support_violation',
        'palm_support_force','peak_palm_support_force','failure_reason')}


def full_state(env):
    state=np.empty(mujoco.mj_stateSize(env.model,mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(env.model,env.data,state,mujoco.mjtState.mjSTATE_INTEGRATION)
    return state


def main():
    if (OUT/'report.json').exists():raise RuntimeError('Refusing to repeat the one-plan diagnostic')
    _,ec=read_config(ROOT/'configs/local_sim.json')
    old=json.loads((SOURCE/'report.json').read_text())
    with np.load(SOURCE/'transitions.npz',allow_pickle=False) as archive:
        actions=archive['actions'].copy();recorded_qpos=archive['qpos'].copy();recorded_qvel=archive['qvel'].copy()
    with np.load(PRIOR,allow_pickle=False) as archive:prior_qpos=archive['qpos'].copy()
    env=AllegroRotationEnv(ec);env.reset(seed=42)
    initial_joints=env.data.qpos[env.q_ids].copy()
    for action in actions[:17]:
        _,_,terminated,truncated,info=env.step(action)
        assert not terminated and not truncated and not info['palm_support_violation']
    assert np.array_equal(env.data.qpos,recorded_qpos[16])
    assert np.array_equal(env.data.qvel,recorded_qvel[16])
    prior=PrimitiveBasis.from_trajectories(np.vstack((initial_joints,prior_qpos[:,env.q_ids])),
        provenance={'source':str(PRIOR),'source_sha256':sha256(PRIOR),
                    'purpose':'diagnostic partial prior only','successful_policy':False})
    assert prior.metadata['extraction']==old['primitive']['extraction']
    config=CEMConfig(**old['cem'])
    cost_values=old['cost'].copy();cost_values['workers']=1
    cost=RolloutCostConfig(**cost_values)
    evaluator=NativeMujocoRolloutEvaluator(env,ec.control_seconds,cost,require_native=True,
                                         cache_directory=OUT/'native_cache')
    report=dict(purpose='One-plan hold-inclusion diagnosis, not a trained policy or production primitive',
        environment_contract=environment_contract(ec),prefix_commands=17,prefix=brief(info),
        recorded_state_match='qpos and qvel bit-for-bit',cem=asdict(config),cost=asdict(cost),
        seed=42,workers=1,prior_extraction=prior.metadata['extraction'],
        old_open_loop_sequence_available=False,
        continuation_caveat='Only executed future controls were saved; they are not the original open-loop CEM forecast.',
        fresh_plan_caveat='Planner resets its nominal trajectory at the current target; original iteration-18 warmstart/RNG state is unavailable.',
        sources={str(path.relative_to(ROOT)):sha256(path) for path in [PRIOR,SOURCE/'transitions.npz',
            ROOT/'mpc_dex/env.py',ROOT/'mpc_dex/mpc.py',ROOT/'mpc_dex/rollouts.py',
            ROOT/'mpc_dex/native_rollouts.py',ROOT/'mpc_dex/native/exact_rollout.cc',Path(__file__)]})
    snapshot=full_state(env)
    held_target=env.data.ctrl.copy()
    hold=np.broadcast_to(held_target,(config.horizon_steps,16)).copy()
    recorded_continuation=env.to_joint_targets(actions[17:17+config.horizon_steps])
    old_next_hold=np.broadcast_to(env.to_joint_targets(actions[17]),hold.shape).copy()
    sequences=dict(exact_current_target_hold=hold,
                   recorded_next_five_executed_controls=recorded_continuation,
                   recorded_next_command_held_constant=old_next_hold)
    comparisons={}
    batches=[]
    started=time.monotonic()
    try:
        for name,sequence in sequences.items():
            value=float(evaluator(sequence[None])[0]);diagnostics=evaluator.last_diagnostics
            comparisons[name]=dict(cost=value if np.isfinite(value) else None,feasible=bool(np.isfinite(value)),
                rejection_reason=diagnostics['rejection_reasons'][0],
                evaluated_physics_steps=diagnostics['candidate_physics_steps'][0],
                raw_final_progress_range=diagnostics['raw_final_progress_range'],
                retained_final_progress_range=diagnostics['retained_final_progress_range'],
                final_simulation_time=float(evaluator.worker_data[0].time),
                final_palm_support_force=env.palm_support_force(evaluator.worker_data[0]))
            np.testing.assert_array_equal(full_state(env),snapshot)
        report['comparisons']=comparisons
        print(json.dumps(dict(prefix=report['prefix'],comparisons=comparisons)),flush=True)
        planner=PrimitiveCEM(prior,env.low,env.high,config,seed=42)
        planner.reset(held_target)
        decoded_nominal=planner._decode(planner.nominal_knots[None])[0]
        report['initial_nominal_max_difference_from_hold']=float(np.max(np.abs(decoded_nominal-hold)))

        def evaluate_candidates(candidate_sequences):
            values=evaluator(candidate_sequences)
            batches.append(dict(sequences=candidate_sequences.copy(),costs=values.copy(),
                diagnostics=evaluator.last_diagnostics.copy()))
            return values

        try:
            selected,diagnostics=planner.plan(evaluate_candidates)  # The sole CEM plan in this task.
        except NoFeasiblePlan as error:
            report['fresh_plan']=dict(feasible=False,error=str(error))
        else:
            selected_sequence=planner.last_sequence.copy()
            sequences['fresh_selected_sequence']=selected_sequence
            selected_cost=float(evaluator(selected_sequence[None])[0])
            assert np.isclose(selected_cost,diagnostics['best_cost'],rtol=1e-10,atol=1e-10)
            np.testing.assert_array_equal(full_state(env),snapshot)
            _,reward,terminated,truncated,selected_info=env.step(env.to_action(selected))
            report['fresh_plan']=dict(feasible=True,cost=selected_cost,planner=diagnostics,
                command_max_difference_from_hold=float(np.max(np.abs(selected-held_target))),
                command_l2_difference_from_hold=float(np.linalg.norm(selected-held_target)),
                first_command_after_execution=brief(selected_info),reward=reward,
                terminated=terminated,truncated=truncated)
        hold_cost=comparisons['exact_current_target_hold']['cost']
        report['sampled_batches']=[]
        for iteration,batch in enumerate(batches):
            valid=np.isfinite(batch['costs'])
            distances=np.max(np.abs(batch['sequences']-hold),axis=(1,2))
            report['sampled_batches'].append(dict(iteration=iteration,candidates=len(valid),
                valid_candidates=int(valid.sum()),
                best_cost=float(batch['costs'][valid].min()) if valid.any() else None,
                minimum_sequence_max_difference_from_hold=float(distances.min()),
                exact_hold_candidates=int((distances<1e-12).sum()),
                sampled_candidates_cheaper_than_hold=int((batch['costs']<hold_cost).sum()) if hold_cost is not None else None,
                rejection_counts=dict(Counter(str(x) for x in batch['diagnostics']['rejection_reasons']))))
        report['plan_calls']=1;report['wall_seconds']=time.monotonic()-started
        report['native_build']=evaluator.native_provenance
        np.savez_compressed(OUT/'sequences.npz',**sequences,held_target=held_target,
            prefix_integration_state=snapshot,
            **{f'iteration_{i}_sequences':batch['sequences'] for i,batch in enumerate(batches)},
            **{f'iteration_{i}_costs':batch['costs'] for i,batch in enumerate(batches)})
        (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps({key:report[key] for key in ('fresh_plan','sampled_batches','wall_seconds')}),flush=True)
    finally:
        evaluator.close();env.close()


if __name__=='__main__':main()
