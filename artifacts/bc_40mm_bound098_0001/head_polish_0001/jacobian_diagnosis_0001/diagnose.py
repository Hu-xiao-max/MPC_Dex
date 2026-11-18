"""Read-only local actor sensitivity on recorded observations; no fitting/physics."""
from pathlib import Path
import hashlib
import json
import sys

sys.dont_write_bytecode=True
OUTPUT=Path(__file__).resolve().parent
POLISH=OUTPUT.parent
PARENT=POLISH.parent
FROZEN=PARENT/'source'
sys.path.insert(0,str(FROZEN))

import numpy as np
import torch
from mpc_dex.learning import SACAgent
import mpc_dex.learning as learning


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


GROUPS=[('joint_position',0,16,'radians'),('object_position',16,19,'metres'),
        ('object_quaternion',19,23,'unitless quaternion components wxyz'),
        ('principal_angles',23,25,'radians: current and previous'),
        ('joint_velocity',25,41,'radians/second: finite differences'),
        ('object_velocity',41,44,'metres/second: finite differences'),
        ('previous_action',44,60,'normalized absolute joint targets')]


def norm(value):return float(np.linalg.norm(value))
def maxabs(value):return float(np.max(np.abs(value)))


def main():
    if (OUTPUT/'report.json').exists():raise FileExistsError(OUTPUT/'report.json')
    assert Path(learning.__file__).resolve()==(FROZEN/'mpc_dex/learning.py').resolve()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    inputs={
        'checkpoint':POLISH/'polished.pt',
        'checkpoint_metadata':POLISH/'polished.pt.metadata.json',
        'teacher_transitions':PARENT/'strict_first_success/transitions.npz',
        'visited_transitions':POLISH/'independent_audit/transitions.npz',
        'visited_audit_report':POLISH/'independent_audit/report.json',
        'saved_teacher_predictions':POLISH/'actual_torch_teacher_predictions.npz',
        'frozen_learning_source':FROZEN/'mpc_dex/learning.py',
        'frozen_environment_source':FROZEN/'mpc_dex/env.py',
        'polish_report':POLISH/'report.json',
        'polish_manifest':POLISH/'manifest.json',
    }
    input_hashes={name:sha256(path) for name,path in inputs.items()}
    with np.load(inputs['teacher_transitions'],allow_pickle=False) as archive:
        teacher_obs=archive['obs'].copy();teacher_actions=archive['actions'].copy()
    with np.load(inputs['visited_transitions'],allow_pickle=False) as archive:
        visited_obs=archive['obs'][:8].copy();visited_actions=archive['actions'][:8].copy()
    with np.load(inputs['saved_teacher_predictions'],allow_pickle=False) as archive:
        saved_teacher_predictions=archive['polished'][:8].copy()
        np.testing.assert_array_equal(teacher_obs,archive['observations'])
        np.testing.assert_array_equal(teacher_actions,archive['teacher_actions'])
    assert teacher_obs.shape==(22,60) and visited_obs.shape==(8,60)
    agent=SACAgent.load(inputs['checkpoint'],'cuda')
    counters_before=(agent.bc_updates,agent.pretrain_updates,agent.online_updates)
    weights_before={name:value.detach().cpu().numpy().tobytes()
                    for name,value in agent.actor.state_dict().items()}
    for parameter in agent.actor.parameters():parameter.requires_grad_(False)
    def actor(observation):return agent.actor(observation)[0].tanh()
    teacher_predictions=[];visited_predictions=[];jacobians=[]
    for teacher,visited in zip(teacher_obs[:8],visited_obs):
        teacher_predictions.append(agent.act(teacher,deterministic=True))
        visited_predictions.append(agent.act(visited,deterministic=True))
        point=torch.as_tensor(teacher,device=agent.device,dtype=torch.float32)
        jacobian=torch.autograd.functional.jacobian(actor,point,create_graph=False,vectorize=False)
        jacobians.append(jacobian.detach().cpu().numpy())
    teacher_predictions=np.asarray(teacher_predictions)
    visited_predictions=np.asarray(visited_predictions)
    jacobians=np.asarray(jacobians)
    np.testing.assert_array_equal(teacher_predictions,saved_teacher_predictions)
    np.testing.assert_array_equal(visited_predictions,visited_actions)
    delta=visited_obs.astype(np.float64)-teacher_obs[:8].astype(np.float64)
    feature_contributions=jacobians.astype(np.float64)*delta[:,None,:]
    linear=feature_contributions.sum(axis=2)
    group_contributions=np.stack([feature_contributions[:,:,start:stop].sum(axis=2)
                                  for _,start,stop,_ in GROUPS],axis=1)
    np.testing.assert_allclose(group_contributions.sum(axis=1),linear,atol=1e-16,rtol=1e-14)
    actual=visited_predictions.astype(np.float64)-teacher_predictions.astype(np.float64)
    fit=teacher_predictions.astype(np.float64)-teacher_actions[:8].astype(np.float64)
    observed_error=visited_actions.astype(np.float64)-teacher_actions[:8].astype(np.float64)
    predicted_error=fit+linear
    residual=actual-linear
    np.testing.assert_array_equal(actual+fit,observed_error)
    # Check the local directional derivative without a simulator or fitted
    # parameters. The step-8 displacement is large enough to resolve float32
    # finite differences; a short fraction keeps the check local.
    point=torch.as_tensor(teacher_obs[7],device=agent.device)
    direction=torch.as_tensor(delta[7],device=agent.device,dtype=torch.float32)
    h=1/64
    with torch.no_grad():
        plus=actor(point+h*direction).cpu().numpy().astype(np.float64)
        minus=actor(point-h*direction).cpu().numpy().astype(np.float64)
    finite_difference=(plus-minus)/(2*h)
    directional_relative_error=norm(finite_difference-linear[7])/norm(linear[7])
    steps=[]
    for index in range(8):
        denominator=float(actual[index]@actual[index])
        predicted_energy=float(linear[index]@linear[index])
        attribution=[]
        for group,(name,start,stop,units) in enumerate(GROUPS):
            contribution=group_contributions[index,group]
            attribution.append(dict(group=name,observation_units=units,
                observed_group_delta_l2=norm(delta[index,start:stop]),
                observed_group_delta_max_abs=maxabs(delta[index,start:stop]),
                action_contribution_l2=norm(contribution),action_contribution_max_abs=maxabs(contribution),
                projection_fraction_of_predicted_change=(float(contribution@linear[index])/predicted_energy
                                                         if predicted_energy else None),
                projection_fraction_of_actual_change=(float(contribution@actual[index])/denominator
                                                      if denominator else None)))
        steps.append(dict(control=index+1,
            teacher_fit_error_max_abs=maxabs(fit[index]),
            observed_teacher_action_error_max_abs=maxabs(observed_error[index]),
            actual_state_induced_action_change_l2=norm(actual[index]),
            actual_state_induced_action_change_max_abs=maxabs(actual[index]),
            linear_action_change_l2=norm(linear[index]),linear_action_change_max_abs=maxabs(linear[index]),
            predicted_teacher_action_error_max_abs=maxabs(predicted_error[index]),
            taylor_residual_l2=norm(residual[index]),taylor_residual_max_abs=maxabs(residual[index]),
            relative_l2_taylor_error=(norm(residual[index])/np.sqrt(denominator) if denominator else None),
            squared_change_explained=(1-float(residual[index]@residual[index])/denominator if denominator else None),
            cosine_alignment=(float(linear[index]@actual[index])/np.sqrt(denominator*predicted_energy)
                              if denominator and predicted_energy else None),
            groups=attribution))
    def aggregate(indices):
        observed=actual[indices];predicted=linear[indices];remaining=residual[indices]
        denominator=float(np.sum(observed**2));predicted_energy=float(np.sum(predicted**2))
        return dict(controls=(np.arange(1,9)[indices]).tolist(),
            relative_l2_taylor_error=norm(remaining)/np.sqrt(denominator),
            squared_change_explained=1-float(np.sum(remaining**2))/denominator,
            group_projection_fraction_of_predicted_change={name:float(np.sum(group_contributions[indices,g]*predicted))/predicted_energy
                for g,(name,_,_,_) in enumerate(GROUPS)},
            group_projection_fraction_of_actual_change={name:float(np.sum(group_contributions[indices,g]*observed))/denominator
                for g,(name,_,_,_) in enumerate(GROUPS)})
    np.savez_compressed(OUTPUT/'arrays.npz',teacher_observations=teacher_obs[:8],
        visited_observations=visited_obs,teacher_actions=teacher_actions[:8],
        teacher_policy_actions=teacher_predictions,visited_policy_actions=visited_predictions,
        jacobian_float32=jacobians,observation_delta_float64=delta,
        per_feature_action_contributions=feature_contributions,group_action_contributions=group_contributions,
        group_names=np.asarray([name for name,_,_,_ in GROUPS]),
        actual_state_induced_action_change=actual,linear_state_induced_action_change=linear,
        teacher_fit_error=fit,observed_teacher_action_error=observed_error,
        predicted_teacher_action_error=predicted_error,taylor_residual=residual,
        step8_directional_finite_difference=finite_difference)
    assert counters_before==(agent.bc_updates,agent.pretrain_updates,agent.online_updates)
    assert all(value.detach().cpu().numpy().tobytes()==weights_before[name]
               for name,value in agent.actor.state_dict().items())
    assert input_hashes=={name:sha256(path) for name,path in inputs.items()}
    report=dict(purpose='Local frozen-actor response to recorded off-teacher states; no fitting or physics',
        controls=8,teacher_controls=22,device=str(agent.device),torch_version=str(torch.__version__),
        numpy_version=np.__version__,cuda_device=torch.cuda.get_device_name(agent.device),
        torch_threads=torch.get_num_threads(),tf32=False,
        source_choice='Archived actor implementation and checkpoint used by the actual singleton CUDA audit',
        teacher_and_visited_actions_match_saved_values_bitwise=True,checkpoint_and_inputs_unchanged=True,
        actor_weights_bitwise_unchanged=True,optimizer_updates=0,physics_steps=0,
        definitions=dict(jacobian='d tanh(actor_mean) / d observation, evaluated separately at each teacher observation',
            actual_change='policy(visited observation) - policy(teacher observation)',
            linear_change='teacher Jacobian times (visited observation - teacher observation)',
            total_error='teacher-state fitting error plus state-induced action change',
            squared_change_explained='1 - ||actual_change-linear_change||^2 / ||actual_change||^2',
            attribution='J_group * observed_delta_group, expressed in common normalized-action units; signed projection shares sum to one for predicted change',
            caution='Raw observation norms and Jacobian column magnitudes have mixed units and are not ranked. Correlated groups and signed cancellation prevent causal-importance interpretation. Autograd ignores floating-point rounding discontinuities.'),
        groups=[dict(name=name,start=start,stop=stop,units=units) for name,start,stop,units in GROUPS],
        steps=steps,aggregate_controls_2_to_4=aggregate(slice(1,4)),
        aggregate_controls_5_to_8=aggregate(slice(4,8)),aggregate_controls_2_to_8=aggregate(slice(1,8)),
        derivative_check=dict(control=8,central_difference_fraction=h,
            relative_l2_error=directional_relative_error,
            note='Arithmetic check using policy evaluations only, no new physical transitions'),
        input_files={name:dict(path=str(path),sha256=input_hashes[name]) for name,path in inputs.items()},
        script_sha256=sha256(__file__),arrays_sha256=sha256(OUTPUT/'arrays.npz'))
    (OUTPUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:report[key] for key in ['aggregate_controls_2_to_4','aggregate_controls_5_to_8','derivative_check']},indent=2))
    for row in steps:
        print(json.dumps({key:row[key] for key in ['control','observed_teacher_action_error_max_abs',
            'linear_action_change_max_abs','relative_l2_taylor_error','squared_change_explained','cosine_alignment']}))


if __name__=='__main__':main()
