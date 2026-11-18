"""Independently replay this one prescribed mechanism test, including final hold."""
from pathlib import Path
import json
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from mpc_dex.cli import read_config
from mpc_dex.contracts import environment_contract,sha256
from mpc_dex.env import AllegroRotationEnv
from scripts.audit_policy import audit_episode,PHYSICS_COLUMNS
from scripts.audit_recorded_controls import RecordedPolicy

artifact=Path(__file__).resolve().parent
source=artifact/'transitions.npz'
config=artifact/'source/config.json'
probe=json.loads((artifact/'report.json').read_text())
assert probe['status']!='running'
baseline_probe=json.loads((ROOT/'artifacts/mpc_matched40_bound098_0001/report.json').read_text())
unchanged_keys=['environment_contract','diagnostic_basis','cem','cost','seed','action_search',
                'reference_candidates','sampling','requested_backend']
unchanged={key:probe[key]==baseline_probe[key] for key in unchanged_keys}
assert all(unchanged.values()),unchanged
assert probe['max_target_delta']==.1 and baseline_probe['max_target_delta'] is None
with np.load(source,allow_pickle=False) as archive:
    expected={key:archive[key].copy() for key in archive.files}


def execute_and_audit(name,hold_steps):
    output=artifact/name;output.mkdir(exist_ok=False)
    cfg,ec=read_config(config)
    env=AllegroRotationEnv(ec)
    actions=expected['actions']
    if hold_steps:actions=np.concatenate([actions,np.repeat(actions[-1:],hold_steps,axis=0)])
    class Policy(RecordedPolicy):
        def act(self,obs,deterministic=True):
            self.last_observation=obs.copy()
            return super().act(obs,deterministic)
    policy=Policy(actions)
    values={key:[] for key in expected}
    original_step=env.step
    def observe_step(action):
        obs,reward,terminated,truncated,info=original_step(action)
        current=dict(obs=policy.last_observation,actions=action,rewards=reward,next_obs=obs,
                     qpos=env.data.qpos.copy(),qvel=env.data.qvel.copy())
        for key in values:values[key].append(np.asarray(current[key]).copy())
        return obs,reward,terminated,truncated,info
    env.step=observe_step
    try:
        report,physics,commands=audit_episode(env,policy,42,len(actions))
        actual={key:np.asarray(value) for key,value in values.items()}
        errors={key:float(np.max(np.abs(value[:len(expected[key])]-expected[key]))) for key,value in actual.items()}
        assert all(value==0 for value in errors.values()),errors
        with np.load(artifact/'physics.npz',allow_pickle=False) as recorded:
            report['original_physics_replay_max_error']=float(np.max(np.abs(physics[:len(recorded['samples'])]-recorded['samples'])))
        assert report['original_physics_replay_max_error']==0
        report.update(original_replay_max_errors=errors,source=str(source),source_sha256=sha256(source),
            environment_contract=environment_contract(ec),requested_hold_seconds=hold_steps*ec.control_seconds,
            completed_hold_seconds=max(0,(len(commands)-len(expected['actions']))*ec.control_seconds),
            completed_full_hold=len(commands)==len(actions),recorded_commands=len(expected['actions']),
            executed_commands=len(commands),production_eligible=False,policy_kind='recorded-control-replay',
            source_sha256_by_file={name:sha256(ROOT/name) for name in
                ['scripts/audit_policy.py','scripts/audit_recorded_controls.py','mpc_dex/env.py','mpc_dex/contracts.py']})
        for relative in report['source_sha256_by_file']:
            destination=output/'source'/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/relative,destination)
        np.savez_compressed(output/'transitions.npz',**actual)
        np.savez_compressed(output/'physics.npz',samples=physics,columns=np.asarray(PHYSICS_COLUMNS))
        (output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        return report,physics
    finally:
        env.step=original_step
        env.close()


exact,exact_physics=execute_and_audit('independent_exact_replay',0)
held,held_physics=execute_and_audit('independent_after_hold_2s',20)
baseline_root=ROOT/'artifacts/bc_40mm_bound098_0001'
baseline=json.loads((baseline_root/'exact_60/report.json').read_text())
baseline_hold=json.loads((baseline_root/'after_full60_hold/report.json').read_text())
keys=['steps','final_raw_angle','final_qualified_angle','final_force_qualified_angle',
      'max_raw_angle','max_qualified_angle','max_force_qualified_angle','max_target_delta',
      'max_command_rate','max_object_angular_speed','max_joint_speed','max_object_displacement',
      'final_axis_tilt_radians','final_force_bearing_fingers','independent_drop',
      'independent_palm_support_violation','finger_supported_rotation_success']
comparison=dict(purpose='One prespecified target-slew mechanism test; no sweep or training',
    changed_setting=dict(max_target_delta_radians_per_control=.1),
    baseline_settings_unchanged=unchanged,
    reference=dict(config=probe['source_hashes']['config'],prior=probe['source_hashes']['input_trajectory']),
    slew={key:exact[key] for key in keys},baseline={key:baseline[key] for key in keys},
    slew_after_hold={key:held[key] for key in keys},baseline_after_hold={key:baseline_hold[key] for key in keys},
    requested_hold_seconds=2.,completed_slew_hold_seconds=held['completed_hold_seconds'],
    max_original_replay_error=max(exact['original_replay_max_errors'].values()),
    original_physics_replay_max_error=exact['original_physics_replay_max_error'],
    probe_wall_seconds=probe['wall_seconds'],
    command_slew_quantization_excess=max(0.,exact['max_target_delta']-.1),
    interpretation='Diagnostic comparison only; success criteria and physical dynamics unchanged.')
(artifact/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
print(json.dumps(comparison,indent=2))
