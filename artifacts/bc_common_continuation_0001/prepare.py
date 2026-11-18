"""Freeze verified common labels and independently replay both source branches."""
from pathlib import Path
from dataclasses import replace
import json, sys, platform, importlib.metadata
D = Path(__file__).resolve().parent
sys.path.insert(0, str(D / 'source'))
import numpy as np
import torch
import mujoco
from mpc_dex.env import EnvConfig
from mpc_dex.contracts import sha256
from scripts.audit_policy import write_json
from scripts.audit_recorded_controls import RecordedPolicy
from audit import run_audit


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    ec = EnvConfig(**json.loads((D/'strict_config.json').read_text())['environment'])
    with np.load(D/'inputs/original_teacher_transitions.npz') as a:
        teacher = {k:a[k].copy() for k in a.files}
    with np.load(D/'inputs/nominal/strict/transitions.npz') as a:
        nominal = {k:a[k].copy() for k in a.files}
    with np.load(D/'inputs/recovery/strict/transitions.npz') as a:
        recovery = {k:a[k].copy() for k in a.files}
    with np.load(D/'inputs/recovery_original_transitions.npz') as a:
        for k in ['obs','actions','next_obs','rewards','qpos','qvel']:
            assert np.array_equal(a[k],recovery[k]), k
    with np.load(D/'inputs/recovery_origins.npz') as a:
        assert np.array_equal(np.flatnonzero(a['actual_mpc_correction_mask']),np.arange(6,9))
    assert nominal['actions'].shape == recovery['actions'].shape == (9,16)
    for k in ['obs','actions','next_obs','rewards','qpos','qvel']:
        assert np.array_equal(nominal[k][:6],teacher[k][:6]), k
    assert np.array_equal(nominal['actions'][6:],recovery['actions'][6:])
    obs = np.concatenate([teacher['obs'][:6], nominal['obs'][6:], recovery['obs'][6:]])
    actions = np.concatenate([teacher['actions'][:6],nominal['actions'][6:],recovery['actions'][6:]])
    assert obs.shape==(12,60) and actions.shape==(12,16)
    clamped = torch.as_tensor(actions).clamp(-.98,.98).numpy()
    assert clamped.tobytes()==actions.tobytes()
    np.savez_compressed(D/'supervised_pairs.npz',obs=obs,actions=actions,
        label_group=np.asarray(['original_teacher_prefix']*6+['nominal_mpc_corrections']*3+['recovery_mpc_corrections']*3),
        source_row_one_based=np.asarray(list(range(1,7))+[7,8,9]*2))
    report=dict(status='validating',production_eligible=False,pairs=12,original_teacher_prefix_pairs=6,
        nominal_correction_pairs=3,recovery_correction_pairs=3,actor_prefix_labels=0,
        original_teacher_future_7_to_22_excluded=True,common_correction_labels_bitwise_equal=True,
        first_six_teacher_pairs_bitwise_equal=True,clamp_changed_scalars=0,
        source_script_sha256=sha256(__file__),sources={})
    for name,source in [('nominal',nominal),('recovery',recovery)]:
        base=D/'inputs'/name
        source_report=json.loads((base/'strict/report.json').read_text())
        hold_report=json.loads((base/'hold_2s/report.json').read_text())
        assert source_report['finger_supported_rotation_success'] and source_report['steps']==9
        assert hold_report['completed_two_seconds_without_failure'] and hold_report['steps']==29
        assert not source_report['independent_drop'] and not source_report['independent_palm_support_violation']
        checked={}
        for label,ec_here,count in [('strict',ec,9),('hold_2s',replace(ec,terminate_on_success=False),29)]:
            source_dir=base/label
            with np.load(source_dir/'transitions.npz') as a:
                actual={k:a[k].copy() for k in a.files}
            policy=RecordedPolicy(actual['actions'])
            out=D/f'verified_{name}_{label}'
            result,_,_=run_audit(policy,ec_here,out,count)
            assert result['steps']==count and result['finger_supported_rotation_success']
            assert not result['independent_drop'] and not result['independent_palm_support_violation']
            with np.load(out/'transitions.npz') as b:
                fields=['obs','actions','next_obs','rewards','qpos','qvel','terminated','truncated','targets']
                for k in fields:
                    if k in actual: assert np.array_equal(actual[k],b[k]),(name,label,k)
            with np.load(out/'physics.npz') as b,np.load(source_dir/'physics.npz') as a:
                assert np.array_equal(a['samples'],b['samples'])
                assert np.array_equal(a['columns'],b['columns'])
            checked[label]=dict(bitwise_transition_and_physics_replay=True,verified_fields=[k for k in fields if k in actual],
                source_report_sha256=sha256(source_dir/'report.json'),report=result)
        report['sources'][name]=checked
    report.update(status='complete',data_sha256=sha256(D/'supervised_pairs.npz'))
    write_json(D/'dataset_validation.json',report)
    runtime=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
        torch=str(torch.__version__),mujoco=mujoco.__version__,numpy=np.__version__,
        cuda_version=torch.version.cuda,cuda_available=torch.cuda.is_available(),
        cuda_device=torch.cuda.get_device_name(0),device_properties=str(torch.cuda.get_device_properties(0)),
        packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']})
    write_json(D/'runtime.json',runtime)
    manifest={str(p.relative_to(D)):sha256(p) for p in sorted(D.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p.name not in ['inputs_manifest.json','prepare.log']}
    write_json(D/'inputs_manifest.json',manifest)
    print(json.dumps(dict(status='complete',pairs=12,common_labels_equal=True,
        physical_replays_bitwise_equal=4,clamp_changes=0)),flush=True)


if __name__=='__main__':main()
