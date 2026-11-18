"""One prescribed physical branch omitting the near-stationary first control."""
from pathlib import Path
from dataclasses import replace
import json
import shutil
import sys

D = Path(__file__).resolve().parent
ROOT = D.parents[1]
sys.path.insert(0, str(D))
from audit_helpers import run_audit
import numpy as np
from mpc_dex.cli import read_config
from mpc_dex.contracts import sha256, environment_contract
from scripts.audit_policy import write_json
from scripts.audit_recorded_controls import RecordedPolicy

source = ROOT / 'artifacts/mpc_polished_actor_recovery_0001/nominal_prefix_corrections_0001/strict_audit/transitions.npz'
config = ROOT / 'artifacts/mpc_polished_actor_recovery_0001/config.json'
assert not (D/'report.json').exists()
shutil.copyfile(source, D/'source_transitions.npz')
shutil.copyfile(config, D/'original_config.json')
cfg = json.loads(config.read_text())
cfg['environment']['initial_grasp'] = str(D/'source/assets/initial_grasp_40mm.npz')
write_json(D/'config.json', cfg)
_, ec = read_config(D/'config.json')
with np.load(source, allow_pickle=False) as archive:
    original = archive['actions'].copy()
assert original.shape == (9,16) and original.dtype == np.float32
actions = original[1:].copy()
np.savez_compressed(D/'prescribed_controls.npz', actions=actions)
inputs = {str(p.relative_to(D)): sha256(p) for p in D.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
report, executed, observations = run_audit(RecordedPolicy(actions), ec, D/'strict_audit', len(actions))
assert np.array_equal(executed, actions[:len(executed)])
hold = None
if report['finger_supported_rotation_success']:
    continued = np.concatenate((executed, np.repeat(executed[-1:],20,axis=0)))
    hold, _, _ = run_audit(RecordedPolicy(continued), replace(ec, terminate_on_success=False), D/'post_success_hold_2s', len(continued))
    with np.load(D/'post_success_hold_2s/physics.npz', allow_pickle=False) as archive:
        x, columns = archive['samples'], archive['columns'].tolist()
    after = x[len(executed)*round(ec.control_seconds/ec.timestep):]
    hold['completed_two_seconds_without_failure'] = bool(len(after)==1000 and not hold['independent_drop'] and not hold['independent_palm_support_violation'])
    hold['minimum_hold_loaded_fingers'] = int(after[:,columns.index('force_bearing_fingers')].min()) if len(after) else None
    write_json(D/'post_success_hold_2s/report.json',hold)
assert all(sha256(D/name)==digest for name,digest in inputs.items())
write_json(D/'report.json', dict(purpose='One cold-reset branch omitting the initial nominal hold; no replanning, clipping, training or state edits', production_eligible=False, policy_only_success=False, source_sha256=sha256(source), original_controls=9, prescribed_controls=8, omitted_source_control=1, environment_contract=environment_contract(ec), strict_audit=report, post_success_hold_2s=hold, input_hashes=inputs))
write_json(D/'manifest.json', {str(p.relative_to(D)):sha256(p) for p in D.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ('manifest.json','run.log')})
print(json.dumps({k:report[k] for k in ('steps','finger_supported_rotation_success','environment_failure_reason','max_force_qualified_angle','final_force_qualified_angle')},indent=2))
