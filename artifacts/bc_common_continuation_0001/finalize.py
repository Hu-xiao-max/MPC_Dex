"""Validate and freeze completed evidence; no simulation or weight updates."""
from pathlib import Path
import hashlib,json
D=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads((D/path).read_text())
def write(path,value):(D/path).write_text(json.dumps(value,indent=2)+'\n')
inputs=read('inputs_manifest.json')
assert all(sha(D/p)==v for p,v in inputs.items())
bc=read('bc/report.json');head=read('head_polish/report.json');state=read('checkpoint_state.json')
assert bc['status']==head['status']=='complete'
assert bc['invariant_initial']==bc['invariant_final']
assert head['all_other_checkpoint_payload_values_bitwise_unchanged']
assert state['bc_updates']==12000 and state['pretrain_updates']==state['online_updates']==0
assert head['strict']['finger_supported_rotation_success']
assert not head['continuous_policy_aftermath']['completed_two_seconds_without_failure']
commands=[dict(command='.venv/bin/python artifacts/bc_common_continuation_0001/prepare.py',log='prepare.log',role='Four bitwise physical source validations and exact12-pair dataset'),
 dict(command='.venv/bin/python artifacts/bc_common_continuation_0001/train.py',log='train.log',role='Fresh BC12k with three prescribed strict policy audits'),
 dict(command='.venv/bin/python artifacts/bc_common_continuation_0001/polish.py',log='polish.log',role='One gated head solve and conditional continued-policy audit'),
 dict(command='.venv/bin/python artifacts/bc_common_continuation_0001/finalize.py',role='Read-only invariant checks and final artifact manifest')]
write('commands_executed.json',commands)
summary=dict(status='complete',production_eligible=False,trained_with_SAC=False,
 bounded_protocol=dict(source_verification_replays=4,bc_updates=12000,bc_checkpoint_evaluations=3,
  head_solves=1,head_policy_evaluations=1,continued_policy_evaluations=1,critic_updates=0,online_updates=0),
 dataset=dict(pairs=12,teacher_prefix6=6,nominal_corrections=3,recovery_corrections=3,
  common_correction_actions_bitwise_identical=True,actor_prefix_labels=0,teacher_future_7_to_22_excluded=True),
 strict_policy_first_turn_success=True,strict_success=head['strict'],
 continued_policy_two_seconds_success=False,continued_policy=head['continuous_policy_aftermath'],
 final_fit=head['polished_fit'],condition_number=head['condition_number'],
 correction_frobenius_norm=head['correction_frobenius_norm'],
 checkpoint=state,all_archived_inputs_unchanged=True,
 baseline_bc_checkpoint_unchanged=sha(D/'bc/checkpoint_12000.pt')==head['baseline_sha256'],
 checkpoint_sha256=sha(D/'head_polish/polished.pt'),checkpoint_metadata_sha256=sha(D/'head_polish/polished.pt.metadata.json'),
 interpretation='Qualified policy-only first turn from the fixed reset; actual continued policy drops. Not a successful SAC policy or production primitive.')
write('summary.json',summary)
manifest={str(p.relative_to(D)):sha(p) for p in sorted(D.rglob('*'))
 if p.is_file() and '__pycache__' not in p.parts and p!=D/'manifest.json'}
write('manifest.json',manifest)
print(json.dumps(dict(files=len(manifest),manifest_sha256=sha(D/'manifest.json'),
 summary_sha256=sha(D/'summary.json'),checkpoint_sha256=summary['checkpoint_sha256'])))
