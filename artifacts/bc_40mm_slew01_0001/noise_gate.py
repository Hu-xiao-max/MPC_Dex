"""Three prespecified command-noise cases gate any subsequent actor fitting."""
from pathlib import Path
from dataclasses import asdict
import json,sys
D=Path(__file__).resolve().parent
sys.path.insert(0,str(D/'source'))
import numpy as np
from mpc_dex.env import EnvConfig
from mpc_dex.contracts import sha256
from scripts.audit_policy import write_json
from scripts.audit_recorded_controls import RecordedPolicy
from audit import run_audit

completion=json.loads((D/'source_completion.json').read_text())
assert completion['generation_complete'] and completion['planner_agent_audits_complete']
validated=json.loads((D/'audit_report.json').read_text())
assert validated['bc_eligible_strict_prefix'] and validated['torch_clamped_actions_bitwise_unchanged']
with np.load(D/'original_transitions.npz',allow_pickle=False) as a:original=a['actions'].copy()
ec=EnvConfig(**json.loads((D/'strict_config.json').read_text())['environment'])
report=dict(purpose='Three prespecified command-noise replays gate the optional paper-settings BC diagnostic',
    source_sha256=sha256(D/'original_transitions.npz'),script_sha256=sha256(__file__),
    noise_seeds=[17011,17012,17013],uniform_noise_amplitude=1e-4,
    formula='(original.astype(float64)+default_rng(seed).uniform(-1e-4,1e-4,original.shape)).astype(float32)',
    extra_action_clipping_or_slew_projection=False,environment=asdict(ec),status='running',conditions=[],
    production_eligible=False,training_updates=0)
write_json(D/'noise_gate_report.json',report)
for seed in report['noise_seeds']:
    actions=(original.astype(np.float64)+np.random.default_rng(seed).uniform(-1e-4,1e-4,original.shape)).astype(np.float32)
    assert abs(actions).max()<1
    output=D/f'noise_0001_seed{seed}'
    result,_,_=run_audit(RecordedPolicy(actions),ec,output,len(actions))
    np.savez_compressed(output/'supplied_controls.npz',actions=actions)
    result.update(noise_seed=seed,uniform_noise_amplitude=1e-4,
        supplied_controls=len(actions),strict_first_turn_stopping=True,
        action_error_max=float(abs(actions-original).max()),
        action_error_rms=float(np.sqrt(np.mean((actions-original)**2))))
    write_json(output/'report.json',result);report['conditions'].append(result)
    write_json(D/'noise_gate_report.json',report)
    print(json.dumps({k:result[k] for k in ['noise_seed','steps','finger_supported_rotation_success',
        'final_force_qualified_angle','max_force_qualified_angle','environment_failure_reason']}),flush=True)
report.update(status='complete',qualified_cases=sum(x['finger_supported_rotation_success'] for x in report['conditions']))
report['all_three_qualified']=report['qualified_cases']==3
report['authorized_next_step']='paper_bc_then_single_head_refinement' if report['all_three_qualified'] else 'stop_without_training'
write_json(D/'noise_gate_report.json',report)
