"""One strict deterministic clock-policy audit after numerical parent review."""
from pathlib import Path
import json,sys
D=Path(__file__).resolve().parent
sys.path.insert(0,str(D/'source'))
import torch
from mpc_dex.learning import SACAgent
from mpc_dex.env import EnvConfig
from mpc_dex.contracts import sha256
from scripts.audit_policy import write_json
from audit_full_state import run_audit,full_turn_events

def main():
    assert not (D/'independent_audit').exists()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    report=json.loads((D/'report.json').read_text())
    assert report['status']=='candidate_ready_physics_review_pending'
    assert sha256(D/'candidate.pt')==report['candidate_checkpoint_sha256']
    inputs=json.loads((D/'inputs_manifest.json').read_text())
    assert all(sha256(D/name)==digest for name,digest in inputs.items())
    sources=['evaluate_candidate.py','audit_full_state.py','source/scripts/audit_policy.py',
        'source/mpc_dex/env.py','source/mpc_dex/learning.py','policy_interface.json','strict_config.json',
        'candidate.pt','candidate.pt.metadata.json']
    protocol=dict(parent_approved_after_numerical_review=True,strict_actual_policy_audits=1,
        max_controls=100,seed=42,stochastic_trials=0,post_success_continuations=0,
        extra_solves=0,optimizer_updates=0,device='cuda',torch_threads=1,torch_interop_threads=1,
        matmul_tf32=False,cudnn_tf32=False,clock='actual simulator elapsed time',
        integration_state_capture='reset and every actual control boundary',
        hashes_before_physics={name:sha256(D/name) for name in sources})
    write_json(D/'audit_execution_protocol.json',protocol)
    agent=SACAgent.load(D/'candidate.pt','cuda')
    before={name:t.detach().cpu().clone() for name,t in agent.actor.state_dict().items()}
    assert agent.obs_dim==61 and agent.bc_updates==12000 and agent.pretrain_updates==agent.online_updates==0
    ec=EnvConfig(**json.loads((D/'strict_config.json').read_text())['environment'])
    strict,_,_=run_audit(agent,ec,D/'independent_audit',100)
    strict.update(full_turn_events=full_turn_events(D/'independent_audit',ec),
        full_integration_state_recorded=True,bc_updates=12000,critic_updates=0,online_updates=0)
    write_json(D/'independent_audit/report.json',strict)
    assert all(sha256(D/name)==digest for name,digest in inputs.items())
    assert all(sha256(D/name)==digest for name,digest in protocol['hashes_before_physics'].items())
    assert all(torch.equal(t,agent.actor.state_dict()[name].detach().cpu()) for name,t in before.items())
    report.update(status='complete',strict=strict,physical_evaluations=1,post_success_continuations=0,
        source_inputs_unchanged=True,additional_solves=0,additional_bc_updates=0,additional_sac_updates=0,
        actual_policy_parameters_unchanged=True,audit_protocol_sha256=sha256(D/'audit_execution_protocol.json'))
    write_json(D/'report.json',report)
    write_json(D/'manifest.json',{str(p.relative_to(D)):sha256(p) for p in sorted(D.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p.name not in ['manifest.json','evaluate.log']})
    print(json.dumps(strict),flush=True)
if __name__=='__main__':main()
