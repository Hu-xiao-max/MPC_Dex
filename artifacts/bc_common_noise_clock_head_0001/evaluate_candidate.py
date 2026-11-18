"""One separately approved strict clock-policy audit, with conditional2s continuation."""
from pathlib import Path
import json,sys
D=Path(__file__).resolve().parent
sys.path.insert(0,str(D/'source'))
import torch
from mpc_dex.learning import SACAgent
from mpc_dex.env import EnvConfig
from mpc_dex.contracts import sha256
from scripts.audit_policy import write_json
from train import evaluate

def main():
    assert not (D/'independent_audit').exists()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    report=json.loads((D/'report.json').read_text())
    assert report['status']=='candidate_ready_physics_review_pending'
    assert sha256(D/'candidate.pt')==report['candidate_checkpoint_sha256']
    inputs=json.loads((D/'inputs_manifest.json').read_text())
    assert all(sha256(D/name)==digest for name,digest in inputs.items())
    # Required explicit authorization record is written only after parent review.
    gate=json.loads((D/'physical_audit_authorization.json').read_text())
    assert gate['candidate_checkpoint_sha256']==report['candidate_checkpoint_sha256'] and gate['approved'] is True
    agent=SACAgent.load(D/'candidate.pt','cuda')
    assert agent.obs_dim==61 and agent.bc_updates==12000 and agent.pretrain_updates==agent.online_updates==0
    ec=EnvConfig(**json.loads((D/'strict_config.json').read_text())['environment'])
    outcome=evaluate(agent,ec,D/'independent_audit','diagnostic-clock73-bc-plus-frozen-head-correction')
    assert sha256(D/'candidate.pt')==report['candidate_checkpoint_sha256']
    assert all(sha256(D/name)==digest for name,digest in inputs.items())
    report.update(status='complete',**outcome,physical_evaluations=1+int('continuous_policy_aftermath' in outcome),
        source_inputs_unchanged=True,additional_solves=0,additional_bc_updates=0,
        additional_sac_updates=0,physical_audit_authorization_sha256=sha256(D/'physical_audit_authorization.json'))
    write_json(D/'report.json',report)
    write_json(D/'manifest.json',{str(p.relative_to(D)):sha256(p) for p in sorted(D.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p.name not in ['manifest.json','evaluate.log']})
    print(json.dumps(outcome),flush=True)

if __name__=='__main__':main()
