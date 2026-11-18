"""Real CUDA actor-import verification; zero bootstrap interactions or learning."""
from pathlib import Path
import json
import shutil
import sys

D = Path(__file__).resolve().parent
ROOT = D.parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from mpc_dex.contracts import sha256
from mpc_dex.env import AllegroRotationEnv
from scripts.train_bootstrap import BootstrapRunner, parser
from scripts.audit_policy import audit_episode, PHYSICS_COLUMNS

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32=False
source = ROOT/'artifacts/bc_common_continuation_0001'
checkpoint = source/'head_polish/polished.pt'
config = json.loads((source/'strict_config.json').read_text())
config['bootstrap'] = {'learner':{'hidden_sizes':[256,256],'initial_std':.05}}
(D/'config.json').write_text(json.dumps(config,indent=2)+'\n')
numerical = ['scripts/train_bootstrap.py','scripts/audit_policy.py','mpc_dex/env.py',
             'mpc_dex/learning.py','mpc_dex/contracts.py','mpc_dex/bootstrap_env.py','mpc_dex/replay.py','uv.lock']
hashes = {name:sha256(ROOT/name) for name in numerical}
for name in numerical:
    target=D/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(ROOT/name,target)
args = parser().parse_args(['--config',str(D/'config.json'),'--steps','0','--output',str(D/'runner'),
    '--initial-actor',str(checkpoint),'--num-envs','1','--seed','17031','--device','cuda','--torch-threads','1'])
runner = BootstrapRunner(args)
with np.load(source/'supervised_pairs.npz',allow_pickle=False) as data:obs=data['obs'].copy()
with np.load(source/'head_polish/solve_inputs_and_result.npz',allow_pickle=False) as data:expected=data['polished_predictions'].copy()
predicted=np.array([runner.agent.act(row,True) for row in obs])
batch_one=np.array([runner.agent.act(row[None],True)[0] for row in obs])
np.testing.assert_array_equal(predicted,expected)
np.testing.assert_array_equal(batch_one,expected)
payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
for key,value in runner.agent.actor.state_dict().items():
    if key.startswith(('trunk.','mean_head.')):assert torch.equal(value.cpu(),payload['actor'][key])
assert runner.agent.bc_updates==12000 and runner.agent.pretrain_updates==runner.agent.online_updates==0
assert all(not getattr(runner.agent,name).state for name in ('bc_optimizer','actor_optimizer','critic_optimizer','alpha_optimizer'))
assert runner.agent.config.initial_std==.05 and runner.demonstration_successes==runner.successes==0
assert runner.run()==0
env=AllegroRotationEnv(runner.ec)
try: report,physics,commands=audit_episode(env,runner.agent,42,100,True)
finally: env.close()
assert report['finger_supported_rotation_success'] and report['steps']==9
expected_commands=json.loads((source/'head_polish/independent_audit/commands.json').read_text())
np.testing.assert_array_equal(np.asarray([x['policy_action'] for x in commands]),np.asarray([x['policy_action'] for x in expected_commands]))
with np.load(source/'head_polish/independent_audit/physics.npz',allow_pickle=False) as data:
    np.testing.assert_array_equal(physics,data['samples'])
np.savez_compressed(D/'physics.npz',samples=physics,columns=np.asarray(PHYSICS_COLUMNS))
(D/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
(D/'audit_report.json').write_text(json.dumps(report,indent=2)+'\n')
assert all(sha256(ROOT/name)==digest for name,digest in hashes.items())
result=dict(purpose='Fresh real CUDA bootstrap actor-import smoke; no training',
    source_checkpoint_sha256=sha256(checkpoint),numerical_source_sha256=hashes,
    source_actor_means_and_actions_bitexact=True,cuda_singleton_and_batch_one_actions_bitexact=True,
    all_nine_executed_actions_and_all_physics_samples_match_source=True,
    imported_bc_updates=12000,bootstrap_bc_updates=0,pretrain_updates=0,online_updates=0,
    bootstrap_environment_steps=0,bootstrap_successes=0,demonstration_successes=0,
    configured_latent_std=.05,all_optimizers_fresh=True,strict_policy_audit=report,
    runtime=runner.provenance.get('numerical_runtime'),
    production_eligible=False,primitive_eligible=False,main_stage_eligible=False)
(D/'report.json').write_text(json.dumps(result,indent=2)+'\n')
(D/'manifest.json').write_text(json.dumps({str(p.relative_to(D)):sha256(p) for p in D.rglob('*')
    if p.is_file() and p.name not in ('manifest.json','run.log') and '__pycache__' not in p.parts},indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('source_actor_means_and_actions_bitexact','cuda_singleton_and_batch_one_actions_bitexact','online_updates','bootstrap_environment_steps')},indent=2))
