from pathlib import Path
import numpy as np,torch,json,hashlib,shutil
p=Path('/home/autel001/code/mpc_dex/artifacts/bootstrap_0005_centered_critic_calibration_0001');a=torch.load(p/'inputs/initial_agent.pt',weights_only=True,map_location='cpu');b=torch.load(p/'centered_initial.pt',weights_only=True,map_location='cpu');center=json.loads((p/'centering.json').read_text())['empirical_training_target_mean']
changed=[]
def walk(a,b,path=''):
 if isinstance(a,torch.Tensor):
  if not torch.equal(a,b):changed.append(path)
 elif isinstance(a,dict):
  assert a.keys()==b.keys()
  for k in a:walk(a[k],b[k],path+'/'+str(k))
 elif isinstance(a,(tuple,list)):
  assert len(a)==len(b)
  for i,(x,y) in enumerate(zip(a,b)):walk(x,y,path+'/'+str(i))
 else:assert a==b,(path,a,b)
walk(a,b);expected={f'/{family}/networks.{i}.1.bias' for family in ('critics','target_critics') for i in (0,1)};assert set(changed)==expected
for family in ('critics','target_critics'):
 for i in (0,1):np.testing.assert_array_equal((a[family][f'networks.{i}.1.bias']+center).numpy(),b[family][f'networks.{i}.1.bias'].numpy())
actual=np.load(p/'minibatch_indices.npy');rng=np.random.default_rng(17051);expected=np.asarray([rng.integers(256,size=256) for _ in range(500)]);np.testing.assert_array_equal(actual,expected)
r=dict(status='verified',initial_changes_only=changed,exact_added_train_mean=center,all_500_minibatches_match_original_seed_protocol=True,minibatch_indices_sha256=hashlib.sha256((p/'minibatch_indices.npy').read_bytes()).hexdigest())
(p/'paired_initialization_verification.json').write_text(json.dumps(r,indent=2)+'\n');shutil.copyfile(__file__,p/'verify_pair.py');print(json.dumps(r))
