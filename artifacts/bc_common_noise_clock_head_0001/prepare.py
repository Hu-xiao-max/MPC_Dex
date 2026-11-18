"""Freeze the completed61D BC actor before one separately authorized head solve."""
from pathlib import Path
import hashlib,json,shutil
D=Path(__file__).resolve().parent;OLD=D.parent/'bc_common_noise_clock_0001'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def main():
    manifest=json.loads((OLD/'manifest.json').read_text())
    assert all(sha(OLD/name)==digest for name,digest in manifest.items())
    shutil.copytree(OLD/'source',D/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (D/'inputs').mkdir()
    for name in ['manifest.json','summary.json','protocol.json','dataset_validation.json','measured_clock_validation.json',
                 'row_provenance.json','bc/report.json']:
        out=D/'inputs'/name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(OLD/name,out)
    for name in ['supervised_pairs.npz','policy_interface.json','train.py','audit.py']:
        shutil.copyfile(OLD/name,D/name)
    for source,name in [('bc/checkpoint_12000.pt','baseline.pt'),('bc/checkpoint_12000.pt.metadata.json','baseline.pt.metadata.json')]:
        shutil.copyfile(OLD/source,D/name);assert sha(OLD/source)==sha(D/name)
    config=json.loads((OLD/'strict_config.json').read_text());config['environment']['initial_grasp']=str(D/'source/assets/initial_grasp_40mm.npz')
    write(D/'strict_config.json',config)
    write(D/'provenance.json',dict(source_artifact=str(OLD),source_manifest_sha256=sha(OLD/'manifest.json'),
        baseline_sha256=sha(D/'baseline.pt'),dataset_sha256=sha(D/'supervised_pairs.npz'),
        prior_completed_bc_updates=12000,new_bc_updates=0,pretrain_updates=0,online_updates=0,
        authorized_residual_solves=1,physical_evaluation_authorized=False,
        method='One float64 minimum-norm residual solve with frozen features; cast/add correction to float32 head once',
        no_ridge_sweep=True,no_scale_sweep=True,no_iterative_solves=True,production_eligible=False))
    write(D/'inputs_manifest.json',{str(p.relative_to(D)):sha(p) for p in sorted(D.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p.name!='inputs_manifest.json'})
    print('Frozen baseline and73 unchanged targets; physics gate pending numerical review')
if __name__=='__main__':main()
