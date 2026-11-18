"""Run the prescribed single slew diagnostic and preserve imported dependencies."""
from pathlib import Path
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import probe_mpc

output=ROOT/'artifacts/mpc_matched40_slew01_0001'
if output.exists():raise FileExistsError(output)
baseline=json.loads((ROOT/'artifacts/mpc_matched40_bound098_0001/report.json').read_text())
for name,digest in baseline['source_hashes'].items():
    if name in ('config','input_trajectory'):continue
    if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
        raise ValueError(f'Prespecified numerical source differs from baseline: {name}')
sources={}
for name,module in tuple(sys.modules.items()):
    if name.startswith(('mpc_dex.','scripts.')) and getattr(module,'__file__',None):
        path=Path(module.__file__).resolve()
        if path.suffix=='.py' and path.is_relative_to(ROOT):
            sources[str(path.relative_to(ROOT))]=path.read_bytes()
sys.argv=['scripts/probe_mpc.py',
    '--config',str(ROOT/'artifacts/mpc_matched40_bound098_0001/source/config.json'),
    '--trajectory',str(ROOT/'artifacts/mpc_matched40_bound098_0001/source/input_trajectory.npz'),
    '--output',str(output),'--steps','60','--seed','42','--workers','8',
    '--backend','native','--progress','retained','--perpendicular-weight','100',
    '--reference-candidates','--action-bound','.98','--max-target-delta','.1']
try:
    probe_mpc.main()
finally:
    if output.is_dir():
        root=output/'imported_dependencies'
        for name,contents in sources.items():
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(contents)
        (root/'manifest.json').write_text(json.dumps({
            'capture':'Source bytes of local modules already imported immediately before probe main',
            'argv':sys.argv,
            'sha256':{name:hashlib.sha256(contents).hexdigest() for name,contents in sources.items()},
            'source_changed_during_probe':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=hashlib.sha256(contents).hexdigest() for name,contents in sources.items()},
        },indent=2)+'\n')
