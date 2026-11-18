"""Fill the runtime summary from the runner's already-recorded provenance."""
from pathlib import Path
import json
import hashlib

D=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
report=json.loads((D/'report.json').read_text())
state=json.loads((D/'runner/run.json').read_text())
assert report['runtime'] is None
report['runtime']=state['provenance']['inference_runtime']
report['metadata_correction']='smoke.py read numerical_runtime; actual runner key is inference_runtime. Copied committed provenance; no physics or learning repeated.'
report['metadata_finalize_sha256']=sha(Path(__file__))
(D/'report.json').write_text(json.dumps(report,indent=2)+'\n')
(D/'manifest.json').write_text(json.dumps({str(p.relative_to(D)):sha(p) for p in D.rglob('*')
    if p.is_file() and p.name not in ('manifest.json','run.log') and '__pycache__' not in p.parts},indent=2)+'\n')
