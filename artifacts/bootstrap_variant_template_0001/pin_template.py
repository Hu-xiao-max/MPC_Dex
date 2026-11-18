"""Write the run.py / test_protocol.py digests into prepare.py PINS (deliberate template freeze)."""
from pathlib import Path
import hashlib, re
T = Path(__file__).resolve().parent
pins = {name: hashlib.sha256((T / name).read_bytes()).hexdigest() for name in ('run.py', 'test_protocol.py')}
text = (T / 'prepare.py').read_text()
new = 'PINS = ' + repr(pins) + '  # pinned by pin_template.py'
text, n = re.subn(r'^PINS = .*$', new, text, count=1, flags=re.M)
assert n == 1
(T / 'prepare.py').write_text(text)
print(pins)
