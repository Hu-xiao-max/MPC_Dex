"""Restore the pinned Menagerie assets and verify their recorded hashes."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

root=Path(__file__).resolve().parents[1]/'assets/allegro'
manifest=json.loads((root/'upstream.json').read_text())
base=f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{manifest['commit']}/{manifest['path']}/"


def fetch(item):
    name,expected=item
    path=root/name
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==expected:return name
    body=urllib.request.urlopen(base+name,timeout=45).read()
    if hashlib.sha256(body).hexdigest()!=expected:raise ValueError(f'Asset checksum mismatch: {name}')
    path.parent.mkdir(exist_ok=True,parents=True)
    path.write_bytes(body)
    return name


if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name in pool.map(fetch,manifest['files_sha256'].items()):print(f'Verified {name}')
