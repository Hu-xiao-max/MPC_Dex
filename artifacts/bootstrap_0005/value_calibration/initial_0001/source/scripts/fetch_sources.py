"""Snapshot public references and repository inventories with provenance."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request


SOURCES = {
    "paper.html": "https://arxiv.org/html/2609.14878v1",
    "project.html": "https://real-world-rl-with-mpc.github.io/",
    "planner-paper.html": "https://arxiv.org/html/2609.14868v1",
    "planner-project.html": "https://primitive-informed-mpc.github.io/",
    "project-repositories.json": "https://api.github.com/users/real-world-rl-with-mpc/repos?per_page=100",
    "planner-repositories.json": "https://api.github.com/users/primitive-informed-mpc/repos?per_page=100",
    "project-tree.json": "https://api.github.com/repos/real-world-rl-with-mpc/real-world-rl-with-mpc.github.io/git/trees/main?recursive=1",
    "planner-tree.json": "https://api.github.com/repos/primitive-informed-mpc/primitive-informed-mpc.github.io/git/trees/main?recursive=1",
}


def fetch(item):
    name, url = item
    request = urllib.request.Request(url, headers={"User-Agent": "mpc-dex-deployment-preflight/0.1"})
    with urllib.request.urlopen(request, timeout=45) as response:
        content = response.read()
    return name, content, {
        "url": url,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def main():
    root = Path(__file__).resolve().parents[1] / "artifacts" / "sources"
    root.mkdir(parents=True, exist_ok=True)
    manifest = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name, content, metadata in pool.map(fetch, SOURCES.items()):
            (root / name).write_bytes(content)
            manifest[name] = metadata
            print(f"Saved {name}: {metadata['bytes']} bytes")
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
