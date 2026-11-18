"""Bind the author's additive findings-only source-manifest finalization."""
from pathlib import Path
import hashlib
import json
import shutil

D = Path(__file__).resolve().parent
S = D.parent / 'bootstrap_suffix_sac_0001'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    expected = '258842818fa0087aa70566996fad768fa088c1e6e3102d533c1c40f3e48edfc0'
    assert sha(S / 'manifest.json') == expected
    original = json.loads((D / 'inputs' / 'source_manifest.json').read_text())
    final = json.loads((S / 'manifest.json').read_text())
    assert all(final.get(name) == digest for name, digest in original.items())
    assert set(final) - set(original) == {'findings.md'}
    assert sha(S / 'findings.md') == final['findings.md']
    shutil.copyfile(D / 'manifest.json', D / 'manifest_before_final_source_binding.json')
    shutil.copyfile(S / 'manifest.json', D / 'inputs' / 'source_manifest_final.json')
    shutil.copyfile(S / 'findings.md', D / 'inputs' / 'source_findings.md')
    record = dict(original_terminal_manifest_sha256=sha(D / 'inputs' / 'source_manifest.json'),
                  final_source_manifest_sha256=expected,
                  original_entries=len(original), final_entries=len(final),
                  every_original_entry_unchanged=True, added_files=['findings.md'],
                  plotted_source_data_unchanged=True, plots_not_rerendered=True)
    (D / 'final_source_binding.json').write_text(json.dumps(record, indent=2) + '\n')
    manifest = {str(p.relative_to(D)): sha(p) for p in sorted(D.rglob('*'))
                if p.is_file() and p != D / 'manifest.json'}
    (D / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(record, plot_manifest_sha256=sha(D / 'manifest.json')), indent=2))


if __name__ == '__main__':
    main()
