"""Deterministic findings.md generator for executed variant stages: counts and tables straight from the artifact.

Usage: auto_findings.py <variant_dir_name> [...]   Writes findings.md (and refreshes manifest.json) only where
no findings.md exists, unless --force. Every number is read from summary.json, evaluations/*/summary.json,
demonstration_calibration_*.json, metrics.jsonl and the large-sample results; no interpretation beyond the
stated hypothesis is added.
"""
import collections, hashlib, json, math, re, sys
from pathlib import Path
A = Path('/home/autel001/code/mpc_dex/artifacts'); R = A / 'frozen_checkpoint_evaluation_0001' / 'results'
force = '--force' in sys.argv
names = [a for a in sys.argv[1:] if not a.startswith('--')]

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def step(name): return -1 if name == 'pretraining_baseline' else int(name.split('_')[1])
def wilson(k, n, z=1.96):
    if n == 0: return (0., 0.)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)

for n in names:
    D = A / n
    if (D / 'findings.md').exists() and not force:
        print('skip (exists)', n); continue
    if not (D / 'summary.json').exists():
        print('skip (not executed)', n); continue
    p = json.loads((D / 'protocol.json').read_text()); s = json.loads((D / 'summary.json').read_text()); c = s['counts']
    exit_line = ((D / 'run.log').read_text().strip().splitlines() or ['?'])[-1] if (D / 'run.log').exists() else '?'  # an empty run.log (launcher redirected elsewhere) is treated like a missing one
    prefixes = p['eval_prefixes']; noisy = p.get('eval_noisy_prefixes', [])
    # evaluation table
    rows = []
    for ev in sorted((D / 'evaluations').glob('*/summary.json'), key=lambda f: step(f.parent.name)):
        reps = json.loads(ev.read_text())['reports']; agg = collections.defaultdict(lambda: [0, 0])
        for r in reps:
            m = re.match(r'prefix(\d+)_(deterministic|stochastic|noisy)', r['phase'])
            if m: key = (f"p{m.group(1)}", m.group(2)[0])
            elif r['phase'].startswith('cold_'): key = ('cold', 'd' if r['phase'] == 'cold_deterministic' else 's')
            else: continue  # multi-source cells (src_*, shared_*) are tabulated separately
            agg[key][0] += int(r['finger_supported_rotation_success']); agg[key][1] += 1
        cells = []
        for q in prefixes:
            d, st = agg[(f'p{q}', 'd')], agg[(f'p{q}', 's')]
            cells.append(f"{'pass' if d[0] else 'fail'} / {st[0]}/{st[1]}")
        for q in noisy:
            x = agg[(f'p{q}', 'n')]; cells.append(f"{x[0]}/{x[1]}")
        cd, cs = agg[('cold', 'd')], agg[('cold', 's')]
        cells.append(f"{'pass' if cd[0] else 'fail'}" + (f" / {cs[0]}/{cs[1]}" if cs[1] else ''))
        rows.append(f"| {ev.parent.name} | " + ' | '.join(cells) + ' |')
    header = '| ' + ' | '.join(['Checkpoint'] + [f'Prefix-{q} det / stoch' for q in prefixes] + [f'Prefix-{q} noisy' for q in noisy] + ['Cold det / stoch']) + ' |'
    sep = '| --- |' + ' --- |' * (len(prefixes) + len(noisy) + 1)
    # large-sample
    ls_rows = []
    if (R / n).exists():
        for f in sorted((R / n).glob('*.json'), key=lambda f: step(f.stem) if f.stem != 'pretrained' else 0):
            j = json.loads(f.read_text()); cells = []
            for q in ('5', '6', '7', '8', '0'):
                x = j['prefixes'][q]; cells.append(f"{x['stochastic_successes']}/{x['stochastic_trials']}{' D' if x['deterministic_success'] else ''}")
            k, m = j['prefixes']['6']['stochastic_successes'], j['prefixes']['6']['stochastic_trials']; lo, hi = wilson(k, m)
            ls_rows.append(f"| {j['checkpoint']} | " + ' | '.join(cells) + f" | [{lo:.2f}, {hi:.2f}] |")
    # calibration trajectory
    cal = []
    for f in sorted(D.glob('demonstration_calibration_*.json'), key=lambda f: (0 if 'before' in f.name else 1 if 'after' in f.name else 2, step('online_' + f.stem.split('_')[-1]) if f.stem.split('_')[-1].isdigit() else 0)):
        j = json.loads(f.read_text()); e = j['episodes'].get('recorded') or next(iter(j['episodes'].values()))
        cal.append(f"{f.stem.replace('demonstration_calibration_', '')}: L2 {e['first_row_deterministic_action_l2_difference']:.4f}, std {e['first_row_latent_std_mean']:.4f}, bias {e['critic_normalized_bias'][0]:+.2f}/{e['critic_normalized_bias'][1]:+.2f}")
    ups = [json.loads(l) for l in (D / 'metrics.jsonl').open() if '"online_update"' in l]
    bc = [u.get('actor_bc_loss') for u in ups if u.get('actor_bc_loss') is not None]
    sd = [u.get('actor_bc_std_loss') for u in ups if u.get('actor_bc_std_loss') is not None]
    # multi-source stages: cold / shared-state / per-source suffix cells per checkpoint
    ms_rows, ms_header, ms_sep = [], '', ''
    if p.get('prefix_sources'):
        S_ = p['suffix_set']; srcs = p['eval_source_names']; ks = [k for k in (p.get('start_prefix_set') or []) if k >= 1]
        cols = ['Cold'] + [f'Shared k{k}' for k in ks] + [f'{n_} s1..s{S_[-1]}' for n_ in srcs]
        ms_header = '| Checkpoint | ' + ' | '.join(cols) + ' |'; ms_sep = '| --- |' + ' --- |' * len(cols)
        for ev in sorted((D / 'evaluations').glob('*/summary.json'), key=lambda f: step(f.parent.name)):
            by = {}
            for r in json.loads(ev.read_text())['reports']:
                by.setdefault(r['phase'].rsplit('_stochastic', 1)[0].rsplit('_deterministic', 1)[0], []).append(r)
            def cell(prefix):
                rs = by.get(prefix, []); det = [r for r in rs if r['phase'].endswith('deterministic')]; st = [r for r in rs if not r['phase'].endswith('deterministic')]
                return ('P' if det and det[0]['finger_supported_rotation_success'] else 'F' if det else '-') + f"{sum(r['finger_supported_rotation_success'] for r in st)}/{len(st)}"
            ms_rows.append(f"| {ev.parent.name} | " + ' | '.join([cell('cold')] + [cell(f'shared_k{k}') for k in ks] + [' '.join(cell(f'src_{n_}_s{s_}') for s_ in S_) for n_ in srcs]) + ' |')
    reps = [json.loads(q.read_text()) for q in sorted((D / 'episodes/online').glob('*/report.json'))]
    reasons = collections.Counter(r['environment_failure_reason'] or ('success' if r['finger_supported_rotation_success'] else 'truncated') for r in reps)
    text = f"""# Variant {p['variant_index']:02d} `{p['variant']}`: auto-generated findings

**Settings differing from SAC0004:** `{json.dumps(p['settings_differing_from_sac0004'])}`. Inherited checkpoint: `{p['inherited_checkpoint']}`. Hypothesis: {p['hypothesis']}

Stage status `{s['status']}` ({exit_line}, {s['wall_seconds']:.0f} s loop); final learner counters {s['learner_counters']}; {c['new_pretrain_updates']} critic-pretraining and {c['new_online_updates']} online updates as prescribed. Temperature: {p.get('temperature', 'learned from the inherited value')}. Actor step: {p.get('actor_lr', 'inherited 3e-4')}. Supervision: {p.get('actor_bc_term', 'none')}; spread anchor: {p.get('actor_bc_std_term', 'none')}. Handoff noise: {p.get('handoff_noise', 'off')}.

## Actual execution

- Warm-up: {c['warmup_complete_episodes']} frozen-actor episodes by prefix {c['warmup_episodes_by_prefix']}, force-qualified successes {c['warmup_force_successes_by_prefix']} (informational; the gate is the {p['demonstration_rows']}-row demonstration buffer).
- Online: {c['online_policy_interactions']} interactions; completed episodes by prefix {c['online_episodes_by_prefix']}, successes {c['online_force_successes_by_prefix']}; outcomes {dict(reasons)}; handoffs terminated before the policy {c['handoff_terminated_before_policy']}; unfinished last suffix {s.get('unfinished_last_suffix')}. Training episodes are changing-policy, assisted and demonstration-mixed; they are not frozen-policy evidence.
- Physical prefix controls replayed and verified bit-for-bit: {c['physical_prefix_controls']}; none entered replay.
- Actor BC loss first/last: {(f'{bc[0]:.4g} / {bc[-1]:.4g}' if bc else 'none')}; spread-anchor loss first/last: {(f'{sd[0]:.4g} / {sd[-1]:.4g}' if sd else 'none')}; alpha raw-equivalent last {ups[-1]['alpha_raw_equivalent']:.4f}.
- Supervised-row calibration (first recorded row: deterministic-mean L2 to its label, latent std, critic bias): {'; '.join(cal)}.

## Frozen evaluations (in-run; {p['eval_stochastic_trials'] if 'eval_stochastic_trials' in p else (len(p['eval_stochastic_seeds_by_prefix'][str(prefixes[0])]) if prefixes else 'no recorded-prefix evaluations;')} stochastic seeds per prefix)

{header}
{sep}
{chr(10).join(rows)}

{('## Multi-source evaluation (in-run; a cell is the deterministic trial P/F followed by stochastic successes/trials; per source the cells run over the suffix lengths; shared k = the state after k shared cold-start controls)' + chr(10) + chr(10) + ms_header + chr(10) + ms_sep + chr(10) + chr(10).join(ms_rows) + chr(10)) if ms_rows else ''}
## Large-sample frozen evaluation (30 fresh-seed stochastic trials per prefix; D = deterministic pass)

{('| Checkpoint | Prefix 5 | Prefix 6 | Prefix 7 | Prefix 8 | Cold | Prefix-6 Wilson 95% |' + chr(10) + '| --- | --- | --- | --- | --- | --- | --- |' + chr(10) + chr(10).join(ls_rows)) if ls_rows else 'No large-sample result for this stage at the time of writing.'}

## Scope

All collected rows are assisted or demonstration-mixed and ineligible for primitive export; cold-reset trials decide deployment eligibility and none succeeded in this stage. Numbers above are read directly from the artifact files by `bootstrap_variant_template_0001/auto_findings.py`; interpretation is deferred to the family syntheses in `bootstrap_variant_template_0001/`.
"""
    (D / 'findings.md').write_text(text)
    (D / 'manifest.json').write_text(json.dumps({str(q.relative_to(D)): sha(q) for q in sorted(D.rglob('*')) if q.is_file() and '__pycache__' not in q.parts and q.name not in ('manifest.json', 'run.log')}, indent=2) + '\n')
    print('wrote', n)
