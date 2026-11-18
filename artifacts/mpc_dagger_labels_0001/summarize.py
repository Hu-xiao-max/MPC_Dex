"""Deterministic summary of the DAgger labelling attempts -> findings.md (exact counts only)."""
import json, hashlib
from pathlib import Path
import numpy as np
D = Path(__file__).resolve().parent
rows = [json.loads(l) for f in sorted(D.glob('attempts_*.jsonl')) for l in f.read_text().splitlines() if l.strip()]
stored = [r for r in rows if r.get('stored')]
demos = sorted({r['demo'] for r in rows})
diag = json.loads((D / 'diag/policy_branch_states.json').read_text()) if (D / 'diag/policy_branch_states.json').exists() else {}
lines = ['# Findings: DAgger-style MPC labels at policy-visited states', '',
         f"Policy: `{rows[0]['policy_checkpoint']['variant']}` checkpoint `{rows[0]['policy_checkpoint']['checkpoint']}` (agent.pt sha256 `{rows[0]['policy_checkpoint']['agent_sha256'][:12]}...`), deterministic inference (`mean.tanh()`), the same contract as the stage evaluators.",
         f"Demonstrations labelled: {', '.join(demos)} (the three sources held out by stage 0079, on which its online_4000 policy scored 0/90 at every suffix length 1-8). Suffix lengths 1-6, branch points 0-3 policy controls, at most 20 MPC corrections per branch (archived recovery recipe: CEM 350/10/3, 6 knots, .5 s horizon, diagnostic PCA prior, .98 search bound fixed at the cold reset).", '',
         f"**Branches attempted: {len(rows)}. Strict force-qualified successes: {sum(r['strict_force_success'] for r in rows)}. Stored after bit-exact independent replay: {len(stored)}.**", '',
         '## Per demonstration and suffix', '', '| demo | s | policy controls before failure (failure) | b=0 | b=1 | b=2 | b=3 |', '|---|---|---|---|---|---|---|']
for demo in demos:
    for s in sorted({r['suffix'] for r in rows if r['demo'] == demo}):
        cells = {r['policy_controls']: r for r in rows if r['demo'] == demo and r['suffix'] == s}
        po = next(iter(cells.values()))['policy_outcome']
        def cell(b):
            r = cells.get(b)
            if r is None: return 'n/a (policy episode already over)' if b >= po['controls'] else '-'
            if r.get('stored'): return f"stored: {r['mpc_controls']} MPC"
            return {'no_feasible_plan': 'no feasible plan', 'budget': 'budget (20) exhausted', 'boundary': f"boundary, not strict ({r.get('failure_reason')})"}[r['status']]
        lines.append(f"| {demo} | {s} | {po['controls']} ({po['failure']}) | {cell(0)} | {cell(1)} | {cell(2)} | {cell(3)} |")
by_b = {}
for r in rows: by_b.setdefault(r['policy_controls'], []).append(r)
lines += ['', '## By branch point', '', '| policy controls kept | attempted | no feasible plan | budget exhausted | boundary not strict | stored |', '|---|---|---|---|---|---|']
for b in sorted(by_b):
    g = by_b[b]; lines.append(f"| {b} | {len(g)} | {sum(r['status'] == 'no_feasible_plan' for r in g)} | {sum(r['status'] == 'budget' for r in g)} | {sum(r['status'] == 'boundary' and not r.get('stored') for r in g)} | {sum(bool(r.get('stored')) for r in g)} |")
if diag:
    lines += ['', '## Why most policy-visited states have no feasible MPC plan', '', 'Object state after each deterministic policy control from the handoff (`diag/policy_branch_states.json`; force-bearing fingers, contactless seconds, finite-cost candidates out of 350 in one planner batch):', '']
    for k, v in diag.items():
        lines.append(f"- {k}: " + '; '.join(f"{x['step']}: fingers {x['fingers']}, contactless {x['contactless']:.3f} s" + (f", L2 to recorded control {x['l2_to_recorded']:.2f}, max |a| {x['max_abs']:.3f}" if x.get('l2_to_recorded') is not None else '') + (f", finite costs {x['finite_costs']}/350" if 'finite_costs' in x else '') + (f", episode ended: {x['failure']}" if x.get('ended') else '') for x in v))
    lines += ['', 'The policy\'s controls at these never-seen states are far from the recorded ones (L2 2.5-4.1 in action space against a success basin of .02-.05) and near-saturated; after one such control the object has often lost finger contact or is heading for the palm, so every .5 s rollout of the planner ends in a rejected state. Labels therefore exist only at the states where the object is still held.']
lines += ['', '## Status', '', 'Every stored episode is a full cold-reset trajectory (demonstration prefix, 0-3 policy controls, MPC corrections) that replays bit-for-bit; it is labelled per control (`control_origin`). The policy rows are states to be labelled, never targets; the template uses only the MPC segment after the last policy control as demonstration rows. Nothing here is primitive-eligible, and no deployment claim follows from this stage; the labels are inputs to `bootstrap_variant_0080_ms_dagger_from79`.']
(D / 'findings.md').write_text('\n'.join(lines) + '\n'); print(len(rows), len(stored))
