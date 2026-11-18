"""Table of large-sample stochastic success rates per checkpoint (Wilson 95% intervals), written to summary.md."""
import json, math, sys
from pathlib import Path
R = Path(__file__).resolve().parent / 'results'
def wilson(k, n, z=1.96):
    if n == 0: return (0., 0.)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)
def step(label):
    return {'pretrained': 0}.get(label, int(label.split('_')[1]) if '_' in label else -1)
lines = ['# Frozen checkpoint evaluation summary', '', 'Stochastic successes out of 30 per prefix (D = deterministic trial passed); cold = prefix 0.', '',
         '| variant | checkpoint | p5 | p6 | p7 | p8 | cold | p6 Wilson 95% |', '| --- | --- | --- | --- | --- | --- | --- | --- |']
best = []
for vdir in sorted(R.iterdir()):
    rows = []
    for f in vdir.glob('*.json'):
        j = json.loads(f.read_text()); rows.append((step(j['checkpoint']), j))
    for _, j in sorted(rows):
        cells = []
        low = ' '.join(f"p{q}:{j['prefixes'][q]['stochastic_successes']}{'D' if j['prefixes'][q]['deterministic_success'] else ''}" for q in ('1', '2', '3', '4') if q in j['prefixes'])
        for p in ('5', '6', '7', '8', '0'):
            x = j['prefixes'][p]; cells.append(f"{x['stochastic_successes']}{'D' if x['deterministic_success'] else ''}")
        if low: cells[0] = low + ' | ' + cells[0]
        k, n = j['prefixes']['6']['stochastic_successes'], j['prefixes']['6']['stochastic_trials']
        lo, hi = wilson(k, n)
        lines.append(f"| {j['variant'].replace('bootstrap_variant_', '')} | {j['checkpoint']} | " + ' | '.join(cells) + f" | {k}/{n} [{lo:.2f}, {hi:.2f}] |")
        score = sum(j['prefixes'][p]['stochastic_successes'] for p in ('6', '7', '8')) + 3 * j['prefixes']['5']['stochastic_successes']
        best.append((j['prefixes']['6']['stochastic_successes'], j['prefixes']['7']['stochastic_successes'] + j['prefixes']['8']['stochastic_successes'], j['prefixes']['5']['stochastic_successes'], j['variant'], j['checkpoint']))
best.sort(reverse=True)
lines += ['', '## Best checkpoints by prefix-6 stochastic successes (ties by prefix 7+8, then prefix 5)', '']
for b in best[:12]:
    lines.append(f"- {b[3].replace('bootstrap_variant_', '')} / {b[4]}: p6 {b[0]}/30, p7+p8 {b[1]}/60, p5 {b[2]}/30")
(Path(__file__).resolve().parent / 'summary.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
