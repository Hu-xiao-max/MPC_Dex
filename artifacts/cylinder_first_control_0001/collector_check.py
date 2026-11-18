"""Cross-check: run the cylinder collector's own attempt function (collect_used.run_attempt, unchanged) for the given seeds; prints status/success/controls. Nothing is stored."""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'cylinder_demonstrations_0001')); sys.path.insert(0, str(HERE.parent / 'control_period_0001')); sys.path.insert(0, str(HERE.parent / 'mpc_prefix6_handoff_recovery_0001')); sys.path.insert(0, '/home/autel001/code/mpc_dex')
import probe; probe.ENV_OVERRIDES['object_shape'] = 'circular'
import collect_used
for seed in [int(x) for x in sys.argv[1:]]:
    rep, rows, initial, physics = collect_used.run_attempt(0.1, seed, 40)
    print(json.dumps({k: rep[k] for k in ('seed', 'status', 'controls', 'strict_force_success', 'failure_reason', 'final_force_qualified_angle', 'planning_seconds_total')}), flush=True)
