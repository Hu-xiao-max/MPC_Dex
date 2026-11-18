"""Compare exact Python/native scoring; no live model or asset writes.

Run: uv run --frozen python -m scripts.benchmark_native_rollouts --output PATH
"""
import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from mpc_dex.env import AllegroRotationEnv, EnvConfig, ROTATION_VALIDATION
from mpc_dex.native_rollouts import NativeMujocoRolloutEvaluator
from mpc_dex.rollouts import MujocoRolloutEvaluator, RolloutCostConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, nargs='+', default=[1, 8])
    parser.add_argument('--repetitions', type=int, default=3)
    arguments = parser.parse_args()
    env = AllegroRotationEnv(EnvConfig(initial_grasp=str(
        Path(__file__).resolve().parents[1] / 'assets/initial_grasp.npz')))
    env.reset(seed=0)
    result = {'mujoco_version': mujoco.__version__, 'physics_timestep': env.config.timestep,
              'planner_timestep': .01, 'scenarios': [],
              'constraint_protocol': ROTATION_VALIDATION,
              'palm_support_force_tolerance': env.config.palm_support_force_tolerance}
    try:
        for workers in arguments.workers:
            config = RolloutCostConfig(workers=workers, rotation_progress='retained')
            python = MujocoRolloutEvaluator(env, .01, config)
            native = NativeMujocoRolloutEvaluator(env, .01, config, require_native=True)
            try:
                result['native_provenance'] = native.native_provenance
                for candidates, horizon in [(100, 10), (350, 50)]:
                    # A stable held target guarantees the full advertised work,
                    # rather than inflating speed with early rejections.
                    sequences = np.broadcast_to(env.data.ctrl, (candidates, horizon, 16)).copy()
                    entry = {'workers': workers, 'candidates': candidates,
                             'horizon_seconds': horizon*.01, 'iterations': arguments.repetitions}
                    native(sequences[:1, :1])  # Exclude build/thread startup from timing.
                    python(sequences[:1, :1])
                    for name, evaluator in [('python', python), ('native', native)]:
                        times = []
                        for _ in range(arguments.repetitions):
                            start = time.perf_counter()
                            costs = evaluator(sequences)
                            times.append(time.perf_counter()-start)
                        if name == 'python':
                            reference = costs.copy()
                        else:
                            np.testing.assert_array_equal(costs, reference)
                        entry[name+'_seconds'] = times
                        entry[name+'_median_seconds'] = float(np.median(times))
                        entry[name+'_evaluated_physics_steps'] = evaluator.last_diagnostics['evaluated_physics_steps']
                        entry[name+'_invalid'] = evaluator.last_diagnostics['invalid']
                    entry['speedup'] = entry['python_median_seconds']/entry['native_median_seconds']
                    entry['estimated_three_iteration_native_seconds'] = 3*entry['native_median_seconds']
                    result['scenarios'].append(entry)
                    print(json.dumps(entry), flush=True)
            finally:
                python.close()
                native.close()
    finally:
        env.close()
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
