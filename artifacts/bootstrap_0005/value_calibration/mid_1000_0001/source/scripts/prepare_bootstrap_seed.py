"""Re-execute a partial contact trajectory as explicitly labeled bootstrap data.

This seed is never a full-turn demonstration or a primitive extraction input.
The actual environment supplies observations, rewards and endpoint flags;
saved controls are converted only if the increment controller can execute them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from mpc_dex.bootstrap_env import create_bootstrap_environment, policy_interface
from mpc_dex.contracts import environment_contract, sha256
from mpc_dex.replay import ReplayBuffer
from scripts.train_bootstrap import read_environment


def prepare(source, config, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    cfg, ec = read_environment(config)
    settings = cfg.get('bootstrap', {})
    interface = policy_interface(settings)
    if interface['kind'] != 'bootstrap_joint_increment':
        raise ValueError('Seed conversion requires the incremental bootstrap environment')
    with np.load(source, allow_pickle=False) as archive:
        controls = archive['actions'].copy()
        expected = archive['next_observations'].copy()
    if controls.ndim != 2 or controls.shape[1] != 16 or expected.shape != (len(controls), 60) or not len(controls):
        raise ValueError('Invalid source contact trajectory')
    env = create_bootstrap_environment(ec, settings)
    replay = ReplayBuffer(capacity=len(controls))
    max_difference = 0.
    try:
        obs, _ = env.reset(seed=0)
        for i, absolute in enumerate(controls):
            targets = env.to_joint_targets(absolute)
            action = (targets - env.data.ctrl) / env.bootstrap.max_delta_radians
            if np.max(np.abs(action)) > 1 + 1e-8:
                raise ValueError(f'Source command {i} exceeds the increment limit')
            action = np.clip(action, -1, 1)
            if not np.allclose(targets, env.joint_targets_for_increment(action), rtol=0, atol=1e-10):
                raise ValueError(f'Source command {i} exceeds the target-error limit')
            next_obs, reward, terminated, truncated, info = env.step(action)
            difference = float(np.max(np.abs(next_obs - expected[i])))
            max_difference = max(max_difference, difference)
            if difference > 1e-5:
                raise ValueError(f'Source trajectory does not replay at command {i}: {difference}')
            if terminated or truncated or info['drop']:
                raise ValueError('Expected a retained partial trajectory without an environment boundary')
            replay.add_transition(obs, action, reward, next_obs, False, i == len(controls) - 1)
            obs = next_obs
        output.parent.mkdir(parents=True, exist_ok=True)
        replay.save(output)
        metadata = dict(kind='partial-bootstrap-seed', successful_full_turns=0,
                        source=str(source), source_sha256=sha256(source), replay_sha256=sha256(output),
                        environment_contract=environment_contract(ec), policy_interface=interface,
                        transitions=len(replay), final_angle=info['angle'],
                        retained_angle=info['retained_angle'], max_observation_replay_error=max_difference,
                        final_boundary='time cutoff; nonterminal; final next action unknown',
                        reward='bootstrap retained-rotation shaping; recomputed in contact simulation')
        output.with_suffix(output.suffix + '.metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        return metadata
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.config, args.output), indent=2))
