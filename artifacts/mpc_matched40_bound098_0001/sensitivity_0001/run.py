"""Bounded diagnostic of recorded-control sensitivity; no learned policy."""
from pathlib import Path
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
from mpc_dex.cli import read_config
from mpc_dex.contracts import environment_contract, sha256
from mpc_dex.env import AllegroRotationEnv
from scripts.audit_policy import PHYSICS_COLUMNS, audit_episode
from scripts.audit_recorded_controls import RecordedPolicy

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'transitions.npz'
CONFIG = ROOT / 'artifacts/mpc_fullturn_prior_40mm_0001/branch_diagnosis/strict_config.json'
if (HERE / 'report.json').exists():
    raise FileExistsError('Preserve completed sensitivity evidence')
cfg, ec = read_config(CONFIG)
with np.load(SOURCE, allow_pickle=False) as archive:
    original = archive['actions'].copy()
assert original.dtype == np.float32
snapshot = HERE / 'source'
snapshot.mkdir(exist_ok=False)
shutil.copyfile(SOURCE, snapshot / 'transitions.npz')
shutil.copyfile(CONFIG, snapshot / 'config.json')
names = ['mpc_dex/env.py', 'scripts/audit_policy.py', 'scripts/audit_recorded_controls.py', 'uv.lock']
hashes = {name: sha256(ROOT / name) for name in names}
for name in names:
    shutil.copyfile(ROOT / name, snapshot / Path(name).name)
results = []
conditions = [(0., 17011)] + [(scale, seed) for scale in (1e-6, 1e-4, 1e-3) for seed in (17011, 17012, 17013)]
for scale, seed in conditions:
    actions = (original.astype(np.float64) + np.random.default_rng(seed).uniform(-scale, scale, original.shape)).astype(np.float32)
    assert np.max(np.abs(actions)) < 1
    env = AllegroRotationEnv(ec)
    try:
        report, physics, commands = audit_episode(env, RecordedPolicy(actions), 0, len(actions))
    finally:
        env.close()
    condition = f'noise_{scale:g}_seed{seed}'
    directory = HERE / condition
    directory.mkdir()
    np.savez_compressed(directory / 'controls.npz', actions=actions)
    np.savez_compressed(directory / 'physics.npz', samples=physics, columns=np.asarray(PHYSICS_COLUMNS))
    (directory / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    report.update(uniform_noise_amplitude=scale, noise_seed=seed, action_error_max=float(np.max(np.abs(actions-original))), action_error_rms=float(np.sqrt(np.mean((actions-original)**2))))
    (directory / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    results.append(dict(condition=condition, **report))
    print(json.dumps({key: report[key] for key in ('uniform_noise_amplitude', 'noise_seed', 'steps', 'finger_supported_rotation_success', 'max_force_qualified_angle', 'environment_failure_reason')}), flush=True)
assert all(sha256(ROOT/name) == digest for name, digest in hashes.items())
report = dict(purpose='Prespecified bounded open-loop sensitivity diagnostic, not a learned policy or robustness estimate', source_sha256=sha256(SOURCE), script_sha256=sha256(__file__), numerical_source_sha256=hashes, environment_contract=environment_contract(ec), conditions=results, production_eligible=False)
(HERE / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
