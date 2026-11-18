"""Read saved trajectories only; no fitting, planning, or physical replay."""
from pathlib import Path
import hashlib
import json
import sys

sys.dont_write_bytecode = True
import numpy as np
import torch

OUT = Path(__file__).resolve().parent
ARTIFACTS = OUT.parent
RECOVERY = ARTIFACTS / 'mpc_polished_actor_recovery_0001'
COMMON = RECOVERY / 'nominal_prefix_corrections_0001'
OLD = ARTIFACTS / 'bc_feedback_round_0001'
GROUPS = [
    ('joint_position', 0, 16, 'rad'),
    ('object_position', 16, 19, 'm'),
    ('object_quaternion', 19, 23, 'dimensionless wxyz'),
    ('principal_angles', 23, 25, 'rad'),
    ('joint_velocity', 25, 41, 'rad/s; finite differences'),
    ('object_velocity', 41, 44, 'm/s; finite differences'),
    ('previous_action', 44, 60, 'normalized absolute joint targets'),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def archive(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: value.copy() for key, value in data.items()}


def norms(delta):
    return dict(l2=float(np.linalg.norm(delta)), max_abs=float(abs(delta).max()))


def group_deltas(delta):
    return [{"group": name, "units": units, **norms(delta[start:stop])}
            for name, start, stop, units in GROUPS]


def latent(actions):
    # The BC objective uses this float32 clamp and atanh. No tanh round trip,
    # clipping outside this expression, or change to stored action labels.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.as_tensor(actions, dtype=torch.float32, device=device)
    clipped = x.clamp(-0.98, 0.98)
    assert torch.equal(x, clipped)
    return torch.atanh(clipped).cpu().numpy()


def pair(dataset, i, j):
    x, a, z, labels = dataset
    dx = x[i].astype(np.float64) - x[j]
    da = a[i].astype(np.float64) - a[j]
    dz = z[i].astype(np.float64) - z[j]
    return dict(first=labels[i], second=labels[j], observation=norms(dx),
                action=norms(da), latent_target=norms(dz),
                action_l2_per_raw_observation_l2=float(np.linalg.norm(da) / np.linalg.norm(dx)),
                latent_l2_per_raw_observation_l2=float(np.linalg.norm(dz) / np.linalg.norm(dx)),
                observation_groups=group_deltas(dx))


def all_pairs(dataset):
    return sorted([pair(dataset, i, j) for i in range(len(dataset[0]))
                   for j in range(i + 1, len(dataset[0]))],
                  key=lambda value: value['observation']['l2'])


def main():
    assert not (OUT / 'report.json').exists()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    files = {
        'nominal_transitions': COMMON / 'strict_audit/transitions.npz',
        'nominal_strict_report': COMMON / 'strict_audit/report.json',
        'nominal_hold_report': COMMON / 'post_success_hold_2s/report.json',
        'nominal_report': COMMON / 'report.json',
        'nominal_manifest': COMMON / 'manifest.json',
        'actor_recovery_transitions': RECOVERY / 'independent_audit/transitions.npz',
        'actor_recovery_strict_report': RECOVERY / 'independent_audit/report.json',
        'actor_recovery_hold_report': RECOVERY / 'post_success_hold_2s/report.json',
        'actor_recovery_origins': RECOVERY / 'executed_control_origins.npz',
        'actor_recovery_report': RECOVERY / 'report.json',
        'actor_recovery_manifest': RECOVERY / 'manifest.json',
        'old_supervised_pairs': OLD / 'supervised_pairs.npz',
        'old_conditioning_report': OLD / 'fit_conditioning.json',
        'old_manifest': OLD / 'manifest.json',
    }
    before = {key: sha(path) for key, path in files.items()}
    # Check each relevant existing member against its previously saved manifest.
    for base in (COMMON, RECOVERY, OLD):
        manifest = read_json(base / 'manifest.json')
        for path in files.values():
            relative = str(path.relative_to(base)) if path.is_relative_to(base) else None
            if relative in manifest:
                assert sha(path) == manifest[relative], path
    n = archive(files['nominal_transitions'])
    r = archive(files['actor_recovery_transitions'])
    old = archive(files['old_supervised_pairs'])
    origins = archive(files['actor_recovery_origins'])
    np.testing.assert_array_equal(origins['actual_mpc_correction_mask'], [False] * 6 + [True] * 3)
    assert n['obs'].shape == r['obs'].shape == (9, 60)
    assert old['obs'].shape == (25, 60)
    for data in (n, r):
        np.testing.assert_array_equal(data['obs'][1:], data['next_obs'][:-1])
        np.testing.assert_array_equal(data['terminated'], [False] * 8 + [True])
        assert not data['truncated'].any()
    for key in ('initial_qpos', 'initial_qvel', 'initial_ctrl', 'initial_joint_positions'):
        np.testing.assert_array_equal(n[key], r[key])
    np.testing.assert_array_equal(n['obs'][:7], old['obs'][:7])
    np.testing.assert_array_equal(n['actions'][:6], old['actions'][:6])
    np.testing.assert_array_equal(r['obs'][6:9], old['obs'][22:25])
    np.testing.assert_array_equal(r['actions'][6:9], old['actions'][22:25])
    np.testing.assert_array_equal(n['actions'][6:9], r['actions'][6:9])
    assert read_json(files['nominal_report'])['environment_contract'] == read_json(files['actor_recovery_report'])['environment_contract']
    x = np.concatenate([n['obs'], r['obs'][6:9]])
    a = np.concatenate([n['actions'], r['actions'][6:9]])
    z = latent(a)
    labels = [f'nominal_{i + 1}' for i in range(9)] + [f'actor_correction_{i + 1}' for i in range(6, 9)]
    old_labels = [f'original_teacher_{i + 1}' for i in range(22)] + [f'actor_correction_{i + 1}' for i in range(6, 9)]
    new_dataset = x, a, z, labels
    old_dataset = old['obs'], old['actions'], latent(old['actions']), old_labels
    new_pairs, old_pairs = all_pairs(new_dataset), all_pairs(old_dataset)
    assert all(p['observation']['l2'] > 0 for p in new_pairs + old_pairs)
    corresponding = [pair(new_dataset, 6 + i, 9 + i) for i in range(3)]
    assert all(p['action']['l2'] == p['latent_target']['l2'] == 0 for p in corresponding)
    summaries = {}
    for name, pairs in [('proposed_12', new_pairs), ('old_25', old_pairs)]:
        summaries[name] = dict(
            closest_pairs=pairs[:12], exact_duplicate_observation_pairs=0,
            maximum_action_pairwise_slope=max(pairs, key=lambda p: p['action_l2_per_raw_observation_l2']),
            maximum_latent_pairwise_slope=max(pairs, key=lambda p: p['latent_l2_per_raw_observation_l2']))
    physical = {}
    for name, strict_key, hold_key in [
        ('nominal', 'nominal_strict_report', 'nominal_hold_report'),
        ('actor_recovery', 'actor_recovery_strict_report', 'actor_recovery_hold_report')]:
        strict, hold = read_json(files[strict_key]), read_json(files[hold_key])
        assert strict['finger_supported_rotation_success'] and not strict['independent_palm_support_violation']
        assert strict['no_external_support'] and not strict['independent_drop']
        assert hold['finger_supported_rotation_success'] and hold['no_external_support']
        assert hold['completed_two_seconds_without_failure']
        physical[name] = dict(strict_controls=strict['steps'], strict_force_qualified_angle=strict['final_force_qualified_angle'],
            strict_success=strict['finger_supported_rotation_success'], strict_drop=strict['independent_drop'],
            strict_palm_force=strict['peak_support_normal_forces']['palm'],
            hold_force_qualified_angle=hold['final_force_qualified_angle'],
            hold_completed_seconds=hold['completed_hold_seconds'],
            hold_minimum_loaded_fingers=hold.get('minimum_hold_loaded_fingers'),
            hold_no_external_support=hold['no_external_support'])
    report = dict(
        purpose='Independent saved-data compatibility review of nominal9 plus actual actor-state MPC correction3',
        production_eligible=False, training_updates=0, planning_calls=0, physical_replay_calls=0,
        inputs={key: {'path': str(path), 'sha256': before[key]} for key, path in files.items()},
        dataset_contract=dict(rows=12, source_rows_one_based=list(range(1, 10)) + [7, 8, 9],
            groups=['nominal_successful_branch'] * 9 + ['actual_actor_state_mpc_correction'] * 3,
            original_teacher_rows_7_through_22_excluded=True, actor_prefix_actions_excluded=True,
            recurrent_or_episode_dataset=False, note='Supervised union of two branches; do not invent replay links across row 9 to row 10.',
            action_abs_max=float(abs(a).max()), bc_epsilon=0.02, clamp_changed_scalars=0,
            environment_contracts_match=True, source_trajectory_chains_and_terminal_cutoffs_verified=True),
        numeric_convention=dict(observations='Original float32 stored values; deltas and norms accumulated in float64',
            latent='torch.atanh(float32_actions.clamp(-0.98,0.98)); CUDA if available',
            device='cuda' if torch.cuda.is_available() else 'cpu', numpy=np.__version__, torch=torch.__version__,
            raw_observation_distance_caveat='Unscaled mixed units. Pairwise slopes are descriptive under this coordinate convention, not physical or causal sensitivity measures.'),
        resolved_continuation_pair_before=pair(old_dataset, 6, 22),
        corresponding_nominal_actor_pairs=corresponding,
        startup_pair=pair(new_dataset, 0, 1),
        startup_action_1_equals_initial_previous_action=bool(np.array_equal(n['actions'][0], n['obs'][0, 44:60])),
        startup_previous_actions_identical=bool(np.array_equal(n['obs'][0, 44:60], n['obs'][1, 44:60])),
        pairwise_summaries=summaries, saved_physical_audit_evidence=physical,
        findings=[
            'The demonstrated original-step7 versus actor-correction-step7 conflict is removed exactly; all three corresponding continuation pairs share bit-identical actions and pre-tanh targets.',
            'The nominal startup hold-versus-motion pair remains: the two closest observations receive very different actions. Global worst pairwise slopes are unchanged.',
            'Two successful physical branches support common continuation compatibility. No inference here establishes a trained memoryless actor will reproduce either branch or be robust nearby.',
            'Action equality across corresponding corrections does not imply state convergence; raw observation separation grows over their three controls, mostly in finite-difference joint velocities.'
        ], source_script_sha256=sha(Path(__file__)))
    np.savez_compressed(OUT / 'arrays.npz', obs=x, actions=a, latent_targets=z,
        labels=np.asarray(labels), source_rows_one_based=np.array(list(range(1, 10)) + [7, 8, 9]),
        old_obs=old['obs'], old_actions=old['actions'], old_latent_targets=old_dataset[2],
        old_labels=np.asarray(old_labels), corresponding_obs_delta=x[9:].astype(np.float64) - x[6:9],
        corresponding_action_delta=a[9:].astype(np.float64) - a[6:9],
        corresponding_latent_delta=z[9:].astype(np.float64) - z[6:9])
    assert before == {key: sha(path) for key, path in files.items()}
    (OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(corresponding=corresponding, startup=report['startup_pair'], physical=physical), indent=2))


if __name__ == '__main__':
    main()
