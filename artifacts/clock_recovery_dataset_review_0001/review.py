"""Read-only artifact checks and frozen-network inference; no physics or fitting."""
from pathlib import Path
import hashlib
import json
import platform
import sys

import numpy as np

D = Path(__file__).resolve().parent
ROOT = D.parent.parent
OLD = ROOT / 'artifacts/bc_common_noise_clock_0001'
HEAD = ROOT / 'artifacts/bc_common_noise_clock_head_0001'
REC = ROOT / 'artifacts/mpc_clock_actor_recovery_0001'
NEW = ROOT / 'artifacts/bc_clock_actor_corrective_0001'
NEW_HEAD = ROOT / 'artifacts/bc_clock_actor_corrective_head_0001'
sys.path.insert(0, str(HEAD / 'source'))
import torch
from mpc_dex.learning import SACAgent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


INPUTS = {}
def record(path):
    path = Path(path)
    INPUTS[str(path.relative_to(ROOT))] = sha(path)
    return path


def read_json(path):
    return json.loads(record(path).read_text())


def read_npz(path):
    with np.load(record(path), allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def manifest_check(base):
    manifest = read_json(base / 'manifest.json')
    failed = [name for name, digest in manifest.items() if sha(base / name) != digest]
    assert not failed, (base, failed)
    return dict(files=len(manifest), sha256=sha(base / 'manifest.json'), mismatches=failed)


def exact_rows(x):
    return [[i+1, j+1] for i in range(len(x)) for j in range(i) if np.array_equal(x[i], x[j])]


def feature_design_stats(features):
    design = np.column_stack((features.astype(np.float64), np.ones(len(features))))
    s = np.linalg.svd(design, compute_uv=False)
    tolerances = {name: float(np.finfo(dtype).eps * max(design.shape) * s[0])
                  for name, dtype in [('float64', np.float64), ('float32', np.float32)]}
    return dict(shape=list(design.shape), condition=float(s[0] / s[-1]),
                singular_values=s.tolist(), tolerances=tolerances,
                rank_at_tolerance={name: int((s > tol).sum()) for name, tol in tolerances.items()},
                exact_feature_duplicate_pairs=exact_rows(features))


def changed_paths(a, b, path='checkpoint'):
    if type(a) is not type(b):
        return [path + '/type']
    if isinstance(a, torch.Tensor):
        return [] if a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b) else [path]
    if isinstance(a, dict):
        assert set(a) == set(b)
        return [p for key in a for p in changed_paths(a[key], b[key], path + '/' + str(key))]
    if isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        return [p for i, (x, y) in enumerate(zip(a, b)) for p in changed_paths(x, y, path + '/' + str(i))]
    return [] if a == b else [path]


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = dict(kind='independent-read-only-clock-recovery-dataset-review', status='checking',
                  new_physics_steps=0, optimizer_updates=0, head_solves=0,
                  manifests={name: manifest_check(base) for name, base in
                             [('old73', OLD), ('head73', HEAD), ('recovery', REC)]})
    old, new = read_npz(OLD / 'supervised_pairs.npz'), read_npz(NEW / 'supervised_pairs.npz')
    recovery = read_npz(REC / 'recovery/transitions.npz')
    source = read_npz(REC / 'inputs/clock_actor/independent_audit/transitions.npz')
    independent = read_npz(REC / 'recovery/independent_audit/transitions.npz')
    physics = read_npz(REC / 'recovery/physics.npz')
    source_physics = read_npz(REC / 'inputs/clock_actor/independent_audit/physics.npz')
    independent_physics = read_npz(REC / 'recovery/independent_audit/physics.npz')
    corrections = read_npz(REC / 'actual_mpc_correction_pairs.npz')
    origins = read_npz(REC / 'recovery/control_origins.npz')
    rows = read_json(NEW / 'row_provenance.json')['rows']
    read_json(NEW / 'protocol.json')
    read_json(NEW / 'dataset_validation.json')
    config = read_json(REC / 'config.json')['environment']
    report = read_json(REC / 'recovery/report.json')
    audit = read_json(REC / 'recovery/independent_audit/report.json')
    for key in ('obs', 'actions'):
        assert np.array_equal(new[key][:73], old[key])
        assert np.array_equal(new[key][73:], recovery[key][1:14])
    assert new['obs'].shape == (86, 61)
    assert len(rows) == 86
    assert np.array_equal(new['obs'][:, -1], new['elapsed_seconds'].astype(np.float32))
    assert np.array_equal(recovery['obs'][:, :60], recovery['environment_obs'])
    assert np.array_equal(recovery['next_obs'][:, :60], recovery['environment_next_obs'])
    assert np.array_equal(recovery['obs'][:, -1], recovery['decision_times'].astype(np.float32))
    assert np.array_equal(recovery['next_obs'][:, -1],
                          (recovery['control_times'][1:] - recovery['initial_time']).astype(np.float32))
    assert np.array_equal(recovery['next_obs'][:-1], recovery['obs'][1:])
    assert not recovery['terminated'][:-1].any() and recovery['terminated'][-1]
    assert not recovery['truncated'].any()
    assert np.array_equal(recovery['obs'][0], new['obs'][0])
    assert not np.array_equal(recovery['actions'][0], new['actions'][0])
    prefix_checks = {}
    for key, x in source.items():
        if key not in recovery:
            continue
        n = 10 if key in ('joint_positions', 'control_times') else 9
        expected = x if key.startswith('initial_') else x[:n]
        actual = recovery[key] if key.startswith('initial_') else recovery[key][:n]
        prefix_checks[key] = bool(np.array_equal(expected, actual))
    assert all(prefix_checks.values())
    assert all(np.array_equal(x, independent[k]) for k, x in recovery.items())
    assert np.array_equal(physics['samples'][:450], source_physics['samples'][:450])
    assert all(np.array_equal(x, independent_physics[k]) for k, x in physics.items())
    assert np.array_equal(origins['expert_correction_mask'], np.arange(14) >= 9)
    for key in set(corrections) & set(recovery):
        assert np.array_equal(corrections[key], recovery[key][9:14]), key
    assert np.array_equal(corrections['source_control_1based'], np.arange(10, 15))

    # Independently trace every row to its actual source observation/action and
    # measured physics boundary, including all inherited 73 labels.
    source_cache = {}
    clock_max_error = 0.
    for i, row in enumerate(rows):
        base = OLD if i < 73 else NEW
        path = base / row['source_path']
        assert sha(path) == row['source_sha256']
        if path not in source_cache:
            source_cache[path] = (read_npz(path), read_npz(path.parent / 'physics.npz'))
        data, phys = source_cache[path]
        r = row['source_row_one_based'] - 1
        assert np.array_equal(data['obs'][r][:60], new['obs'][i, :60])
        assert np.array_equal(data['actions'][r], new['actions'][i])
        measured = 0. if r == 0 else float(phys['samples'][r * 50 - 1, 0])
        assert np.float32(measured) == new['obs'][i, -1]
        clock_max_error = max(clock_max_error, abs(measured - new['elapsed_seconds'][i]))
    duplicates = exact_rows(new['obs'])
    raw_duplicates = exact_rows(new['obs'][:, :60])
    assert not duplicates
    result['dataset'] = dict(pairs=86, original73_bitexact=True, new_actual_policy_prefix8_bitexact=True,
        new_actual_mpc5_bitexact=True, all86_rows_source_observation_action_and_clock_verified=True,
        observation_exact_duplicates=duplicates, raw_observation_exact_duplicates=raw_duplicates,
        clocks_use_measured_reset_relative_time=True, clock_max_float64_difference=clock_max_error,
        new_decision_times=recovery['decision_times'][1:].tolist(), excluded_reset_same_input=True,
        excluded_reset_action_max_difference=float(abs(recovery['actions'][0]-new['actions'][0]).max()),
        action_max_abs=float(abs(new['actions']).max()),
        clamp_epsilon_01_changed_scalars=int((new['actions'] != np.clip(new['actions'], -.99, .99)).sum()),
        clamp_epsilon_02_changed_scalars=int((new['actions'] != np.clip(new['actions'], -.98, .98)).sum()),
        source_prefix_fields_verified=prefix_checks, source_original_full_integration_available=False,
        source_prefix_physics_rows_bitexact=450, independent_replay_physics_rows_bitexact=700,
        independent_replay_all_new_transition_fields_bitexact=True,
        new_transitions_form_one_actual_14_control_episode=True)

    columns = {name: i for i, name in enumerate(physics['columns'])}
    p = physics['samples']
    take = lambda name: p[:, columns[name]]
    loaded = take('force_bearing_fingers')
    retained = (loaded >= 2) & (take('finger_contacts') >= 2) & (
        take('axis_tilt_radians') <= np.deg2rad(config['success_axis_tolerance_degrees'])) & (
        take('palm_support_violation') == 0)
    force_angle = np.cumsum(np.where((take('twist_increment') < 0) | retained, take('twist_increment'), 0.))
    assert np.allclose(force_angle, take('force_qualified_angle'), rtol=0, atol=1e-12)
    assert audit['finger_supported_rotation_success'] and audit['strict_force_success']
    assert force_angle[-1] >= 2*np.pi and retained[-1]
    assert np.all(take('palm_normal_force') <= config['palm_support_force_tolerance'])
    branch = read_npz(REC / 'recovery/actual_branch_state.npz')
    assert np.array_equal(branch['integration_state'], recovery['integration_state'][8])
    assert np.array_equal(branch['qpos'], recovery['qpos'][8])
    assert np.array_equal(branch['qvel'], recovery['qvel'][8])
    assert np.array_equal(branch['ctrl'], recovery['targets'][8])
    assert np.array_equal(branch['clock_observation'], recovery['obs'][9])
    boundary_rows = []
    for i in range(1, 14):
        before, after = p[i*50-1], p[(i+1)*50-1]
        boundary_rows.append(dict(source_control_one_based=i+1,
            decision_loaded_fingers=int(before[columns['force_bearing_fingers']]),
            decision_tilt=float(before[columns['axis_tilt_radians']]),
            post_loaded_fingers=int(after[columns['force_bearing_fingers']]),
            step_loaded_min=int(p[i*50:(i+1)*50, columns['force_bearing_fingers']].min()),
            step_force_qualified_fraction=float(retained[i*50:(i+1)*50].mean())))
    commands = report['commands'][9:]
    assert len(commands) == 5
    assert all(c['prediction']['joint_target_execution'] == 'float32_normalized_action' for c in commands)
    assert all(c['planning_left_live_integration_bitexact'] for c in commands)
    result['physics'] = dict(source_prefix_qualified_angle=float(take('force_qualified_angle')[449]),
        final_raw_angle=float(take('raw_angle')[-1]), final_force_qualified_angle=float(force_angle[-1]),
        max_force_qualified_angle=float(force_angle.max()), independent_success=True,
        final_loaded_fingers=int(loaded[-1]), final_tilt=float(take('axis_tilt_radians')[-1]),
        unqualified_physics_times=take('time')[~retained].tolist(),
        zero_contact_steps=int((take('finger_contacts') == 0).sum()),
        palm_normal_force_max=float(take('palm_normal_force').max()),
        external_force_max=float(max(take(n).max() for n in ['palm_force_norm','table_force_norm','other_force_norm'])),
        branch_actual_state_matches_prefix=True, per_control_boundary_qualification=boundary_rows,
        corrected_action_mapping_in_all5_selected_prediction_diagnostics=True,
        planner_live_state_unchanged_assertions_all5=True,
        projected_reference_batches=sum(r['projected'] for r in report['reference_records']),
        reference_projection_max_radians=max(r['projection_max_radians'] for r in report['reference_records']),
        post_success_continuation_tested=False, policy_only_success=False)

    obs, actions = new['obs'], new['actions']
    distances = np.linalg.norm(obs.astype(np.float64)[:,None] - obs.astype(np.float64)[None,:], axis=2)
    np.fill_diagonal(distances, np.inf)
    pairs = []
    for i in range(73, 86):
        j = int(np.argmin(distances[i, :73]))
        pairs.append(dict(new_row_one_based=i+1, nearest_old_row_one_based=j+1,
            observation_l2=float(distances[i,j]), clock_difference=float(obs[i,-1]-obs[j,-1]),
            action_max_difference=float(abs(actions[i]-actions[j]).max())))
    result['nearest_existing_labels'] = pairs

    # No parameter changes, solve, or backward pass. Singleton CUDA reproduces
    # the established diagnostic's actual inference arithmetic.
    record(HEAD / 'candidate.pt')
    read_json(HEAD / 'candidate.pt.metadata.json')
    for file in sorted((HEAD / 'source').rglob('*')):
        if file.is_file() and '__pycache__' not in file.parts:
            record(file)
    agent = SACAgent.load(HEAD / 'candidate.pt', 'cuda')
    before = {key: x.detach().cpu().clone() for key, x in agent.actor.state_dict().items()}
    features, means, predictions = [], [], []
    with torch.no_grad():
        for row in obs:
            t = torch.as_tensor(row[None], device='cuda')
            f = agent.actor.trunk(t)
            mean, _ = agent.actor(t)
            features.append(f.cpu().numpy()[0]); means.append(mean.cpu().numpy()[0])
            predictions.append(mean.tanh().cpu().numpy()[0])
        batch_mean, _ = agent.actor(torch.as_tensor(obs, device='cuda'))
        batch_pred = batch_mean.tanh().cpu().numpy()
    features, means, predictions = map(np.asarray, (features, means, predictions))
    assert np.array_equal(predictions[73:81], actions[73:81])
    assert all(torch.equal(before[k], x.cpu()) for k,x in agent.actor.state_dict().items())
    stored = read_npz(HEAD / 'solve_inputs_and_result.npz')
    assert np.array_equal(features[:73], stored['features_float32'])
    assert np.array_equal(predictions[:73], stored['candidate_predictions'])
    old_stats = feature_design_stats(features[:73])
    new_stats = feature_design_stats(features)
    result['frozen73_trunk_conditioning'] = dict(old73=old_stats, proposed86=new_stats,
        interpretation='Read-only diagnostic on the existing 73-row-trained trunk; not a prediction of the freshly trained 86-row trunk.')
    result['frozen73_head'] = dict(old73_action_max_error=float(abs(predictions[:73]-actions[:73]).max()),
        added_actor8_predictions_bitexact=True,
        new_mpc5_action_max_error_by_row=abs(predictions[81:]-actions[81:]).max(axis=1).tolist(),
        batch_vs_singleton_action_max_difference=float(abs(batch_pred-predictions).max()),
        batch_vs_singleton_changed_scalars=int((batch_pred != predictions).sum()),
        all_parameters_unchanged=True, bc_updates=agent.bc_updates,
        pretrain_updates=agent.pretrain_updates, online_updates=agent.online_updates)
    np.savez_compressed(D / 'frozen_feature_analysis.npz', obs=obs, actions=actions,
        features=features, singleton_latent_means=means, singleton_predictions=predictions,
        batch_predictions=batch_pred, pairwise_observation_distances=distances)
    result['runtime'] = dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
        cuda=torch.version.cuda, device=torch.cuda.get_device_name(), threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(), matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32)

    # Independently check the now-completed 86-pair fit/head artifacts. No
    # additional residual solve: only verify the published correction and its
    # actual float32 result, and evaluate three inference input shapes.
    result['manifests']['new86_bc'] = manifest_check(NEW)
    summary86 = read_json(NEW / 'summary.json')
    assert summary86['bc_updates'] == 12000 and summary86['online_updates'] == summary86['critic_updates'] == 0
    assert sha(NEW_HEAD/'baseline.pt') == sha(NEW/'bc/checkpoint_12000.pt')
    assert sha(NEW_HEAD/'supervised_pairs.npz') == sha(NEW/'supervised_pairs.npz')
    assert sha(NEW_HEAD/'source/mpc_dex/learning.py') == sha(HEAD/'source/mpc_dex/learning.py')
    a = torch.load(record(NEW_HEAD/'baseline.pt'), map_location='cpu', weights_only=True)
    b = torch.load(record(NEW_HEAD/'candidate.pt'), map_location='cpu', weights_only=True)
    changed = changed_paths(a, b)
    assert set(changed) == {'checkpoint/actor/mean_head.weight', 'checkpoint/actor/mean_head.bias'}
    solved = read_npz(NEW_HEAD/'solve_inputs_and_result.npz')
    assert np.array_equal(solved['observations'], obs) and np.array_equal(solved['actions'], actions)
    assert np.array_equal(solved['design_float64'], np.column_stack((
        solved['features_float32'].astype(np.float64), np.ones(86))))
    ideal_residual = solved['design_float64'] @ solved['correction_float64'] - solved['residual_float64']
    assert np.array_equal(ideal_residual, solved['ideal_residual_float64'])
    assert np.array_equal(solved['candidate_weight'], solved['baseline_weight'] +
                          solved['correction_float64'][:-1].T.astype(np.float32))
    assert np.array_equal(solved['candidate_bias'], solved['baseline_bias'] +
                          solved['correction_float64'][-1].astype(np.float32))
    new_agent = SACAgent.load(NEW_HEAD/'candidate.pt', 'cuda')
    assert new_agent.bc_updates == 12000 and new_agent.pretrain_updates == new_agent.online_updates == 0
    with torch.no_grad():
        scalar_pred = np.asarray([new_agent.act(row, deterministic=True) for row in obs])
        singleton_batch_pred = np.asarray([new_agent.act(row[None], deterministic=True)[0] for row in obs])
        full_batch_pred = new_agent.act(obs, deterministic=True)
        new_features = np.asarray([new_agent.actor.trunk(torch.as_tensor(row, device='cuda')).cpu().numpy() for row in obs])
        new_targets = torch.atanh(torch.as_tensor(actions, device='cuda')).cpu().numpy()
    assert np.array_equal(scalar_pred, solved['candidate_predictions'])
    assert np.array_equal(new_features, solved['features_float32'])
    assert np.array_equal(new_targets, solved['latent_targets_float32'])
    assert np.array_equal(solved['residual_float64'],
                          solved['latent_targets_float32'].astype(np.float64)-solved['baseline_means'].astype(np.float64))
    shape_predictions = {'rank1_61': scalar_pred, 'batch1_1x61': singleton_batch_pred, 'batch86_86x61': full_batch_pred}
    def errors(x, y):
        d = x.astype(np.float64) - y.astype(np.float64)
        return dict(max_abs=float(abs(d).max()), mse=float((d*d).mean()), changed_scalars=int((x != y).sum()))
    result['new86_head'] = dict(checkpoint_sha256=sha(NEW_HEAD/'candidate.pt'),
        baseline_equals_final86_bc=True, dataset_matches86=True, changed_checkpoint_paths=changed,
        all_other_checkpoint_values_bitexact=True, bc_updates=new_agent.bc_updates,
        pretrain_updates=new_agent.pretrain_updates, online_updates=new_agent.online_updates,
        additional_review_solves=0, correction_application_float32_bitexact=True,
        ideal_residual_max_abs=float(abs(ideal_residual).max()),
        correction_frobenius=float(np.linalg.norm(solved['correction_float64'])),
        baseline_head_weight_norm=float(np.linalg.norm(solved['baseline_weight'].astype(np.float64))),
        candidate_head_weight_norm=float(np.linalg.norm(solved['candidate_weight'].astype(np.float64))),
        design=feature_design_stats(new_features),
        singleton_features_targets_predictions_match_saved_solve_bitexact=True,
        deterministic_shape_errors_vs_labels={name: errors(value, actions) for name,value in shape_predictions.items()},
        deterministic_shape_pairwise_errors={x+'__'+y: errors(shape_predictions[x], shape_predictions[y])
            for x,y in [('rank1_61','batch1_1x61'),('rank1_61','batch86_86x61'),('batch1_1x61','batch86_86x61')]},
        reset_action_errors_vs_label={name: errors(value[:1], actions[:1]) for name,value in shape_predictions.items()},
        new_stochastic_inference_or_physics=False, runtime=result['runtime'])
    np.savez_compressed(D/'new86_head_shape_comparison.npz', observations=obs, labels=actions,
        rank1_predictions=scalar_pred, batch1_predictions=singleton_batch_pred,
        batch86_predictions=full_batch_pred, features=new_features,
        stored_correction_ideal_residual=ideal_residual)
    result['new86_bc_completed_outcomes'] = [{k:r[k] for k in ['bc_updates','steps','failure_reason',
        'peak_force_qualified_angle','peak_palm_force']} for r in summary86['results']]
    result['manifests']['new86_head'] = manifest_check(NEW_HEAD)
    head_summary = read_json(NEW_HEAD/'summary.json')
    head_trace = read_npz(NEW_HEAD/'independent_audit/transitions.npz')
    head_physics = read_npz(NEW_HEAD/'independent_audit/physics.npz')
    with torch.no_grad():
        recomputed_actions = np.asarray([new_agent.act(row, deterministic=True) for row in head_trace['obs']])
    assert np.array_equal(recomputed_actions, head_trace['actions'])
    assert np.array_equal(head_trace['obs'][:,:60], head_trace['environment_obs'])
    assert np.array_equal(head_trace['obs'][:,-1], head_trace['decision_times'].astype(np.float32))
    assert np.array_equal(head_trace['next_obs'][:-1], head_trace['obs'][1:])
    head_columns = {n:i for i,n in enumerate(head_physics['columns'])}
    h = head_physics['samples']
    eligible = (h[:,head_columns['force_bearing_fingers']] >= 2) & (
        h[:,head_columns['finger_contacts']] >= 2) & (
        h[:,head_columns['axis_tilt_radians']] <= np.deg2rad(config['success_axis_tolerance_degrees'])) & (
        h[:,head_columns['palm_support_violation']] == 0)
    twist = h[:,head_columns['twist_increment']]
    angles = np.cumsum(np.where((twist < 0) | eligible, twist, 0.))
    assert np.allclose(angles,h[:,head_columns['force_qualified_angle']], rtol=0, atol=1e-12)
    assert not head_summary['strict_force_success'] and head_summary['failure_reason'] == 'drop'
    result['new86_head_physical_trace'] = dict(steps=len(head_trace['actions']),
        actor_commands_recomputed_bitexact=True, observation_raw_and_clock_consistent=True,
        force_angle_recomputed_from_saved_every2ms_twist_and_loaded_qualifications=True,
        peak_force_qualified_angle=float(angles.max()), final_force_qualified_angle=float(angles[-1]),
        failure_reason=head_summary['failure_reason'], strict_force_success=False,
        palm_support=head_summary['palm'], post_success_continuations=0,
        qualification_comparison_only_no_physics_reexecution=True)
    result['status'] = 'complete'
    for name,digest in INPUTS.items():
        assert sha(ROOT/name) == digest, name
    (D/'inputs_manifest.json').write_text(json.dumps(INPUTS, indent=2)+'\n')
    (D/'checks.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(status='complete', dataset=result['dataset'], physics=result['physics'],
        head=result['frozen73_head'], condition73=old_stats['condition'], condition86=new_stats['condition'],
        ranks86=new_stats['rank_at_tolerance']), indent=2))


if __name__ == '__main__':
    main()
