"""One rank-gated frozen-feature correction after the authorized feedback round."""
from pathlib import Path
from dataclasses import replace
import json
import shutil
import sys

D = Path(__file__).resolve().parent
F = D.parent
sys.path.insert(0, str(F))
sys.path.insert(0, str(F / 'source'))
import numpy as np
import torch
from mpc_dex.learning import SACAgent
from mpc_dex.env import EnvConfig
from mpc_dex.contracts import sha256, write_checkpoint_metadata
from scripts.audit_policy import write_json
from audit import run_audit
from train_feedback import invariant_hashes, module_hash


@torch.no_grad()
def singleton_values(agent, observations):
    features, means, predictions = [], [], []
    for observation in observations:
        x = torch.as_tensor(observation, device=agent.device, dtype=torch.float32)
        feature = agent.actor.trunk(x)
        mean = agent.actor.mean_head(feature)
        features.append(feature.cpu().numpy())
        means.append(mean.cpu().numpy())
        predictions.append(agent.act(observation, deterministic=True))
    return tuple(np.asarray(x) for x in (features, means, predictions))


def metrics(means, predictions, targets, actions):
    delta = predictions.astype(np.float64) - actions.astype(np.float64)
    latent_delta = means.astype(np.float64) - targets.astype(np.float64)
    return dict(latent_mse=float(np.mean(latent_delta**2)),
        latent_max_abs_error=float(abs(latent_delta).max()),
        action_mse=float(np.mean(delta**2)), action_max_abs_error=float(abs(delta).max()),
        first_action_max_abs_error=float(abs(delta[0]).max()),
        action_max_abs_error_by_row=abs(delta).max(1).tolist())


def grouped_metrics(means, predictions, targets, actions):
    return {name: metrics(means[s], predictions[s], targets[s], actions[s])
        for name, s in [('all25', slice(None)), ('original22', slice(0, 22)), ('correction3', slice(22, 25))]}


def changed_payload_paths(a, b, path='checkpoint'):
    if type(a) is not type(b):
        return [path + '[type]']
    if isinstance(a, torch.Tensor):
        return [] if a.dtype == b.dtype and a.shape == b.shape and a.numpy().tobytes() == b.numpy().tobytes() else [path]
    if isinstance(a, dict):
        assert set(a) == set(b)
        return [p for key in a for p in changed_payload_paths(a[key], b[key], path + '/' + str(key))]
    if isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        return [p for i, (x, y) in enumerate(zip(a, b)) for p in changed_payload_paths(x, y, path + '/' + str(i))]
    return [] if a == b else [path]


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    input_hashes = json.loads((F / 'inputs_manifest.json').read_text())
    assert all(sha256(F / path) == expected for path, expected in input_hashes.items())
    source = F / 'bc_feedback/checkpoint_12000.pt'
    baseline_source_hash = sha256(source)
    shutil.copyfile(source, D / 'baseline.pt')
    shutil.copyfile(str(source) + '.metadata.json', D / 'baseline.pt.metadata.json')
    agent = SACAgent.load(D / 'baseline.pt', 'cuda')
    ec = EnvConfig(**json.loads((F / 'strict_config.json').read_text())['environment'])
    with np.load(F / 'supervised_pairs.npz') as a:
        obs, actions = a['obs'].copy(), a['actions'].copy()
    assert obs.shape == (25, 60) and actions.shape == (25, 16)
    assert agent.bc_updates == 24000 and agent.pretrain_updates == agent.online_updates == 0
    frozen = dict(**invariant_hashes(agent), trunk=module_hash(agent.actor.trunk),
                  log_std_head=module_hash(agent.actor.log_std_head))
    with torch.no_grad():
        targets = torch.atanh(torch.as_tensor(actions, device=agent.device).clamp(-.98, .98)).cpu().numpy()
        assert np.array_equal(torch.as_tensor(actions).clamp(-.98, .98).numpy(), actions)
        weight_before = agent.actor.mean_head.weight.cpu().numpy().copy()
        bias_before = agent.actor.mean_head.bias.cpu().numpy().copy()
    features, means_before, predictions_before = singleton_values(agent, obs)
    design = np.column_stack((features.astype(np.float64), np.ones(len(features))))
    residual = targets.astype(np.float64) - means_before.astype(np.float64)
    singular_values = np.linalg.svd(design, compute_uv=False)
    threshold = np.finfo(np.float64).eps * max(design.shape) * singular_values[0]
    rank = int(np.count_nonzero(singular_values > threshold))
    condition = float(singular_values[0] / singular_values[-1]) if singular_values[-1] > 0 else float('inf')
    permitted = rank == len(obs) and np.isfinite(condition)
    report = dict(purpose='One numerical head refinement after one feedback imitation round; not paper BC or SAC',
        production_eligible=False, input_sha256=sha256(F / 'supervised_pairs.npz'),
        initial_actor_sha256=sha256(F / 'initial_polished.pt'),
        baseline_sha256=baseline_source_hash, baseline_copy_sha256=sha256(D / 'baseline.pt'),
        source_script_sha256=sha256(__file__), source_manifest_sha256=sha256(F / 'inputs_manifest.json'),
        rank_gate=dict(checked_before_solve=True, required_row_rank=len(obs), rank=rank,
            numerical_rank_threshold=threshold, condition_number=condition,
            singular_values=singular_values.tolist(), permitted=bool(permitted),
            criterion='Full float64 numerical row rank and finite condition; no sweep or regularization'),
        baseline_fit=grouped_metrics(means_before, predictions_before, targets, actions),
        critic_updates=0, online_updates=0, optimizer_updates=0, numpy_version=np.__version__,
        torch_version=str(torch.__version__), frozen_before=frozen, status='rank_checked')
    write_json(D / 'report.json', report)
    print(json.dumps(dict(rank_gate=report['rank_gate'])), flush=True)
    if not permitted:
        np.savez_compressed(D / 'rank_inputs.npz', observations=obs, actions=actions,
            features_float32=features, design_float64=design, targets_float32=targets,
            residual_float64=residual, singular_values=singular_values)
        report.update(status='stopped_rank_gate', refinement_solves=0, physical_evaluations=0)
        write_json(D / 'report.json', report)
        return
    correction, _, solved_rank, solved_singular_values = np.linalg.lstsq(design, residual, rcond=None)
    assert solved_rank == len(obs) and np.isfinite(correction).all()
    projected_residual = design @ correction - residual
    with torch.no_grad():
        agent.actor.mean_head.weight.add_(torch.as_tensor(correction[:-1].T.copy(), device=agent.device, dtype=torch.float32))
        agent.actor.mean_head.bias.add_(torch.as_tensor(correction[-1].copy(), device=agent.device, dtype=torch.float32))
        weight_after = agent.actor.mean_head.weight.cpu().numpy().copy()
        bias_after = agent.actor.mean_head.bias.cpu().numpy().copy()
    features_after, means_after, predictions_after = singleton_values(agent, obs)
    frozen_after = dict(**invariant_hashes(agent), trunk=module_hash(agent.actor.trunk),
                       log_std_head=module_hash(agent.actor.log_std_head))
    assert frozen == frozen_after and np.array_equal(features, features_after)
    checkpoint = D / 'polished.pt'
    agent.save(checkpoint)
    metadata = write_checkpoint_metadata(checkpoint, ec, agent.config,
        stage='diagnostic-feedback-fixed-trunk-head-refinement',
        offline_replay_sha256=report['input_sha256'])
    metadata.update(production_eligible=False, actor_kind='One feedback BC round plus one numerical head correction; no SAC',
        baseline_checkpoint_sha256=baseline_source_hash, refinement_solves=1, extra_optimizer_updates=0,
        frozen_numerical_source=str(F / 'source'), refinement_script_sha256=sha256(__file__))
    write_json(str(checkpoint) + '.metadata.json', metadata)
    changed = changed_payload_paths(torch.load(D / 'baseline.pt', map_location='cpu', weights_only=True),
                                    torch.load(checkpoint, map_location='cpu', weights_only=True))
    assert set(changed) == {'checkpoint/actor/mean_head.weight', 'checkpoint/actor/mean_head.bias'}, changed
    np.savez_compressed(D / 'solve_inputs_and_result.npz', observations=obs, actions=actions,
        target_latents_float32=targets, singleton_features_float32=features, design_float64=design,
        residual_float64=residual, correction_float64=correction, projected_residual_float64=projected_residual,
        singular_values=singular_values, baseline_mean_float32=means_before, polished_mean_float32=means_after,
        baseline_predictions=predictions_before, polished_predictions=predictions_after,
        baseline_weight=weight_before, baseline_bias=bias_before, polished_weight=weight_after, polished_bias=bias_after)
    report.update(polished_sha256=sha256(checkpoint), polished_fit=grouped_metrics(means_after, predictions_after, targets, actions),
        changed_checkpoint_paths=changed, all_other_payload_values_bitwise_unchanged=True,
        frozen_after=frozen_after, frozen_components_bitwise_unchanged=True,
        solve=dict(method='One numpy.linalg.lstsq minimum-norm float64 residual correction', rcond=None,
            design_shape=list(design.shape), rank=int(solved_rank),
            correction_frobenius_norm=float(np.linalg.norm(correction)),
            correction_max_abs=float(abs(correction).max()),
            ideal_residual_mse=float(np.mean(projected_residual**2)),
            actual_weight_max_change=float(abs(weight_after - weight_before).max()),
            actual_bias_max_change=float(abs(bias_after - bias_before).max()),
            feature_route='Singleton float32 Torch inference matching SACAgent.act',
            target_route='Float32 Torch atanh of actual actions clamped at .02; clamp changes no actions',
            application='Cast correction to float32 and add once to original float32 head'),
        status='auditing')
    write_json(D / 'report.json', report)
    evaluation, executed, _ = run_audit(agent, ec, D / 'independent_audit', 100)
    evaluation['policy_kind'] = 'diagnostic-feedback-plus-one-head-correction'
    write_json(D / 'independent_audit/report.json', evaluation)
    report['evaluation'] = evaluation
    if evaluation['finger_supported_rotation_success']:
        total = len(executed) + 20
        continuous_ec = replace(ec, terminate_on_success=False, horizon_steps=max(100, total))
        aftermath, _, _ = run_audit(agent, continuous_ec, D / 'continuous_policy_aftermath', total)
        with np.load(D / 'independent_audit/physics.npz') as a, np.load(D / 'continuous_policy_aftermath/physics.npz') as b:
            assert np.array_equal(a['samples'], b['samples'][:len(a['samples'])])
        aftermath.update(policy_kind='diagnostic-feedback-plus-one-head-correction',
            requested_post_success_seconds=2., continuation='Actual deterministic actor each step; no held target',
            completed_post_success_seconds=max(0., aftermath['simulated_seconds'] - .1 * len(executed)),
            completed_two_seconds_without_failure=bool(aftermath['steps'] == total and
                not aftermath['independent_drop'] and not aftermath['independent_palm_support_violation']))
        report['continuous_policy_aftermath'] = aftermath
        write_json(D / 'continuous_policy_aftermath/report.json', aftermath)
    assert sha256(source) == baseline_source_hash
    assert all(sha256(F / path) == expected for path, expected in input_hashes.items())
    report.update(status='complete', refinement_solves=1, physical_evaluations=1,
                  baseline_source_unchanged=True, all_input_hashes_unchanged=True)
    write_json(D / 'report.json', report)
    print(json.dumps(dict(polished_fit=report['polished_fit'], evaluation=evaluation)), flush=True)


if __name__ == '__main__':
    main()
