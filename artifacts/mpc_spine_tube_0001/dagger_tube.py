"""DAgger around the spine with the reference-tracking planner: roll out a frozen learner deterministically from spine prefixes,
branch after b of its own controls while the object is still held, and let the tracking CEM (reference = the spine's controls
from the branch index on) complete the turn. Stored like tube branches (origin 'demo_prefix' x k, 'policy' x b, 'mpc' ...).
Usage: dagger_tube.py <agent.pt> <k_from> <k_to> <branches comma, e.g. 1,2,3,4> <out_dir> [--max-corrections 25] [--floor-scale 0.1]"""
import argparse, json, sys, time
from pathlib import Path
A = Path('/home/autel001/code/mpc_dex/artifacts'); R = A / 'mpc_prefix6_handoff_recovery_0001'; HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(A / 'mpc_dagger_labels_0001')); sys.path.insert(0, str(HERE))
import numpy as np, torch
import label as L
from collect_tube import load_source
from tracking_cem import TrackingCEM, TrackingPenaltyEvaluator
from mpc_dex.learning import SACAgent
from mpc_dex.mpc import CEMConfig, NoFeasiblePlan
from mpc_dex.primitives import PrimitiveBasis
from mpc_dex.rollouts import RolloutCostConfig
from mpc_dex.native_rollouts import NativeMujocoRolloutEvaluator
from mpc_dex.contracts import sha256
from scripts.probe_mpc import action_search_bounds
from scripts.audit_policy import PHYSICS_COLUMNS, write_json
from recovery_recipe import integration_state
HORIZON = None; AVG_SEEDS = 1; AVG_STEPS = 0; TRACK_WEIGHT = 0.
torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False


def policy_rollout(agent, src, phys, k, limit=40):
    rec = L.Recorder(); rec.replay(src, phys, k); acts = []
    while not (rec.term or rec.trunc) and len(acts) < limit:
        a = L.policy_action(agent, rec.obs); acts.append(a); rec.step(a, 'policy')
    out = dict(controls=len(acts), failure=rec.info.get('failure_reason'), success=bool(rec.strict()[0]), final_angle=rec.observer.force_qualified_angle); rec.close()
    return acts, out


def branch(agent, src, phys, k, policy_actions, b, max_corrections, floor_scale, cem_seed):
    rec = L.Recorder(); started = time.monotonic()
    try:
        low, high, search = action_search_bounds(rec.env, L.protocol['action_bound'])
        rec.replay(src, phys, k)
        for j in range(b):
            np.testing.assert_array_equal(L.policy_action(agent, rec.obs), policy_actions[j]); ended = rec.step(policy_actions[j], 'policy')
            assert not ended
        n = len(src['actions']); ref_from = min(k + b, n - 1)
        reference = np.stack([rec.env.to_joint_targets(a.astype(np.float32)) for a in src['actions'][ref_from:]])
        with np.load(R / 'inputs/previous_recovery/prior_transitions.npz') as a_:
            frames = np.vstack((rec.initial['qpos'][rec.q_ids].copy(), a_['qpos'][:, rec.q_ids]))
        prior = PrimitiveBasis.from_trajectories(frames, provenance=dict(source=str(R / 'inputs/previous_recovery/prior_transitions.npz'), sha256=sha256(R / 'inputs/previous_recovery/prior_transitions.npz'), purpose='measured 40 mm diagnostic PCA prior; not a SAC primitive', successful_policy=False))
        cfg = dict(L.protocol['cem'])
        if HORIZON: cfg.update(horizon_seconds=HORIZON, knots=int(round(HORIZON / cfg['control_seconds'])) + 1)
        planners = []
        for s_ in range(AVG_SEEDS):
            pl = TrackingCEM(prior, low, high, CEMConfig(**cfg), cem_seed + 1000003 * s_, floor_scale=floor_scale); pl.execution_start = rec.env.data.ctrl.copy(); pl.reset(rec.env.data.ctrl.copy()); pl.set_reference(reference, 0); planners.append(pl)
        evaluator = NativeMujocoRolloutEvaluator(rec.env, timestep_seconds=L.protocol['cem']['timestep_seconds'], config=RolloutCostConfig(**L.protocol['cost']), require_native=True, cache_directory=R / 'native_cache')
        status = 'budget'; planning = []; label_spread = []
        try:
            for step_i in range(max_corrections):
                before = integration_state(rec.env); t0 = time.monotonic(); active = planners if step_i < AVG_STEPS else planners[:1]; targets = []
                for pl in active:
                    pl.execution_start = rec.env.data.ctrl.copy(); ev_ = TrackingPenaltyEvaluator(evaluator, pl, TRACK_WEIGHT) if TRACK_WEIGHT > 0 else evaluator
                    try: target, diagnostics = pl.plan(ev_); targets.append(np.asarray(target, dtype=np.float64))
                    except NoFeasiblePlan: pass
                assert np.array_equal(before, integration_state(rec.env)); planning.append(time.monotonic() - t0)
                if not targets: status = 'no_feasible_plan'; break
                acts = np.stack([rec.env.to_action(tg) for tg in targets]); action = rec.env.to_action(np.mean(targets, axis=0)) if len(targets) > 1 else rec.env.to_action(targets[0]); assert float(abs(action).max()) <= search['effective_bound']
                label_spread.append(float(np.median(np.linalg.norm(acts - acts.mean(0), axis=1))) if len(targets) > 1 else 0.)
                if rec.step(action, 'mpc'): status = 'boundary'; break
        finally: evaluator.close()
        strict, physics = rec.strict(); rows = {k_: np.asarray(v) for k_, v in rec.rows.items()}
        report = dict(status=status, strict_force_success=strict, controls=len(rows['actions']), demo_prefix_controls=k, policy_controls=b, mpc_controls=int(sum(o == 'mpc' for o in rec.origins)), perturbed_control_index=k + b - 1,
                      failure_reason=rec.info.get('failure_reason'), final_force_qualified_angle=rec.observer.force_qualified_angle, planning_seconds_total=float(sum(planning)), wall_seconds=time.monotonic() - started, hand_joint_qpos_indices=rec.q_ids.tolist(),
                      cem_seed=cem_seed, planner_mode='tracking', planner_floor_scale=floor_scale, planner_horizon_seconds=HORIZON or L.protocol['cem']['horizon_seconds'], label_average_seeds=AVG_SEEDS, label_average_steps=AVG_STEPS, label_seed_spread=label_spread, tracking_penalty_weight=TRACK_WEIGHT, policy_deviation_l2=[float(np.linalg.norm(policy_actions[j] - src['actions'][k + j])) for j in range(b) if k + j < n])
        return report, rows, rec.initial, physics, list(rec.origins)
    finally: rec.close()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('agent'); ap.add_argument('k_from', type=int); ap.add_argument('k_to', type=int); ap.add_argument('branches'); ap.add_argument('out'); ap.add_argument('--max-corrections', type=int, default=25); ap.add_argument('--floor-scale', type=float, default=.1); ap.add_argument('--spine', default='seed_193'); ap.add_argument('--horizon', type=float, default=None); ap.add_argument('--avg-seeds', type=int, default=1); ap.add_argument('--avg-steps', type=int, default=0); ap.add_argument('--track-weight', type=float, default=0.)
    a = ap.parse_args(); global HORIZON, AVG_SEEDS, AVG_STEPS, TRACK_WEIGHT; HORIZON = a.horizon; AVG_SEEDS = a.avg_seeds; AVG_STEPS = a.avg_steps; TRACK_WEIGHT = a.track_weight; OUT = Path(a.out).resolve(); (OUT / 'episodes').mkdir(parents=True, exist_ok=True); bs = [int(x) for x in a.branches.split(',')]
    agent = SACAgent.load(a.agent, 'cuda'); agent_sha = sha256(a.agent)
    src, phys, _ = load_source(A / 'mpc_cold_demonstrations_0001/episodes', a.spine); n = len(src['actions'])
    with (OUT / f'attempts_k{a.k_from}_{a.k_to}.jsonl').open('a') as log:
        for k in range(a.k_from, a.k_to + 1):
            acts, outcome = policy_rollout(agent, src, phys, k); print(json.dumps(dict(k=k, policy=outcome)), flush=True)
            for b in bs:
                if outcome['success'] or b >= outcome['controls'] or k + b >= n: continue
                cem_seed = 42 + 1000 * k + 100 * b + 7
                report, rows, initial, physics, origins = branch(agent, src, phys, k, acts, b, a.max_corrections, a.floor_scale, cem_seed)
                ep = f'dagger_{a.spine}_k{k}_b{b}'; report.update(name=ep, demo=a.spine, spine=a.spine, policy_outcome=outcome, policy_agent_sha256=agent_sha, perturbation_radius=None, perturbation_direction=-1)
                if report['strict_force_success']:
                    audit = L.independent_replay(rows['actions'], rows, initial, physics)
                    report['independent_audit'] = dict(bitexact_replay=audit['bitexact_replay'], strict_force_success=audit['strict_force_success'], final_force_qualified_angle=audit['final_force_qualified_angle'])
                    if audit['bitexact_replay'] and audit['strict_force_success']:
                        d = OUT / 'episodes' / ep; d.mkdir(exist_ok=True)
                        np.savez_compressed(d / 'transitions.npz', **rows, **{'initial_' + k_: v for k_, v in initial.items()}, joint_positions=np.vstack((initial['qpos'][np.asarray(report['hand_joint_qpos_indices'])][None, :], rows['qpos'][:, np.asarray(report['hand_joint_qpos_indices'])])), control_origin=np.asarray(origins))
                        np.savez_compressed(d / 'physics.npz', samples=physics, columns=np.asarray(PHYSICS_COLUMNS))
                        write_json(d / 'report.json', dict(report, transitions_sha256=sha256(d / 'transitions.npz'), physics_sha256=sha256(d / 'physics.npz'), stored=True)); report['stored'] = True
                log.write(json.dumps(report) + '\n'); log.flush()
                print(json.dumps({k_: report[k_] for k_ in ('name', 'status', 'controls', 'policy_controls', 'mpc_controls', 'strict_force_success')} | {'stored': report.get('stored', False), 'dev': [round(x, 3) for x in report['policy_deviation_l2']], 'wall': round(report['wall_seconds'], 1)}), flush=True)


if __name__ == '__main__':
    main()
