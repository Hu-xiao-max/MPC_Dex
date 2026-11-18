"""Tube of MPC-corrected trajectories around a spine demonstration (DART-style corrections at perturbed states).

Usage: collect_tube.py <demo> <j_from> <j_to> <radii comma> <directions> [--max-corrections 25]
For every control index j in [j_from, j_to], radius r and direction d: fresh cold reset (physical seed 42), bit-exact replay
of the demonstration's first j controls, one perturbed control a_j + r * u (u a unit vector drawn from a Generator seeded
by (j, r, d); clipped to [-1, 1]), then the archived corrected native MPC (ProjectedReferenceCEM, diagnostic prior, eight
workers, CEM seed derived from (j, r, d)) until the environment boundary or the correction budget. A strict force-qualified
success is re-executed from a fresh reset with a separately recomputed force audit and stored only on a bit-exact match,
in the demonstration layout plus a per-control origin array ('demo_prefix' | 'perturbed' | 'mpc'). The perturbed control is
never a learning target; the MPC controls after it are the labels at the perturbed states. Nothing here is primitive-eligible.
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
A = Path('/home/autel001/code/mpc_dex/artifacts'); R = A / 'mpc_prefix6_handoff_recovery_0001'; OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(A / 'mpc_dagger_labels_0001'))
import numpy as np
import label as L  # the DAgger labelling machinery: Recorder, independent_replay, protocol, environment config
from mpc_dex.mpc import CEMConfig, NoFeasiblePlan
from mpc_dex.primitives import PrimitiveBasis
from mpc_dex.rollouts import RolloutCostConfig
from mpc_dex.native_rollouts import NativeMujocoRolloutEvaluator
from mpc_dex.contracts import sha256
from scripts.probe_mpc import action_search_bounds
from scripts.audit_policy import PHYSICS_COLUMNS, write_json
from recovery_recipe import ProjectedReferenceCEM, integration_state
sys.path.insert(0, str(OUT)); from tracking_cem import TrackingCEM, TrackingPenaltyEvaluator, StateTrackingPenaltyEvaluator
MODE = dict(name='replan', floor_scale=None, horizon=None, avg_seeds=1, avg_steps=0, track_weight=0., state_weight=0.)


def cem_config(horizon=None):
    base = dict(L.protocol['cem'])
    if horizon: base.update(horizon_seconds=float(horizon), knots=int(round(float(horizon) / base['control_seconds'])) + 1)
    return CEMConfig(**base)


def load_source(collection_episodes, name):
    """A verified cold-reset trajectory from any collection in the demonstration layout (bit-exact replay audited at storage)."""
    d = Path(collection_episodes) / name
    with np.load(d / 'transitions.npz', allow_pickle=False) as z: src = {k: z[k].copy() for k in z.files}
    with np.load(d / 'physics.npz', allow_pickle=False) as z: phys = z['samples'].copy()
    rep = json.loads((d / 'report.json').read_text()); assert rep['strict_force_success'] and rep['stored'] and rep['independent_audit']['bitexact_replay']
    assert sha256(d / 'transitions.npz') == rep['transitions_sha256']
    return src, phys, rep


def tube_branch(src, phys, j, radius, direction, max_corrections, mode=None, seed_salt=0):
    mode = mode or MODE
    rng = np.random.default_rng(np.random.SeedSequence([80000, j, int(round(radius * 1e4)), direction, seed_salt]) if seed_salt >= 1000000 else 80000 + 1000 * j + int(round(radius * 1e4)) * 10 + direction + 7919 * seed_salt)  # the injective form is used by future collections (seed_salt >= 1e6); existing data keep their recorded seeds
    u = rng.standard_normal(16); u = u / np.linalg.norm(u) if radius > 0 else np.zeros(16)
    perturbed = np.clip(src['actions'][j].astype(np.float64) + radius * u, -1, 1).astype(np.float32)
    rec = L.Recorder(); started = time.monotonic()
    try:
        low, high, search = action_search_bounds(rec.env, L.protocol['action_bound'])
        rec.replay(src, phys, j)
        np.testing.assert_array_equal(rec.obs, src['obs'][j])
        ended = rec.step(perturbed, 'perturbed')
        status = 'perturbed_control_ended_episode' if ended else 'budget'; planning = []; label_spread = []; S_avg, K_avg = int(mode.get('avg_seeds', 1)), int(mode.get('avg_steps', 0))
        if not ended:
            with np.load(R / 'inputs/previous_recovery/prior_transitions.npz') as a:
                frames = np.vstack((rec.initial['qpos'][rec.q_ids].copy(), a['qpos'][:, rec.q_ids]))
            prior = PrimitiveBasis.from_trajectories(frames, provenance=dict(source=str(R / 'inputs/previous_recovery/prior_transitions.npz'), sha256=sha256(R / 'inputs/previous_recovery/prior_transitions.npz'), purpose='measured 40 mm diagnostic PCA prior; not a SAC primitive', successful_policy=False))
            cem_seed = 42 + 100000 * (direction + 1) + 1000 * j + int(round(radius * 1e4)) + 104729 * seed_salt
            S_avg, K_avg = int(mode.get('avg_seeds', 1)), int(mode.get('avg_steps', 0)); label_spread = []
            if mode['name'] == 'tracking':
                # warm start = the spine's own controls from j+1 on (actuator targets), local search around them. With avg_seeds > 1 the
                # first avg_steps corrections execute the mean over independent CEM seeds (the centre of the feasible-and-cheap set,
                # instead of one random point of it); every planner is advanced each step so their reference cursors stay aligned.
                reference = np.stack([rec.env.to_joint_targets(a.astype(np.float32)) for a in src['actions'][j + 1:]]) if j + 1 < len(src['actions']) else np.stack([rec.env.to_joint_targets(src['actions'][-1])])
                planners = []
                for s_ in range(S_avg):
                    pl = TrackingCEM(prior, low, high, cem_config(mode.get('horizon')), cem_seed + 1000003 * s_, floor_scale=mode['floor_scale']); pl.execution_start = rec.env.data.ctrl.copy(); pl.reset(rec.env.data.ctrl.copy()); pl.set_reference(reference, 0); planners.append(pl)
            else:
                planners = [ProjectedReferenceCEM(prior, low, high, CEMConfig(**L.protocol['cem']), cem_seed)]; planners[0].execution_start = rec.env.data.ctrl.copy(); planners[0].reset(rec.env.data.ctrl.copy())
            evaluator = NativeMujocoRolloutEvaluator(rec.env, timestep_seconds=L.protocol['cem']['timestep_seconds'], config=RolloutCostConfig(**L.protocol['cost']), require_native=True, cache_directory=R / 'native_cache')
            tw = float(mode.get('track_weight') or 0.); sw = float(mode.get('state_weight') or 0.)
            if sw > 0:
                # state reference: the spine's recorded qpos after each control from j+1 on (the penalised dims are the hand joints and the object's 7 pose coordinates)
                ref_q = src['qpos'][j + 1:] if j + 1 < len(src['qpos']) else src['qpos'][-1:]
                qa = rec.env.object_qadr; dims = np.concatenate([rec.q_ids, np.arange(qa, qa + 7)])
            try:
                for step_i in range(max_corrections):
                    before = integration_state(rec.env); t0 = time.monotonic(); active = planners if step_i < K_avg else planners[:1]; targets = []
                    for pl in active:
                        pl.execution_start = rec.env.data.ctrl.copy(); ev_ = TrackingPenaltyEvaluator(evaluator, pl, tw) if tw > 0 and mode['name'] == 'tracking' else evaluator
                        if sw > 0 and mode['name'] == 'tracking': ev_ = StateTrackingPenaltyEvaluator(ev_, pl, sw, ref_q, dims)
                        try: target, diagnostics = pl.plan(ev_); targets.append(np.asarray(target, dtype=np.float64))
                        except NoFeasiblePlan: pass
                    assert np.array_equal(before, integration_state(rec.env)); planning.append(time.monotonic() - t0)
                    if not targets: status = 'no_feasible_plan'; break
                    acts = np.stack([rec.env.to_action(tg) for tg in targets]); mean_t = np.mean(targets, axis=0)
                    action = rec.env.to_action(mean_t) if len(targets) > 1 else rec.env.to_action(targets[0]); assert float(abs(action).max()) <= search['effective_bound']
                    label_spread.append(float(np.median(np.linalg.norm(acts - acts.mean(0), axis=1))) if len(targets) > 1 else 0.)
                    if rec.step(action, 'mpc'): status = 'boundary'; break
            finally: evaluator.close()
        strict, physics = rec.strict(); rows = {k: np.asarray(v) for k, v in rec.rows.items()}
        report = dict(status=status, strict_force_success=strict, controls=len(rows['actions']), demo_prefix_controls=j, perturbed_control_index=j, perturbation_radius=radius, perturbation_direction=direction,
                      perturbation_l2=float(np.linalg.norm(perturbed.astype(np.float64) - src['actions'][j])), policy_controls=1, mpc_controls=int(sum(o == 'mpc' for o in rec.origins)),
                      failure_reason=rec.info.get('failure_reason'), final_force_qualified_angle=rec.observer.force_qualified_angle, planning_seconds_total=float(sum(planning)), wall_seconds=time.monotonic() - started, hand_joint_qpos_indices=rec.q_ids.tolist(),
                      cem_seed=(42 + 100000 * (direction + 1) + 1000 * j + int(round(radius * 1e4)) + 104729 * seed_salt) if not ended else None, planner_mode=mode['name'], planner_floor_scale=mode['floor_scale'], planner_horizon_seconds=mode.get('horizon') or L.protocol['cem']['horizon_seconds'], seed_salt=seed_salt,
                      label_average_seeds=S_avg, label_average_steps=K_avg, label_seed_spread=label_spread, tracking_penalty_weight=float(mode.get('track_weight') or 0.), state_penalty_weight=float(mode.get('state_weight') or 0.))
        return report, rows, rec.initial, physics, list(rec.origins)
    finally: rec.close()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('demo'); ap.add_argument('j_from', type=int); ap.add_argument('j_to', type=int); ap.add_argument('radii'); ap.add_argument('directions', type=int); ap.add_argument('--max-corrections', type=int, default=25)
    ap.add_argument('--tracking', type=float, default=None, help='reference-tracking planner with this floor scale (episodes go to <out>/tracking)'); ap.add_argument('--out', default=None)
    ap.add_argument('--state-weight', type=float, default=0., help='tracking planner: penalty weight on the squared deviation of the rolled-out hand-joint and object-pose state from the spine recorded states'); ap.add_argument('--track-weight', type=float, default=0., help='tracking planner: quadratic penalty weight on deviation from the reference (per-step mean squared deviation in actuator target space)'); ap.add_argument('--avg-seeds', type=int, default=1, help='tracking planner: average the first control over this many CEM seeds'); ap.add_argument('--avg-steps', type=int, default=0, help='... for the first this many corrections of each branch'); ap.add_argument('--horizon', type=float, default=None, help='tracking planner horizon in seconds (default: the archived 0.5 s; knots at the control period)'); ap.add_argument('--seed-salt', type=int, default=0, help='>= 1000000 selects the injective SeedSequence directions (fresh, non-repeating across (j, radius))'); ap.add_argument('--second-order', default=None, help='episodes dir of first-order branches: perturb their corrected controls instead of the spine (j_from/j_to select parents by their perturbed index)')
    ap.add_argument('--offsets', default='1,3', help='second order: perturbed index = parent perturbed index + offset'); ap.add_argument('--shard', default='0/1', help='second order: take parents with index %% n == k (k/n)')
    a = ap.parse_args(); radii = [float(x) for x in a.radii.split(',')]
    global OUT, MODE
    if a.tracking is not None: MODE = dict(name='tracking', floor_scale=a.tracking, horizon=a.horizon, avg_seeds=a.avg_seeds, avg_steps=a.avg_steps, track_weight=a.track_weight, state_weight=a.state_weight)
    assert (a.track_weight == 0 and a.state_weight == 0) or a.tracking is not None, 'the tracking penalty applies to the tracking planner'
    assert a.avg_seeds == 1 or a.tracking is not None, 'seed averaging applies to the tracking planner'
    assert a.horizon is None or a.tracking is not None, '--horizon applies to the tracking planner'
    OUT = Path(a.out).resolve() if a.out else (OUT / 'tracking' if a.tracking is not None else OUT); OUT.mkdir(parents=True, exist_ok=True)
    if a.second_order:
        return second_order(a, radii)
    src, phys = L.load_demo(a.demo); n = len(src['actions']); assert 0 <= a.j_from <= a.j_to < n
    (OUT / 'episodes').mkdir(exist_ok=True)
    with (OUT / f'attempts_{a.demo}_{a.j_from}_{a.j_to}.jsonl').open('a') as log:
        for j in range(a.j_from, a.j_to + 1):
            for radius in radii:
                for d in range(a.directions):
                    report, rows, initial, physics, origins = tube_branch(src, phys, j, radius, d, a.max_corrections, seed_salt=a.seed_salt)
                    ep = f"{'track' if MODE['name'] == 'tracking' else 'tube'}_{a.demo}_j{j}_r{int(round(radius * 1e4)):04d}_d{d}"; report.update(name=ep, demo=a.demo, spine=a.demo)
                    if report['strict_force_success']:
                        audit = L.independent_replay(rows['actions'], rows, initial, physics)
                        report['independent_audit'] = dict(bitexact_replay=audit['bitexact_replay'], strict_force_success=audit['strict_force_success'], final_force_qualified_angle=audit['final_force_qualified_angle'])
                        if audit['bitexact_replay'] and audit['strict_force_success']:
                            dd = OUT / 'episodes' / ep; dd.mkdir(exist_ok=True)
                            np.savez_compressed(dd / 'transitions.npz', **rows, **{'initial_' + k: v for k, v in initial.items()}, joint_positions=np.vstack((initial['qpos'][np.asarray(report['hand_joint_qpos_indices'])][None, :], rows['qpos'][:, np.asarray(report['hand_joint_qpos_indices'])])), control_origin=np.asarray(origins))
                            np.savez_compressed(dd / 'physics.npz', samples=physics, columns=np.asarray(PHYSICS_COLUMNS))
                            write_json(dd / 'report.json', dict(report, transitions_sha256=sha256(dd / 'transitions.npz'), physics_sha256=sha256(dd / 'physics.npz'), stored=True)); report['stored'] = True
                    log.write(json.dumps(report) + '\n'); log.flush()
                    print(json.dumps({k: report[k] for k in ('name', 'status', 'controls', 'mpc_controls', 'strict_force_success')} | {'stored': report.get('stored', False), 'wall': round(report['wall_seconds'], 1)}), flush=True)


def second_order(a, radii):
    """Perturb a corrected control of a first-order branch (the parent trajectory is the reference of the tracking planner)."""
    assert a.tracking is not None, 'second-order branches need the tracking planner (the parent is the reference)'
    assert all(int(x) >= 1 for x in a.offsets.split(',')), 'offsets must be >= 1'
    E = Path(a.second_order).resolve(); k_, n_ = (int(x) for x in a.shard.split('/')); offsets = [int(x) for x in a.offsets.split(',')]
    parents = []
    for d in sorted(E.iterdir()):
        if not (d / 'report.json').exists(): continue
        rep = json.loads((d / 'report.json').read_text())
        if rep.get('mpc_controls', 0) and a.j_from <= rep['perturbed_control_index'] <= a.j_to: parents.append((d.name, rep))
    parents = [p_ for i, p_ in enumerate(parents) if i % n_ == k_]
    (OUT / 'episodes').mkdir(exist_ok=True)
    with (OUT / f'attempts2_{a.demo}_{a.j_from}_{a.j_to}_shard{k_}of{n_}.jsonl').open('a') as log:
        for pname, prep in parents:
            src, phys, _ = load_source(E, pname); n = len(src['actions']); j0 = prep['perturbed_control_index']
            for off in offsets:
                j = j0 + off
                if j >= n - 1: continue  # perturbing the last control carries no correction rows
                for radius in radii:
                    for d in range(a.directions):
                        ep = f"track2_{pname.replace('track_seed_193_', '')}_i{j}_r{int(round(radius * 1e4)):04d}_d{d}"
                        if (OUT / 'episodes' / ep / 'report.json').exists(): continue  # resume-safe: a branch stored by an earlier run or another shard is never recomputed
                        report, rows, initial, physics, origins = tube_branch(src, phys, j, radius, d, a.max_corrections, seed_salt=1 + int(hashlib.sha256(pname.encode()).hexdigest()[:6], 16) % 1000)
                        report.update(name=ep, demo=a.demo, spine=a.demo, order=2, parent=pname, parent_perturbed_control_index=j0, offset=off)
                        if report['strict_force_success']:
                            audit = L.independent_replay(rows['actions'], rows, initial, physics)
                            report['independent_audit'] = dict(bitexact_replay=audit['bitexact_replay'], strict_force_success=audit['strict_force_success'], final_force_qualified_angle=audit['final_force_qualified_angle'])
                            if audit['bitexact_replay'] and audit['strict_force_success']:
                                dd = OUT / 'episodes' / ep; dd.mkdir(exist_ok=True)
                                np.savez_compressed(dd / 'transitions.npz', **rows, **{'initial_' + k: v for k, v in initial.items()}, joint_positions=np.vstack((initial['qpos'][np.asarray(report['hand_joint_qpos_indices'])][None, :], rows['qpos'][:, np.asarray(report['hand_joint_qpos_indices'])])), control_origin=np.asarray(origins))
                                np.savez_compressed(dd / 'physics.npz', samples=physics, columns=np.asarray(PHYSICS_COLUMNS))
                                write_json(dd / 'report.json', dict(report, transitions_sha256=sha256(dd / 'transitions.npz'), physics_sha256=sha256(dd / 'physics.npz'), stored=True)); report['stored'] = True
                        log.write(json.dumps(report) + '\n'); log.flush()
                        print(json.dumps({k: report[k] for k in ('name', 'status', 'controls', 'mpc_controls', 'strict_force_success')} | {'stored': report.get('stored', False), 'wall': round(report['wall_seconds'], 1)}), flush=True)


if __name__ == '__main__':
    main()
