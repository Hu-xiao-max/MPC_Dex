"""DAgger-style MPC labels at policy-visited states of held-out demonstrations.

Usage: label.py <policy_variant> <checkpoint> <demo_name> [<demo_name> ...] [--suffixes 1,2,3,4] [--branches 0,1,2] [--max-corrections 20]
For each held-out demonstration (seed) and suffix length s: cold reset (physical seed 42), replay the first n-s
recorded controls bit-for-bit, then execute the frozen policy deterministically (same inference contract as the
variant runner: CUDA float32 singleton, mean.tanh()) and record its controls. For each branch point b (number of
policy controls kept, while the episode is still open), a fresh episode replays the demonstration prefix plus the
first b recorded policy controls and hands over to the corrected native MPC (archived recovery recipe, diagnostic
prior, eight workers, at most --max-corrections controls). A strict success is re-executed from a fresh reset with
a separately recomputed force audit and stored only on a bit-exact match, in the demonstration layout
(episodes/<name>/{transitions.npz,physics.npz,report.json}) with a per-control origin array
('demo_prefix' | 'policy' | 'mpc'). Nothing here is primitive-eligible; the labels are inputs for a later
multi-source stage.
"""
import argparse, hashlib, importlib.util, json, sys, time
from pathlib import Path
A = Path('/home/autel001/code/mpc_dex/artifacts'); R = A / 'mpc_prefix6_handoff_recovery_0001'; OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(R / 'source')); sys.path.insert(0, str(R))
import numpy as np, torch
from mpc_dex.env import AllegroRotationEnv, EnvConfig
from mpc_dex.learning import SACAgent
from mpc_dex.mpc import CEMConfig, NoFeasiblePlan
from mpc_dex.primitives import PrimitiveBasis
from mpc_dex.rollouts import RolloutCostConfig
from mpc_dex.native_rollouts import NativeMujocoRolloutEvaluator
from mpc_dex.contracts import sha256
from scripts.probe_mpc import action_search_bounds
from scripts.audit_policy import PhysicsObserver, PHYSICS_COLUMNS, audit_episode, write_json
from scripts.audit_recorded_controls import RecordedPolicy
from recovery_recipe import ProjectedReferenceCEM, integration_state
FIELDS = ('obs', 'actions', 'rewards', 'next_obs', 'terminated', 'truncated', 'qpos', 'qvel', 'targets', 'integration_state')
protocol = json.loads((R / 'protocol.json').read_text()); ec = EnvConfig(**json.loads((R / 'config.json').read_text())['environment'])
COLL = A / 'mpc_cold_demonstrations_0001' / 'episodes'
torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False


@torch.no_grad()
def policy_action(agent, obs):
    mean, log_std = agent.actor(torch.as_tensor(obs[None], dtype=torch.float32, device=agent.device))
    return mean.tanh()[0].cpu().numpy().astype(np.float32)


def load_demo(name):
    with np.load(COLL / name / 'transitions.npz', allow_pickle=False) as z: src = {k: z[k].copy() for k in z.files}
    with np.load(COLL / name / 'physics.npz', allow_pickle=False) as z: phys = z['samples'].copy()
    rep = json.loads((COLL / name / 'report.json').read_text()); assert rep['strict_force_success'] and rep['stored']
    assert sha256(COLL / name / 'transitions.npz') == rep['transitions_sha256']
    return src, phys


class Recorder:
    """Cold-reset episode with bit-exact replay of recorded controls and a continuous observer."""
    def __init__(self):
        self.env = AllegroRotationEnv(ec); self.obs, self.info = self.env.reset(seed=42)
        self.initial = dict(obs=self.obs.copy(), qpos=self.env.data.qpos.copy(), qvel=self.env.data.qvel.copy(), ctrl=self.env.data.ctrl.copy(), integration_state=integration_state(self.env))
        self.q_ids = np.asarray(self.env.q_ids).copy(); self.observer = PhysicsObserver(self.env); self.rows = {k: [] for k in FIELDS}; self.origins = []; self.term = self.trunc = False
    def step(self, action, origin):
        assert not (self.term or self.trunc); prev = self.obs.copy()
        self.obs, reward, self.term, self.trunc, self.info = self.env.step(np.asarray(action, np.float32))
        for k, v in dict(obs=prev, actions=np.asarray(action, np.float32).copy(), rewards=reward, next_obs=self.obs.copy(), terminated=self.term, truncated=self.trunc, qpos=self.env.data.qpos.copy(), qvel=self.env.data.qvel.copy(), targets=self.env.data.ctrl.copy(), integration_state=integration_state(self.env)).items():
            self.rows[k].append(v)
        self.origins.append(origin); return self.term or self.trunc
    def replay(self, src, phys, count, origin='demo_prefix'):
        for i in range(count):
            np.testing.assert_array_equal(self.obs, src['obs'][i]); ended = self.step(src['actions'][i], origin)
            for k in FIELDS: np.testing.assert_array_equal(self.rows[k][-1], src[k][i])
            assert not ended
        if count: np.testing.assert_array_equal(np.asarray(self.observer.rows), phys[:count * self.env.substeps])
    def strict(self):
        ob = self.observer; physics = np.asarray(ob.rows, np.float64); col = {k: physics[:, i] for i, k in enumerate(PHYSICS_COLUMNS)}
        ext = ('palm_force_norm', 'table_force_norm', 'other_force_norm', 'table_normal_force', 'other_normal_force', 'object_applied_wrench_norm', 'object_applied_generalized_force_norm')
        no_external = not ob.palm_support_violation and all(np.max(col[k], initial=0) <= 1e-6 for k in ext)
        return bool(self.term and not self.trunc and self.info.get('success') and not self.info.get('drop') and col['force_bearing_fingers'][-1] >= 2 and ob.raw_angle >= 2 * np.pi and ob.qualified_angle >= 2 * np.pi and ob.force_qualified_angle >= 2 * np.pi and col['force_retained'][-1] and not ob.independent_drop and no_external and self.env.model.neq == 0), physics
    def close(self): self.observer.close(); self.env.close()


def independent_replay(actions, original_rows, initial, original_physics):
    env = AllegroRotationEnv(ec); rows = {k: [] for k in FIELDS}; init = {}; reset, step = env.reset, env.step; rec = RecordedPolicy(actions)
    def rreset(*a, **k):
        r_ = reset(*a, **k); init.update(obs=r_[0].copy(), qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(), ctrl=env.data.ctrl.copy(), integration_state=integration_state(env)); return r_
    class Pol:
        def act(self, obs, deterministic=True):
            a = rec.act(obs, deterministic=deterministic); rows['obs'].append(obs.copy()); rows['actions'].append(a.copy()); return a
    def rstep(a):
        r_ = step(a); ob, reward, term, trunc, _ = r_
        for k, v in dict(next_obs=ob.copy(), rewards=reward, terminated=term, truncated=trunc, qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(), targets=env.data.ctrl.copy(), integration_state=integration_state(env)).items(): rows[k].append(v)
        return r_
    env.reset, env.step = rreset, rstep
    try:
        report, physics, commands = audit_episode(env, Pol(), 42, len(actions), True)
        same = all(np.array_equal(np.asarray(rows[k]), original_rows[k]) for k in FIELDS) and all(np.array_equal(init[k], initial[k]) for k in initial) and np.array_equal(physics, original_physics)
        report['bitexact_replay'] = bool(same)
        report['strict_force_success'] = bool(report['environment_success'] and report['finger_supported_rotation_success'] and report['terminated'] and not report['truncated'] and not report['environment_drop'] and not report['independent_drop'] and not report['independent_palm_support_violation'] and report['no_external_support'])
        return report
    finally: env.close()


def mpc_branch(agent, src, phys, prefix, policy_actions, b, max_corrections, seed=42):
    rec = Recorder(); started = time.monotonic()
    try:
        # Search bounds are fixed at the cold reset (as in the collector); a policy control may exceed the
        # .98 search window, in which case the planner's first knot starts outside its own search range.
        low, high, search = action_search_bounds(rec.env, protocol['action_bound'])
        rec.replay(src, phys, prefix)
        for j in range(b):
            np.testing.assert_array_equal(policy_action(agent, rec.obs), policy_actions[j])
            assert not rec.step(policy_actions[j], 'policy')
        handoff_target_within_search_bounds = bool(np.all(rec.env.data.ctrl >= low) and np.all(rec.env.data.ctrl <= high))
        with np.load(R / 'inputs/previous_recovery/prior_transitions.npz') as a:
            frames = np.vstack((rec.initial['qpos'][rec.q_ids].copy(), a['qpos'][:, rec.q_ids]))
        prior = PrimitiveBasis.from_trajectories(frames, provenance=dict(source=str(R / 'inputs/previous_recovery/prior_transitions.npz'), sha256=sha256(R / 'inputs/previous_recovery/prior_transitions.npz'), purpose='measured 40 mm diagnostic PCA prior; not a SAC primitive', successful_policy=False))
        planner = ProjectedReferenceCEM(prior, low, high, CEMConfig(**protocol['cem']), seed); planner.execution_start = rec.env.data.ctrl.copy(); planner.reset(rec.env.data.ctrl.copy())
        evaluator = NativeMujocoRolloutEvaluator(rec.env, timestep_seconds=protocol['cem']['timestep_seconds'], config=RolloutCostConfig(**protocol['cost']), require_native=True, cache_directory=R / 'native_cache')
        status = 'budget'; planning = []
        try:
            for _ in range(max_corrections):
                before = integration_state(rec.env); planner.execution_start = rec.env.data.ctrl.copy(); t0 = time.monotonic()
                try: target, diagnostics = planner.plan(evaluator)
                except NoFeasiblePlan: status = 'no_feasible_plan'; break
                assert np.array_equal(before, integration_state(rec.env)); planning.append(time.monotonic() - t0)
                action = rec.env.to_action(target); assert float(abs(action).max()) <= search['effective_bound']
                if rec.step(action, 'mpc'): status = 'boundary'; break
        finally: evaluator.close()
        strict, physics = rec.strict()
        rows = {k: np.asarray(v) for k, v in rec.rows.items()}
        report = dict(status=status, strict_force_success=strict, controls=len(rows['actions']), demo_prefix_controls=prefix, policy_controls=b, mpc_controls=int(sum(o == 'mpc' for o in rec.origins)), handoff_target_within_search_bounds=handoff_target_within_search_bounds, failure_reason=rec.info.get('failure_reason'), final_force_qualified_angle=rec.observer.force_qualified_angle, planning_seconds_total=float(sum(planning)), wall_seconds=time.monotonic() - started, hand_joint_qpos_indices=rec.q_ids.tolist())
        return report, rows, rec.initial, physics, list(rec.origins)
    finally: rec.close()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('variant'); ap.add_argument('checkpoint'); ap.add_argument('demos', nargs='+'); ap.add_argument('--suffixes', default='1,2,3,4'); ap.add_argument('--branches', default='0,1,2'); ap.add_argument('--max-corrections', type=int, default=20); ap.add_argument('--out', default=None, help='collection directory (default: this stage)')
    a = ap.parse_args(); suffixes = [int(x) for x in a.suffixes.split(',')]; branches = [int(x) for x in a.branches.split(',')]
    global OUT
    if a.out: OUT = Path(a.out).resolve(); OUT.mkdir(parents=True, exist_ok=True)
    ck = A / a.variant / 'checkpoints' / a.checkpoint; manifest = json.loads((ck / 'manifest.json').read_text()); assert sha256(ck / 'agent.pt') == manifest['agent.pt']
    agent = SACAgent.load(ck / 'agent.pt', 'cuda'); (OUT / 'episodes').mkdir(exist_ok=True)
    with (OUT / f"attempts_{'_'.join(a.demos)}.jsonl").open('a') as log:
        for name in a.demos:
            src, phys = load_demo(name); n = len(src['actions'])
            for s in suffixes:
                prefix = n - s
                # 1. the policy's own deterministic continuation from the handoff (records its controls and the failure point)
                rec = Recorder(); rec.replay(src, phys, prefix); pol = []
                while not (rec.term or rec.trunc) and len(pol) < 100:
                    act = policy_action(agent, rec.obs); pol.append(act); rec.step(act, 'policy')
                policy_outcome = dict(controls=len(pol), failure=rec.info.get('failure_reason'), success=bool(rec.strict()[0])); rec.close()
                for b in branches:
                    if policy_outcome['success'] or b >= len(pol): continue  # nothing to label, or the policy had already ended the episode
                    report, rows, initial, physics, origins = mpc_branch(agent, src, phys, prefix, pol, b, a.max_corrections)
                    ep = f'dagger_{name}_s{s}_b{b}'; report.update(name=ep, demo=name, suffix=s, policy_outcome=policy_outcome, policy_checkpoint=dict(variant=a.variant, checkpoint=a.checkpoint, agent_sha256=manifest['agent.pt']))
                    if report['strict_force_success']:
                        audit = independent_replay(rows['actions'], rows, initial, physics)
                        report['independent_audit'] = dict(bitexact_replay=audit['bitexact_replay'], strict_force_success=audit['strict_force_success'], final_force_qualified_angle=audit['final_force_qualified_angle'])
                        if audit['bitexact_replay'] and audit['strict_force_success']:
                            d = OUT / 'episodes' / ep; d.mkdir(exist_ok=True)
                            np.savez_compressed(d / 'transitions.npz', **rows, **{'initial_' + k: v for k, v in initial.items()}, joint_positions=np.vstack((initial['qpos'][np.asarray(report['hand_joint_qpos_indices'])][None, :], rows['qpos'][:, np.asarray(report['hand_joint_qpos_indices'])])), control_origin=np.asarray(origins))
                            np.savez_compressed(d / 'physics.npz', samples=physics, columns=np.asarray(PHYSICS_COLUMNS))
                            write_json(d / 'report.json', dict(report, transitions_sha256=sha256(d / 'transitions.npz'), physics_sha256=sha256(d / 'physics.npz'), stored=True)); report['stored'] = True
                    log.write(json.dumps(report) + '\n'); log.flush()
                    print(json.dumps({k: report[k] for k in ('name', 'status', 'controls', 'policy_controls', 'mpc_controls', 'strict_force_success')} | {'stored': report.get('stored', False), 'wall': round(report['wall_seconds'], 1)}), flush=True)


if __name__ == '__main__':
    main()
