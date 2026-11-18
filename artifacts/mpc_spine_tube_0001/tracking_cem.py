"""Reference-tracking variant of the archived corrected CEM: the warm start is the spine's own upcoming controls and the
sampling floors are scaled down, so the planner searches locally around the reference and returns it unchanged whenever it
is feasible and lowest-cost. Same cost, same rollouts, same bounds; only the proposal distribution differs."""
import numpy as np
from recovery_recipe import ProjectedReferenceCEM


class TrackingCEM(ProjectedReferenceCEM):
    def __init__(self, *args, floor_scale=.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.exploration_floor = self.exploration_floor * float(floor_scale)
        self.refinement_floor = self.refinement_floor * float(floor_scale)
        self.floor_scale = float(floor_scale); self.reference_sequence = None; self.reference_cursor = 0

    def encode(self, target):
        target = np.clip(np.asarray(target, dtype=np.float64), self.action_low, self.action_high)
        c = self.primitive.project(target)
        return np.concatenate([c, target - self.primitive.reconstruct(c)])

    def set_reference(self, joint_targets, cursor=0):
        """joint_targets: (T, 16) upcoming reference targets (actuator space); the knot k of the warm start is the target at cursor + k (held at the end)."""
        self.reference_sequence = np.asarray(joint_targets, dtype=np.float64); self.reference_cursor = int(cursor)
        self._impose_reference()

    def _impose_reference(self):
        T = len(self.reference_sequence)
        for k in range(self.nominal_knots.shape[0]):
            self.nominal_knots[k] = self.encode(self.reference_sequence[min(self.reference_cursor + k, T - 1)])

    def plan(self, evaluate):
        if self.reference_sequence is not None:
            self._impose_reference()  # receding-horizon tracking: the warm start is always the reference, the spread is the elites' (floored)
        target, diagnostics = super().plan(evaluate)
        self.reference_cursor += 1
        diagnostics['tracking'] = dict(floor_scale=self.floor_scale, reference_cursor=self.reference_cursor)
        return target, diagnostics


class TrackingPenaltyEvaluator:
    """Adds weight * mean-squared deviation from the planner's current reference (per step, actuator target space) to the task
    cost of every candidate, so that the optimum is the feasible trajectory nearest the reference instead of whichever feasible
    deviation the task cost happens to prefer. Infeasible candidates keep their infinite cost."""
    def __init__(self, inner, planner, weight):
        self.inner, self.planner, self.weight = inner, planner, float(weight)

    def __call__(self, sequences):
        costs = np.asarray(self.inner(sequences), dtype=np.float64)
        ref = self.planner.reference_sequence; T = len(ref); c = self.planner.reference_cursor; H = sequences.shape[1]
        ref_steps = np.stack([ref[min(c + h, T - 1)] for h in range(H)])
        return costs + self.weight * ((sequences - ref_steps[None]) ** 2).sum(axis=(1, 2)) / H

    @property
    def last_diagnostics(self): return self.inner.last_diagnostics

    def close(self): self.inner.close()


class StateTrackingPenaltyEvaluator:
    """Adds weight * mean over control steps of the squared deviation of the rolled-out state (object pose and hand joint
    positions) from the spine's recorded state after the same control, so that candidates are preferred for keeping the STATE
    near the demonstration rather than the actions. Captures the evaluator's shared rollout-state buffer through its
    _simulate_candidates hook; the base cost and feasibility are unchanged, infeasible candidates stay infinite.
    reference_qpos: (T, nq) recorded qpos after controls t = 0..T-1 of the demonstration; cursor: index of the next control."""
    def __init__(self, inner, planner, weight, reference_qpos, qpos_dims, state_qpos_offset=1):
        self.inner, self.planner, self.weight = inner, planner, float(weight)
        self.reference_qpos = np.asarray(reference_qpos, dtype=np.float64); self.dims = np.asarray(qpos_dims); self.off = int(state_qpos_offset)
        self._states = None; base = inner
        while not hasattr(base, '_simulate_candidates'):  # walk through wrapper evaluators to the rollout evaluator
            base = base.inner
        original = base._simulate_candidates
        def hooked(worker, candidates, sequences, snapshot, contactless, states, finger_counts, step_counts, reasons):
            self._states = states
            return original(worker, candidates, sequences, snapshot, contactless, states, finger_counts, step_counts, reasons)
        base._simulate_candidates = hooked

    def __call__(self, sequences):
        costs = np.asarray(self.inner(sequences), dtype=np.float64); states = self._states; n, H = sequences.shape[0], sequences.shape[1]
        repeat = states.shape[1] // H; idx = np.arange(1, H + 1) * repeat - 1  # physics step after each control
        c = self.planner.reference_cursor; T = len(self.reference_qpos)
        ref = np.stack([self.reference_qpos[min(c + h, T - 1)] for h in range(H)])[:, self.dims]  # (H, d)
        q = states[:, idx][:, :, self.off + self.dims]  # (n, H, d)
        pen = self.weight * ((q - ref[None]) ** 2).sum(axis=2).mean(axis=1)
        pen = np.where(np.isfinite(costs), pen, 0.)
        return costs + pen

    @property
    def last_diagnostics(self): return self.inner.last_diagnostics

    def close(self): self.inner.close()
