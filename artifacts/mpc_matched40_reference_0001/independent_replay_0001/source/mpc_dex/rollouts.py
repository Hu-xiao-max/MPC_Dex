"""Parallel synchronized MuJoCo rollouts for primitive CEM planning."""
from __future__ import annotations

import os
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import mujoco
from mujoco.rollout import Rollout
import numpy as np
from scipy.spatial.transform import Rotation

from .env import AllegroRotationEnv


@dataclass
class RolloutCostConfig:
    # Unpublished weights are explicit reproduction choices, not author values.
    position_weight: float = 1000.0
    principal_weight: float = 1.0
    perpendicular_weight: float = 10.0
    workspace_radius: float = 0.18
    workers: int = 8
    rotation_progress: str = 'raw'

    def __post_init__(self):
        if self.rotation_progress not in {'raw', 'retained'}:
            raise ValueError("rotation_progress must be 'raw' or 'retained'")


class MujocoRolloutEvaluator:
    """Predict candidate joint targets without mutating the physical environment.

    Each worker owns a separate data instance and performs the environment's
    exact ``mj_step``/``mj_forward`` transition. Object--finger contact loss,
    table contact, palm support and workspace escape are checked at every physics step, and
    an invalid candidate stops immediately. This costs more Python overhead
    than whole-horizon native rollouts, but cannot score an airborne spin as a
    feasible plan merely because it has not reached the table yet.
    """

    def __init__(self, env: AllegroRotationEnv, timestep_seconds=.01, config=None):
        self.env = env
        self.config = config or RolloutCostConfig()
        self.repeat = round(timestep_seconds / env.config.timestep)
        if self.repeat < 1 or not np.isclose(self.repeat*env.config.timestep,timestep_seconds):
            raise ValueError('Planner timestep must be multiple of simulation timestep')
        self.workers = max(1,min(self.config.workers,os.cpu_count() or 1))
        # Retained for direct native prediction comparisons. Production scoring
        # uses the exact workers below; native Rollout has no rejection callback.
        self.pool = Rollout(nthread=self.workers)
        self.exact_pool = ThreadPoolExecutor(max_workers=self.workers)
        self.worker_data = [mujoco.MjData(env.model) for _ in range(self.workers)]
        self.state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        self._finger_masks = np.zeros(env.model.ngeom, dtype=np.uint8)
        self._palm_geoms = np.asarray(env.model.geom_bodyid == env.palm_id)
        finger_masks = {'ff': 1, 'mf': 2, 'rf': 4, 'th': 8}
        for geom, body in enumerate(env.model.geom_bodyid):
            name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, int(body)) or ''
            self._finger_masks[geom] = finger_masks.get(name[:2], 0)
        self.last_diagnostics = {}

    def _simulate_candidates(self, worker, candidates, sequences, snapshot,
                             initial_contactless, states, finger_counts, step_counts, reasons):
        """Stop each candidate at its first invalid post-transition state."""
        e = self.env
        m = e.model
        d = self.worker_data[worker]
        for candidate in candidates:
            # Copy warmstart, applied forces and all other integration inputs,
            # not only qpos/qvel. No worker ever writes to the live environment.
            mujoco.mj_copyData(d, m, snapshot)
            if self._initial_palm_support_violation:
                step_counts[candidate] = 0
                reasons[candidate] = 'palm_support'
                continue
            contactless = initial_contactless
            index = 0
            reason = None
            for target in sequences[candidate]:
                d.ctrl[:] = target
                for _ in range(self.repeat):
                    previous_time = d.time
                    mujoco.mj_step(m, d)
                    mujoco.mj_forward(m, d)
                    state = states[candidate, index]
                    mujoco.mj_getState(m, d, state, self.state_spec)
                    index += 1
                    if not np.isfinite(state).all() or d.time <= previous_time:
                        reason = 'divergence'
                        break
                    # The contact set must be computed after mj_forward, just
                    # as in env.step; mj_step alone exposes pre-transition contacts.
                    contacts = d.contact.geom
                    other = np.concatenate((contacts[contacts[:, 0] == e.object_geom, 1],
                                            contacts[contacts[:, 1] == e.object_geom, 0]))
                    # Multiple contact points or links of one finger still
                    # count as one finger, matching env.finger_contacts.
                    fingers = int(np.bitwise_or.reduce(self._finger_masks[other], initial=0)).bit_count()
                    finger_counts[candidate, index-1] = fingers
                    contactless = 0.0 if fingers else contactless + e.config.timestep
                    if (other == e.table_geom).any():
                        reason = 'table_contact'
                    elif (e.config.task == 'rotation' and self._palm_geoms[other].any()
                          and e.palm_support_force(d) >
                          e.config.palm_support_force_tolerance):
                        reason = 'palm_support'
                    elif np.linalg.norm(e.object_position(d) - e.initial_position) > self.config.workspace_radius:
                        reason = 'workspace'
                    elif contactless >= e.config.contact_loss_seconds:
                        reason = 'contact_loss'
                    if reason is not None:
                        break
                if reason is not None:
                    break
            step_counts[candidate] = index
            reasons[candidate] = reason

    def __call__(self, sequences):
        start=time.perf_counter()
        sequences=np.asarray(sequences,dtype=np.float64)
        if (sequences.ndim!=3 or sequences.shape[2]!=16 or min(sequences.shape[:2])<1
                or not np.isfinite(sequences).all()):
            raise ValueError('Expected finite sequences [N,H,16]')
        if np.any(sequences<self.env.low-1e-8) or np.any(sequences>self.env.high+1e-8):
            raise ValueError('Joint targets exceed actuator limits')
        e=self.env;m=e.model
        self._initial_palm_support_violation = (
            e.config.task == 'rotation' and e._palm_support_violation)
        initial=np.empty(mujoco.mj_stateSize(m,self.state_spec))
        mujoco.mj_getState(m,e.data,initial,self.state_spec)
        snapshot=mujoco.MjData(m)
        mujoco.mj_copyData(snapshot,m,e.data)
        physics_steps=sequences.shape[1]*self.repeat
        # Unexecuted tails are initialized to a valid state solely for the
        # vectorized cost calculation. Every stopped candidate receives +inf.
        states=np.broadcast_to(initial,(len(sequences),physics_steps,len(initial))).copy()
        finger_counts=np.zeros((len(sequences),physics_steps),dtype=np.uint8)
        step_counts=np.zeros(len(sequences),dtype=int)
        reasons=[None]*len(sequences)
        active_workers=min(self.workers,len(sequences))
        futures=[self.exact_pool.submit(self._simulate_candidates,worker,
            range(worker,len(sequences),active_workers),sequences,snapshot,
            e._contactless_seconds,states,finger_counts,step_counts,reasons)
            for worker in range(active_workers)]
        for future in futures:
            future.result()
        qa=1+e.object_qadr
        positions=states[:,:,qa:qa+3]
        quats=states[:,:,qa+3:qa+7]
        finite=np.isfinite(states).all(axis=(1,2))
        safeq=np.nan_to_num(quats.copy())
        norm=np.linalg.norm(safeq,axis=2)
        bad=norm<1e-8;safeq[bad]=[1,0,0,0]
        finite &= ~bad.any(axis=1)
        mats=Rotation.from_quat(safeq.reshape(-1,4),scalar_first=True).as_matrix().reshape(*safeq.shape[:2],3,3)
        palm_rot=e.data.xmat[e.palm_id].reshape(3,3)
        local_mats=np.einsum('ij,ntjk->ntik',palm_rot.T,mats)
        local_pos=np.einsum('ij,ntj->nti',palm_rot.T,positions-e.data.xpos[e.palm_id])
        # Integrate local principal-axis twist from short-arc quaternion deltas,
        # consistent with the environment. Euler projection can miscount tumbles.
        initial_quat=np.broadcast_to(e.data.xquat[e.object_id],(len(sequences),1,4))
        previous=np.concatenate((initial_quat,safeq[:,:-1]),axis=1)
        delta_w=np.sum(previous*safeq,axis=2)
        delta_y=(previous[:,:,0]*safeq[:,:,2]-previous[:,:,2]*safeq[:,:,0]
                 +previous[:,:,1]*safeq[:,:,3]-previous[:,:,3]*safeq[:,:,1])
        sign=np.where(delta_w<0,-1.,1.)
        axial_increments=2*np.arctan2(sign*delta_y,sign*delta_w)
        axial_progress=np.cumsum(axial_increments,axis=1)
        initial_r=Rotation.from_quat(e.initial_orientation,scalar_first=True).as_matrix()
        if e.config.task=='rotation':
            # The saved grasp may orient the rod away from palm +y. Match
            # env.axis_tilt by comparing its current and episode-initial axes.
            axis_dot=np.einsum('nti,i->nt',local_mats[:,:,:,1],initial_r[:,1])
        else:
            heading_mats=np.einsum('ij,ntjk->ntik',initial_r.T,local_mats)
            heading=np.arctan2(heading_mats[:,:,1,0],heading_mats[:,:,0,0])
            angle_error=(heading-e.goal+math.pi)%(2*math.pi)-math.pi
            axis_dot=heading_mats[:,:,2,2]
        tilt_error=np.arccos(np.clip(axis_dot,-1,1))
        if e.config.task=='rotation':
            retained=(finger_counts>=2) & (tilt_error<=math.radians(e.config.success_axis_tolerance_degrees))
            # Match the environment's retained-angle increments: forward
            # twist needs a retained grasp; every reverse increment counts.
            qualified=np.where((axial_increments<0) | retained,axial_increments,0.)
            retained_progress=np.cumsum(qualified,axis=1)
            progress=retained_progress if self.config.rotation_progress=='retained' else axial_progress
            # Preserve the existing horizon-relative target in both modes.
            angle_error=progress-2*math.pi
        delta=local_pos-e.initial_position
        cost=(self.config.position_weight*np.sum(delta*delta,axis=2)
              +self.config.principal_weight*angle_error**2
              +self.config.perpendicular_weight*tilt_error**2).sum(axis=1)*e.config.timestep
        invalid=~finite | np.array([reason is not None for reason in reasons])
        cost[invalid]=np.inf
        self.last_diagnostics={'candidates':len(sequences),'invalid':int(invalid.sum()),
            'physics_steps_per_candidate':physics_steps,
            'candidate_physics_steps':step_counts.tolist(),
            'evaluated_physics_steps':int(step_counts.sum()),
            'rejection_reasons':reasons,
            'palm_support_constraint':e.config.task == 'rotation',
            'palm_support_force_tolerance':e.config.palm_support_force_tolerance,
            'initial_palm_support_violation':bool(self._initial_palm_support_violation),
            'wall_seconds':time.perf_counter()-start,
            'early_termination_implemented':True,
            'backend':'exact_step_threads'}
        if e.config.task=='rotation':
            valid=~invalid
            self.last_diagnostics.update(rotation_progress=self.config.rotation_progress,
                raw_final_progress_range=([float(axial_progress[valid,-1].min()),
                    float(axial_progress[valid,-1].max())] if valid.any() else None),
                retained_final_progress_range=([float(retained_progress[valid,-1].min()),
                    float(retained_progress[valid,-1].max())] if valid.any() else None))
        return cost

    def close(self):
        self.exact_pool.shutdown(wait=True)
        self.pool.close()
