"""Measure planning mismatch caused by float32 normalized-action conversion."""
from pathlib import Path
import json
import shutil
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from mpc_dex.cli import read_config
from mpc_dex.env import AllegroRotationEnv
from mpc_dex.contracts import sha256
from mpc_dex.mpc import CEMConfig
from mpc_dex.primitives import PrimitiveBasis
from mpc_dex.rollouts import RolloutCostConfig
from mpc_dex.native_rollouts import NativeMujocoRolloutEvaluator
from scripts.probe_mpc import SlewLimitedCEM,action_search_bounds

HERE=Path(__file__).resolve().parent
if (HERE/'report.json').exists():raise FileExistsError(HERE/'report.json')
config=ROOT/'configs/diagnostic_rotation_40mm_continuous.json'
source=ROOT/'artifacts/mpc_matched40_bound098_0001/transitions.npz'
prior_source=ROOT/'artifacts/mpc_fullturn_prior_40mm_0001/branch_diagnosis/prefix19_blend00/transitions.npz'
_,ec=read_config(config)
with np.load(source,allow_pickle=False) as data:actions=data['actions'].copy()
with np.load(prior_source,allow_pickle=False) as data:prior_q=data['qpos'].copy()
snapshot=HERE/'source';snapshot.mkdir()
names=['mpc_dex/env.py','mpc_dex/mpc.py','mpc_dex/rollouts.py','mpc_dex/native_rollouts.py','scripts/probe_mpc.py','uv.lock']
hashes={name:sha256(ROOT/name) for name in names}
for name in names:shutil.copyfile(ROOT/name,snapshot/Path(name).name)
shutil.copyfile(config,snapshot/'config.json')
shutil.copyfile(source,snapshot/'transitions.npz')
results=[]
for prefix in (0,5,15):
    env=AllegroRotationEnv(ec);env.reset(seed=42)
    initial=env.data.qpos[env.q_ids].copy()
    for action in actions[:prefix]:
        _,_,terminated,truncated,_=env.step(action)
        assert not (terminated or truncated)
    prior=PrimitiveBasis.from_trajectories(np.vstack((initial,prior_q[:,env.q_ids])))
    low,high,_=action_search_bounds(env,.98)
    planner=SlewLimitedCEM(prior,low,high,CEMConfig(timestep_seconds=.1),42,reference_candidates=True)
    planner.max_target_delta=None;planner.execution_start=env.data.ctrl.copy();planner.reset(env.data.ctrl)
    sequence=planner._decode(planner._sample(planner.nominal_knots,planner.std_knots))
    mapped=env.to_joint_targets(env.to_action(sequence))
    evaluator=NativeMujocoRolloutEvaluator(env,.1,RolloutCostConfig(workers=8,rotation_progress='retained',perpendicular_weight=100),require_native=True)
    try:
        raw_cost=evaluator(sequence);raw_diag=evaluator.last_diagnostics.copy()
        mapped_cost=evaluator(mapped);mapped_diag=evaluator.last_diagnostics.copy()
        valid=np.isfinite(raw_cost)&np.isfinite(mapped_cost)
        best=int(raw_cost.argmin())
        result=dict(prefix=prefix,candidates=len(sequence),max_target_mapping_difference=float(np.max(np.abs(mapped-sequence))),raw_feasible=int(np.isfinite(raw_cost).sum()),mapped_feasible=int(np.isfinite(mapped_cost).sum()),feasibility_disagreements=int(np.count_nonzero(np.isfinite(raw_cost)!=np.isfinite(mapped_cost))),shared_feasible=int(valid.sum()),max_shared_cost_difference=float(np.max(np.abs(raw_cost[valid]-mapped_cost[valid]))) if valid.any() else None,raw_best=best,mapped_best=int(mapped_cost.argmin()),raw_best_cost=float(raw_cost[best]) if np.isfinite(raw_cost[best]) else None,raw_best_after_mapping_cost=float(mapped_cost[best]) if np.isfinite(mapped_cost[best]) else None)
        np.savez_compressed(HERE/f'prefix{prefix}_candidates.npz',joint_targets=sequence,executed_targets=mapped,raw_cost=raw_cost,mapped_cost=mapped_cost)
        if np.isfinite(raw_cost[best]) and np.isfinite(mapped_cost[best]):
            evaluator(sequence[best:best+1]);raw_qpos=evaluator.worker_data[0].qpos.copy();raw_qvel=evaluator.worker_data[0].qvel.copy()
            evaluator(mapped[best:best+1]);mapped_qpos=evaluator.worker_data[0].qpos.copy();mapped_qvel=evaluator.worker_data[0].qvel.copy()
            for action in env.to_action(sequence[best]):
                _,_,terminated,truncated,_=env.step(action)
                assert not (terminated or truncated)
            result.update(raw_prediction_qpos_max_error=float(np.max(np.abs(raw_qpos-env.data.qpos))),raw_prediction_qvel_max_error=float(np.max(np.abs(raw_qvel-env.data.qvel))),mapped_prediction_qpos_bit_exact=bool(np.array_equal(mapped_qpos,env.data.qpos)),mapped_prediction_qvel_bit_exact=bool(np.array_equal(mapped_qvel,env.data.qvel)))
        results.append(result)
        print(json.dumps(result),flush=True)
    finally:evaluator.close();env.close()
assert all(sha256(ROOT/name)==digest for name,digest in hashes.items())
(HERE/'report.json').write_text(json.dumps(dict(purpose='Action-mapping precision diagnosis; no algorithm or physics changes',results=results,source_hashes=hashes,script_sha256=sha256(__file__),input_sha256=sha256(source),primitive_source_sha256=sha256(prior_source)),indent=2)+'\n')
