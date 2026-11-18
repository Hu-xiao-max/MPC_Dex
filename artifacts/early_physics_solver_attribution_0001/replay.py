"""Two fixed three-control replays; no settings changes or policy inference."""
from pathlib import Path
import hashlib
import json
import platform
import sys

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D/'source'))
import mujoco
import numpy as np
from mpc_dex.env import AllegroRotationEnv, EnvConfig
from scripts.audit_policy import PhysicsObserver, PHYSICS_COLUMNS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


def state(env):
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    result = np.empty(mujoco.mj_stateSize(env.model, spec))
    mujoco.mj_getState(env.model, env.data, result, spec)
    return result


TOTAL_STEPS = 0


def capture(env, index, phase):
    m, d = env.model, env.data
    contacts = []
    for c in d.contact:
        force = np.zeros(6)
        mujoco.mj_contactForce(m, d, len(contacts), force)
        fields = {k: np.asarray(getattr(c,k)).tolist() for k in
                  ['dim','dist','geom1','geom2','efc_address','exclude','includemargin',
                   'pos','frame','friction','solref','solreffriction','solimp','mu','H']}
        fields.update(force=force.tolist(), object_contact=bool(env.object_geom in (c.geom1,c.geom2)))
        contacts.append(fields)
    stats=[]
    for island in range(max(1,d.nisland)):
        count=int(d.solver_niter[island])
        assert count<=mujoco.mjNSOLVER
        stats.append(dict(island=island,niter=count,nnz=int(d.solver_nnz[island]),
            iterations=[{key:np.asarray(getattr(d.solver[island*mujoco.mjNSOLVER+i],key)).item()
                        for key in ['improvement','gradient','lineslope','nactive','nchange','neval','nupdate']}
                        for i in range(count)]))
    Ma=np.empty(m.nv);mujoco.mj_mulM(m,d,Ma,d.qacc)
    residual=Ma-d.qfrc_smooth-d.qfrc_constraint
    matrix=np.empty((m.nv,m.nv));mujoco.mj_fullM(m,d,matrix)
    # This is the primal stationarity residual, not forward/inverse dynamics
    # comparison. FWDINV is disabled and its untouched zero fields are invalid
    # as a convergence certificate.
    scale=1/float(np.trace(matrix))
    return dict(index=index,phase=phase,time=float(d.time),nisland=int(d.nisland),
        ncon=int(d.ncon),nefc=int(d.nefc),solver=stats,solver_fwdinv=d.solver_fwdinv.tolist(),
        residual=residual.tolist(),residual_norm=float(np.linalg.norm(residual)),
        mass_trace_scale=scale,scaled_residual_norm=float(scale*np.linalg.norm(residual)),
        scaled_mass_gap=float(.5*scale*residual@np.linalg.solve(matrix,residual)),
        contacts=contacts,efc_type=d.efc_type.tolist(),efc_id=d.efc_id.tolist(),
        efc_state=d.efc_state.tolist(),efc_force=d.efc_force.tolist(),
        efc_pos=d.efc_pos.tolist(),efc_aref=d.efc_aref.tolist(),efc_b=d.efc_b.tolist(),
        efc_D=d.efc_D.tolist(),efc_R=d.efc_R.tolist(),
        qacc=d.qacc.tolist(),qacc_smooth=d.qacc_smooth.tolist(),
        qfrc_smooth=d.qfrc_smooth.tolist(),qfrc_constraint=d.qfrc_constraint.tolist(),
        warning_counts=d.warning.number.tolist())


def replay(label, config):
    global TOTAL_STEPS
    output=D/label
    output.mkdir()
    source=load(D/'inputs'/f'{label}_transitions.npz')
    source_physics=load(D/'inputs'/f'{label}_physics.npz')
    env=AllegroRotationEnv(config)
    obs,info=env.reset(seed=42)
    initial=state(env)
    for field in ['qpos','qvel','ctrl']:
        assert np.array_equal(getattr(env.data,field), source['initial_'+field])
    if 'initial_integration_state' in source:
        assert np.array_equal(initial,source['initial_integration_state'])
    mujoco.mj_saveModel(env.model,str(output/'model.mjb'),None)
    option={name:np.asarray(getattr(env.model.opt,name)).tolist() for name in dir(env.model.opt)
            if not name.startswith('_')}
    geoms=[]
    for i in range(env.model.ngeom):
        geoms.append(dict(id=i,name=mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_GEOM,i),
            body=mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_BODY,int(env.model.geom_bodyid[i])),
            type=int(env.model.geom_type[i]),friction=env.model.geom_friction[i].tolist(),
            condim=int(env.model.geom_condim[i]),margin=float(env.model.geom_margin[i]),
            gap=float(env.model.geom_gap[i]),solref=env.model.geom_solref[i].tolist(),
            solimp=env.model.geom_solimp[i].tolist()))
    write(output/'model_settings.json',dict(option=option,geoms=geoms,
        qpos_addresses=env.q_ids.tolist(),qvel_addresses=env.v_ids.tolist(),
        object_qpos_address=env.object_qadr,object_dof_address=env.object_dadr,
        object_geom=env.object_geom,nv=env.model.nv,nq=env.model.nq,
        integrator_name='implicitfast',solver_name='Newton',cone_name='elliptic',
        fwdinv_diagnostics_enabled=False,solver_statistics_both_phases=True))
    observer=PhysicsObserver(env)
    real_step,real_forward=mujoco.mj_step,mujoco.mj_forward
    diagnostics=[];physics_states=[];pre_states=[]
    count=0
    def step(model,data,*args,**kwargs):
        nonlocal count
        global TOTAL_STEPS
        assert model is env.model and data is env.data and not args and not kwargs
        assert TOTAL_STEPS<300
        pre_states.append(state(env))
        real_step(model,data)
        count+=1;TOTAL_STEPS+=1
        diagnostics.append(capture(env,count,'after_mj_step_pre_forward'))
    def forward(model,data,*args,**kwargs):
        assert model is env.model and data is env.data and not args and not kwargs
        real_forward(model,data)
        diagnostics.append(capture(env,count,'after_mj_forward'))
        physics_states.append(state(env))
    mujoco.mj_step,mujoco.mj_forward=step,forward
    fields={k:[] for k in ['obs','actions','next_obs','rewards','terminated','truncated','qpos','qvel','targets','integration_state']}
    try:
        for index,action in enumerate(source['actions'][:3]):
            assert action.dtype==np.float32
            before=obs.copy()
            obs,reward,terminated,truncated,info=env.step(action)
            values=dict(obs=before,actions=action.copy(),next_obs=obs.copy(),rewards=reward,
                terminated=terminated,truncated=truncated,qpos=env.data.qpos.copy(),
                qvel=env.data.qvel.copy(),targets=env.data.ctrl.copy(),integration_state=state(env))
            for key,value in values.items():
                fields[key].append(value)
                if key not in source:continue
                expected=source[key][index]
                if key in ('obs','next_obs'):expected=expected[:60]
                assert np.array_equal(value,expected),(label,index,key)
            assert not terminated and not truncated
        assert count==150 and len(observer.rows)==150 and len(diagnostics)==300
        assert np.array_equal(np.asarray(observer.rows),source_physics['samples'][:150])
        np.savez_compressed(output/'transitions.npz',**{k:np.asarray(v) for k,v in fields.items()},
            initial_integration_state=initial)
        np.savez_compressed(output/'physics.npz',samples=np.asarray(observer.rows),columns=np.asarray(PHYSICS_COLUMNS),
            pre_integration_states=np.asarray(pre_states),post_integration_states=np.asarray(physics_states))
        write(output/'solver_diagnostics.json',diagnostics)
        write(output/'report.json',dict(status='complete',source_fields_bitexact=True,
            source_physics_rows_bitexact=150,physics_steps=150,controls=3,
            diagnostics_records=len(diagnostics),config_settings_unchanged=True,
            source_initial_integration_available='initial_integration_state' in source,
            max_action_abs=float(abs(source['actions'][:3]).max()),
            final_info={k:info[k] for k in ['angle','retained_angle','success','drop','palm_support_violation']}))
    finally:
        mujoco.mj_step,mujoco.mj_forward=real_step,real_forward
        observer.close();env.close()


def main():
    assert not (D/'nominal').exists() and not (D/'head').exists()
    frozen={str(p.relative_to(D)):sha(p) for p in sorted(D.rglob('*')) if p.is_file()
            and '__pycache__' not in p.parts and p.name!='replay.log'}
    write(D/'frozen_inputs.json',frozen)
    config=EnvConfig(**json.loads((D/'config.json').read_text())['environment'])
    assert config.timestep==.002 and config.control_seconds==.1 and config.terminate_on_success
    write(D/'protocol.json',dict(kind='two-exact-first3-prefixes-extra-solver-diagnostics',
        branches=['nominal','head'],controls_per_branch=3,physics_steps_per_branch=150,
        total_physics_budget=300,altered_solver_or_physics_parameters=False,
        extra_forward_solve_calls=False,callbacks='Python read-only wrappers capture after existing mj_step and mj_forward',
        no_policy_inference_or_learning=True,source_config_transformation='only byte-identical initial_grasp path'))
    replay('nominal',config)
    replay('head',config)
    assert TOTAL_STEPS==300
    assert sha(D/'nominal/model.mjb')==sha(D/'head/model.mjb')
    assert all(sha(D/name)==digest for name,digest in frozen.items())
    write(D/'runtime.json',dict(python=platform.python_version(),platform=platform.platform(),numpy=np.__version__,
        mujoco=mujoco.__version__,mujoco_runtime=mujoco.mj_version(),
        library_sha256=sha(Path(mujoco.__file__).parent/f'libmujoco.so.{mujoco.__version__}'),
        compiled_model_sha256=sha(D/'nominal/model.mjb'),total_physics_steps=TOTAL_STEPS))
    print(json.dumps(dict(status='complete',total_physics_steps=TOTAL_STEPS,both_source_traces_bitexact=True)))


if __name__=='__main__':main()
