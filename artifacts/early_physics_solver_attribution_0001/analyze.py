"""Read saved diagnostics only. Does not import MuJoCo or run dynamics."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import numpy as np

D=Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def npz(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


def signature(row):
    return [(c['geom1'],c['geom2'],c['dim'],c['exclude']) for c in row['contacts']]


def main():
    raw={label:read(D/label/'solver_diagnostics.json') for label in ['nominal','head']}
    states={label:npz(D/label/'physics.npz') for label in raw}
    controls={label:npz(D/label/'transitions.npz') for label in raw}
    settings=read(D/'nominal/model_settings.json')
    # mjSTATE_INTEGRATION begins with time,qpos,qvel,act. Dimensions are checked
    # against saved control boundaries before using that standard state layout.
    A=states['nominal']['post_integration_states'];B=states['head']['post_integration_states']
    nq,nv=settings['nq'],settings['nv']
    for label in raw:
        s=states[label]['post_integration_states']
        assert np.array_equal(s[[49,99,149],1:1+nq],controls[label]['qpos'])
        assert np.array_equal(s[[49,99,149],1+nq:1+nq+nv],controls[label]['qvel'])
    qpos=np.max(abs(A[:,1:1+nq]-B[:,1:1+nq]),axis=1)
    qvel=np.max(abs(A[:,1+nq:1+nq+nv]-B[:,1+nq:1+nq+nv]),axis=1)
    time=A[:,0]
    assert np.array_equal(time,B[:,0])
    phases=['after_mj_step_pre_forward','after_mj_forward']
    grouped={phase:{label:[x for x in values if x['phase']==phase] for label,values in raw.items()} for phase in phases}
    summaries={}
    for phase,traces in grouped.items():
        summaries[phase]={}
        for label,values in traces.items():
            solvers=[s for r in values for s in r['solver']]
            ends=[s['iterations'][-1] for s in solvers if s['niter']]
            summaries[phase][label]=dict(solves=len(solvers),
                iteration_histogram=dict(sorted(Counter(s['niter'] for s in solvers).items())),
                max_iterations=max(s['niter'] for s in solvers),iteration_cap_hits=sum(s['niter']>=50 for s in solvers),
                max_last_gradient=max(s['gradient'] for s in ends),
                max_last_improvement=max(s['improvement'] for s in ends),
                last_gradient_at_least_config_tolerance=sum(s['gradient']>=1e-8 for s in ends),
                max_last_line_slope=max(abs(s['lineslope']) for s in ends),
                max_recorded_linesearch_cost_evaluations=max(s['neval'] for r in solvers for s in r['iterations']),
                max_stationarity_residual_norm=max(x['residual_norm'] for x in values),
                warnings_observed=any(any(x['warning_counts']) for x in values),
                fwdinv_enabled=False,fwdinv_fields_not_used_as_convergence_evidence=True)
    a,b=grouped['after_mj_forward']['nominal'],grouped['after_mj_forward']['head']
    contact_differences=[i for i,(x,y) in enumerate(zip(a,b)) if signature(x)!=signature(y)]
    iteration_differences=[i for i,(x,y) in enumerate(zip(a,b)) if [s['niter'] for s in x['solver']] != [s['niter'] for s in y['solver']]]
    state_differences=[i for i,(x,y) in enumerate(zip(a,b)) if x['efc_state']!=y['efc_state']]
    assert contact_differences[0]==131 and iteration_differences[0]==131 and state_differences[0]==119
    first=contact_differences[0]
    events=[]
    for i in [0,49,99,104,105,119,120,130,131,132,149]:
        aa,bb=a[i],b[i]
        events.append(dict(physics_step=i+1,time=float(time[i]),qpos_max_difference=float(qpos[i]),
            qvel_max_difference=float(qvel[i]),ncon=[aa['ncon'],bb['ncon']],nefc=[aa['nefc'],bb['nefc']],
            qacc_max_difference=float(abs(np.array(aa['qacc'])-bb['qacc']).max()),
            constraint_force_vector_difference_norm=float(np.linalg.norm(np.array(aa['qfrc_constraint'])-bb['qfrc_constraint'])),
            stationarity_residual_norms=[aa['residual_norm'],bb['residual_norm']],
            last_solver_stats=[aa['solver'][0]['iterations'][-1],bb['solver'][0]['iterations'][-1]],
            iterations=[aa['solver'][0]['niter'],bb['solver'][0]['niter']]))
    branch_contacts={label:[c for c in row[first]['contacts'] if (c['geom1'],c['geom2'])==(6,43)]
                     for label,row in [('nominal',a),('head',b)]}
    assert len(branch_contacts['nominal'])==1 and len(branch_contacts['head'])==2
    separation=float(np.linalg.norm(np.array(branch_contacts['head'][0]['pos'])-branch_contacts['head'][1]['pos']))
    zero_force_changes=[]
    for i in state_differences:
        if i>=first:break
        changed=np.flatnonzero(np.array(a[i]['efc_state'])!=b[i]['efc_state'])
        assert all(a[i]['efc_force'][j]==b[i]['efc_force'][j]==0 for j in changed)
        zero_force_changes.append(dict(step=i+1,time=float(time[i]),rows=changed.tolist(),
            all_changed_constraint_forces_zero=True,
            nominal_state=[a[i]['efc_state'][j] for j in changed],head_state=[b[i]['efc_state'][j] for j in changed]))
    crossings={}
    for name,values in [('qpos_max_difference',qpos),('qvel_max_difference',qvel)]:
        crossings[name]={}
        for t in [1e-8,1e-6,1e-5,1e-4,.001,.01]:
            idx=np.flatnonzero(values>t)
            crossings[name][str(t)]=None if not len(idx) else dict(step=int(idx[0]+1),time=float(time[idx[0]]),value=float(values[idx[0]]))
    result=dict(status='complete',total_new_physics_steps=300,extra_physics_in_analysis=0,
        exact_reproduction={label:read(D/label/'report.json') for label in raw},
        original_physics_and_solver_settings_unchanged=True,
        action_max_differences_by_control=np.max(abs(controls['nominal']['actions']-controls['head']['actions']),axis=1).tolist(),
        target_max_differences_by_control=np.max(abs(controls['nominal']['targets']-controls['head']['targets']),axis=1).tolist(),
        actual_solver_summaries=summaries,early_zero_force_constraint_state_changes=zero_force_changes,
        first_contact_set_difference=dict(step=first+1,time=float(time[first]),
            pair=dict(geom1=6,body1='ff_proximal',type1='box',geom2=43,body2='object',type2='mesh'),
            contacts=branch_contacts,head_contact_position_separation=separation,
            first_difference_is_post_forward=True,next_integrated_step=first+2,
            native_multicontact_enabled=True,ccd_tolerance=1e-6,ccd_iteration_cap=35,
            ccd_iteration_and_termination_status_not_exposed_in_mjData=True),
        contact_set_difference_steps=[i+1 for i in contact_differences],
        first_solver_iteration_count_difference_step=iteration_differences[0]+1,
        state_difference_crossings=crossings,selected_events=events,
        interpretation=dict(iteration_cap_exhaustion_not_observed=True,
            large_first_state_branch_follows_loaded_contact_manifold_difference=True,
            tighter_constraint_or_collision_convergence_not_tested=True,
            continuum_contact_physics_or_solver_independence_not_established=True,
            solver_final_stop_reason_not_public=True,
            global_trace_scaled_residual_is_not_the_island_solver_stopping_field=True))
    write(D/'analysis.json',result)
    np.savez_compressed(D/'state_difference_analysis.npz',time=time,qpos_max_difference=qpos,qvel_max_difference=qvel,
        nominal_ncon=np.array([x['ncon'] for x in a]),head_ncon=np.array([x['ncon'] for x in b]),
        nominal_qacc=np.array([x['qacc'] for x in a]),head_qacc=np.array([x['qacc'] for x in b]))
    print(json.dumps(dict(status='complete',first_contact_difference_step=first+1,
        first_contact_difference_time=float(time[first]),head_contact_separation=separation,
        final_qpos_difference=float(qpos[-1]),final_qvel_difference=float(qvel[-1]))))


if __name__=='__main__':main()
