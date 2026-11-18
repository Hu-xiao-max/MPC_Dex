"""Freeze86 authentic61D pairs for one prescribed fresh corrective BC fit."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
D=Path(__file__).resolve().parent
OLD=D.parent/'bc_common_noise_clock_0001';NEW=D.parent/'mpc_clock_actor_recovery_0001'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def read(path):
    with np.load(path,allow_pickle=False) as z:return {k:v.copy() for k,v in z.items()}
def main():
    checks={}
    for label,source in [('previous73',OLD),('actual_recovery',NEW)]:
        manifest=json.loads((source/'manifest.json').read_text())
        assert all(sha(source/path)==digest for path,digest in manifest.items())
        checks[label]=dict(path=str(source),manifest_sha256=sha(source/'manifest.json'),files=len(manifest))
    shutil.copytree(OLD/'source',D/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(OLD/'inputs',D/'inputs',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ['manifest.json','summary.json','protocol.json','supervised_pairs.npz','row_provenance.json',
                 'dataset_validation.json','measured_clock_validation.json','bc/initial_fresh.pt','bc/report.json']:
        out=D/'inputs/clock73'/name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(OLD/name,out)
    for name in ['manifest.json','summary.json','protocol.json','config.json','actual_mpc_correction_pairs.npz',
                 'actual_mpc_correction_pairs.metadata.json','recovery/transitions.npz','recovery/control_origins.npz',
                 'recovery/physics.npz','recovery/report.json','recovery/independent_audit/transitions.npz',
                 'recovery/independent_audit/physics.npz','recovery/independent_audit/report.json']:
        out=D/'inputs/clock_recovery'/name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(NEW/name,out)
        assert sha(out)==sha(NEW/name)
    for name in ['audit.py','policy_interface.json']:shutil.copyfile(OLD/name,D/name)
    config=json.loads((OLD/'strict_config.json').read_text());config['environment']['initial_grasp']=str(D/'source/assets/initial_grasp_40mm.npz')
    write(D/'strict_config.json',config)
    old=read(D/'inputs/clock73/supervised_pairs.npz');new=read(D/'inputs/clock_recovery/recovery/transitions.npz')
    audited=read(D/'inputs/clock_recovery/recovery/independent_audit/transitions.npz')
    for key in new:np.testing.assert_array_equal(new[key],audited[key])
    physical=read(D/'inputs/clock_recovery/recovery/physics.npz')
    for key,value in physical.items():np.testing.assert_array_equal(value,read(D/'inputs/clock_recovery/recovery/independent_audit/physics.npz')[key])
    origins=read(D/'inputs/clock_recovery/recovery/control_origins.npz')
    np.testing.assert_array_equal(origins['expert_correction_mask'],np.arange(14)>=9)
    np.testing.assert_array_equal(new['obs'][:,:60],new['environment_obs'])
    np.testing.assert_array_equal(new['next_obs'][:,:60],new['environment_next_obs'])
    np.testing.assert_array_equal(new['obs'][:,60],new['decision_times'].astype(np.float32))
    np.testing.assert_array_equal(new['next_obs'][:,60],new['control_times'][1:].astype(np.float32))
    np.testing.assert_array_equal(new['decision_times'],new['control_times'][:-1])
    np.testing.assert_array_equal(new['obs'][:,60],(np.arange(14)*.1).astype(np.float32))
    np.testing.assert_array_equal(new['obs'][1:],new['next_obs'][:-1])
    np.testing.assert_array_equal(new['terminated'],np.arange(14)==13);assert not new['truncated'].any()
    np.testing.assert_array_equal(new['obs'][0],old['obs'][0]);assert not np.array_equal(new['actions'][0],old['actions'][0])
    correction=read(D/'inputs/clock_recovery/actual_mpc_correction_pairs.npz')
    for key in ['obs','actions','next_obs','rewards','qpos','qvel','integration_state']:np.testing.assert_array_equal(new[key][9:],correction[key])
    result=json.loads((D/'inputs/clock_recovery/recovery/independent_audit/report.json').read_text())
    assert result['strict_force_success'] and result['finger_supported_rotation_success'] and result['no_external_support']
    assert not result['independent_drop'] and not result['independent_palm_support_violation']
    obs=np.concatenate([old['obs'],new['obs'][1:]]);actions=np.concatenate([old['actions'],new['actions'][1:]])
    elapsed=np.r_[old['elapsed_seconds'],new['decision_times'][1:]]
    origin=np.r_[old['origin'],['recorded_actual_clock_actor_prefix']*8,['actual_clock_actor_state_mpc_correction']*5]
    assert obs.shape==(86,61) and actions.shape==(86,16)
    assert np.array_equal(np.clip(actions,-.99,.99),actions)
    rows=json.loads((OLD/'row_provenance.json').read_text())['rows']
    for k in range(1,14):
        rows.append(dict(dataset_row_one_based=len(rows)+1,source_label='actual_clock_actor_recovery',
            source_path='inputs/clock_recovery/recovery/transitions.npz',source_sha256=sha(NEW/'recovery/transitions.npz'),
            source_row_one_based=k+1,origin=str(origin[len(rows)]),recorded_original_origin=str(origins['origin'][k]),
            measured_decision_seconds=float(new['decision_times'][k])))
    np.savez_compressed(D/'supervised_pairs.npz',obs=obs,actions=actions,elapsed_seconds=elapsed,origin=origin,
        source_label=np.r_[old['source_label'],['actual_clock_actor_recovery']*13],
        source_row_one_based=np.r_[old['source_row_one_based'],np.arange(2,15)])
    write(D/'row_provenance.json',dict(rows=rows,excluded_new_rows=[dict(source_row_one_based=1,
        reason='Same initial observation; preserve the original nominal hold once, exclude slightly different learned-action label',
        action_max_difference=float(abs(new['actions'][0]-old['actions'][0]).max()))],
        unchanged_original73=True,no_relabeling=True,no_interpolation=True,no_synthetic_episode_links=True))
    distances=np.linalg.norm(obs.astype(float)[:,None]-obs.astype(float)[None,:],axis=2)
    pairs=np.array([(i,j) for i in range(86) for j in range(i+1,86)])
    duplicate=pairs[distances[pairs[:,0],pairs[:,1]]==0]
    conflicts=[(int(i),int(j)) for i,j in duplicate if not np.array_equal(actions[i],actions[j])]
    assert not len(duplicate) and not conflicts
    latent=np.arctanh(actions.astype(float))
    def pairrecord(i,j):
        delta=actions[i].astype(float)-actions[j];z=latent[i]-latent[j]
        return dict(first=int(i)+1,second=int(j)+1,observation_l2=float(distances[i,j]),
            observation_max_abs=float(abs(obs[i].astype(float)-obs[j]).max()),action_max_abs=float(abs(delta).max()),
            latent_l2=float(np.linalg.norm(z)),clock_difference=float(obs[i,60]-obs[j,60]),
            first_source=rows[i],second_source=rows[j])
    newpairs=pairs[np.any(pairs>=73,axis=1)];newpairs=newpairs[np.argsort(distances[newpairs[:,0],newpairs[:,1]])]
    mpcpairs=pairs[np.any(pairs>=81,axis=1)];mpcpairs=mpcpairs[np.argsort(distances[mpcpairs[:,0],mpcpairs[:,1]])]
    validation=dict(status='complete',pairs=86,prior73_pairs_unchanged=True,new_policy_prefix_pairs=8,new_mpc_pairs=5,
        same_reset_labels_excluded=1,source_manifests_verified=checks,all_source_transition_physics_bitexact=True,
        all_clock_columns_verified=True,clamp_epsilon=.01,clamp_changed_scalars=0,action_abs_max=float(abs(actions).max()),
        exact_duplicate_observations=0,exact_label_conflicts=0,
        nearest_new_pairs=[pairrecord(*p) for p in newpairs[:12]],
        nearest_new_mpc_pairs=[pairrecord(*p) for p in mpcpairs[:8]],
        distance_note='Raw61D observation norm mixes physical units and elapsed seconds; descriptive only',
        source_recovery_force_angle=result['final_force_qualified_angle'],source_recovery_post_success_hold=False,
        production_eligible=False,main_demonstrations=0,online_updates=0)
    write(D/'dataset_validation.json',validation)
    protocol=json.loads((OLD/'protocol.json').read_text())
    protocol.update(purpose='One fresh61D clock corrective BC fit on86 authentic pairs',
        same_previous61D_initialization_required=True,rows=86,sampling='Uniform with replacement overall86pairs',
        dataset_sha256=sha(D/'supervised_pairs.npz'),dataset_validation_sha256=sha(D/'dataset_validation.json'),
        source73_pair_summary_sha256=sha(OLD/'summary.json'),
        predeclared_optional_head_refinement=dict(method='One float64 minimum-norm residual solve with frozen finalfeatures',
            fit_error_gate=1e-6,report_to_parent_before_solve=True,extra_epochs=0,ridge_sweeps=0,scale_sweeps=0))
    write(D/'protocol.json',protocol)
    for source in [OLD,NEW]:
        manifest=json.loads((source/'manifest.json').read_text());assert all(sha(source/path)==digest for path,digest in manifest.items())
    print(json.dumps(dict(pairs=86,clamp_changes=0,duplicates=0,nearest=validation['nearest_new_pairs'][:2],
        nearest_new_mpc=validation['nearest_new_mpc_pairs'][:2],dataset_sha256=sha(D/'supervised_pairs.npz')),indent=2))
if __name__=='__main__':main()
