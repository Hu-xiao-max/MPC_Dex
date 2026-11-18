"""Append measured decision time to the frozen73-pair diagnostic, no relabeling."""
from pathlib import Path
import hashlib,json,shutil
import numpy as np
D=Path(__file__).resolve().parent
OLD=D.parent/'bc_common_noise_corrective_0001'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def main():
    manifest=json.loads((OLD/'manifest.json').read_text())
    assert all(sha(OLD/name)==digest for name,digest in manifest.items())
    for name in ['source','inputs']:
        shutil.copytree(OLD/name,D/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ['supervised_pairs.npz','row_provenance.json','dataset_validation.json','protocol.json',
                 'manifest.json','summary.json','startup_fit_diagnosis.json','findings.md','train.py','audit.py']:
        dest=D/'inputs/previous73'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(OLD/name,dest)
        assert sha(OLD/name)==sha(dest)
    shutil.copyfile(OLD/'row_provenance.json',D/'row_provenance.json')
    config=json.loads((OLD/'strict_config.json').read_text())
    config['environment']['initial_grasp']=str(D/'source/assets/initial_grasp_40mm.npz')
    write(D/'strict_config.json',config)
    with np.load(D/'inputs/previous73/supervised_pairs.npz') as z:data={k:v.copy() for k,v in z.items()}
    rows=json.loads((D/'row_provenance.json').read_text())['rows'];elapsed=[];checks=[]
    dt=config['environment']['control_seconds'];substeps=round(dt/config['environment']['timestep'])
    for row in rows:
        path=D/row['source_path'];assert sha(path)==row['source_sha256']
        index=row['source_row_one_based']-1
        with np.load(path) as z:
            np.testing.assert_array_equal(data['obs'][row['dataset_row_one_based']-1],z['obs'][index])
            np.testing.assert_array_equal(data['actions'][row['dataset_row_one_based']-1],z['actions'][index])
            initial_time=float(z['initial_time']) if 'initial_time' in z else 0.
            if 'control_times' in z:
                measured=float(z['control_times'][index]);method='saved reset-relative measured control_times'
            else:
                with np.load(path.parent/'physics.npz') as physics:
                    columns=physics['columns'].tolist();times=physics['samples'][:,columns.index('time')]
                assert np.isclose(times[0],config['environment']['timestep'],rtol=0,atol=1e-12)
                measured=float(times[index*substeps-1]) if index else float(times[0]-config['environment']['timestep'])
                method='actual preceding2ms physics boundary; initialzero verified fromfirstphysics timestamp'
        nominal=index*dt
        assert np.isclose(measured,nominal,rtol=0,atol=1e-12)
        assert np.float32(measured)==np.float32(nominal)
        elapsed.append(nominal)
        checks.append(dict(dataset_row_one_based=row['dataset_row_one_based'],source_row_one_based=index+1,
            measured_elapsed_seconds=measured,control_index_elapsed_seconds=nominal,
            appended_float32_seconds=float(np.float32(nominal)),initial_time=initial_time,measurement=method))
    elapsed=np.asarray(elapsed);clock=elapsed.astype(np.float32)
    obs=np.column_stack([data['obs'],clock]);assert obs.shape==(73,61)
    np.testing.assert_array_equal(obs[:,:60],data['obs'])
    assert np.array_equal(np.clip(data['actions'],-.99,.99),data['actions'])
    np.savez_compressed(D/'supervised_pairs.npz',**dict(data,obs=obs),elapsed_seconds=elapsed)
    write(D/'measured_clock_validation.json',dict(status='complete',rows=checks,
        max_elapsed_error_seconds=max(abs(x['measured_elapsed_seconds']-x['control_index_elapsed_seconds']) for x in checks),
        appended_float32_equals_measured_float32=True,no_interpolated_states_or_actions=True))
    interface=dict(kind='diagnostic_bootstrap_elapsed_time_absolute_joint_target',observation_dimension=61,
        base_observation_dimension=60,task='rotation',appended_feature='actual reset-relative decision elapsed seconds',
        clock_dtype='float32',goal_feature=False,main_policy_compatible=False,production_eligible=False)
    write(D/'policy_interface.json',interface)
    previous=json.loads((OLD/'dataset_validation.json').read_text())
    validation=dict(status='complete',pairs=73,original_source_dataset_sha256=sha(OLD/'supervised_pairs.npz'),
        source_dataset_sha256=sha(D/'supervised_pairs.npz'),source_manifest_sha256=sha(OLD/'manifest.json'),
        same_original60_observations=True,all_actions_bitexact=True,same_rows_and_sampling_order=True,
        diagnostic_epsilon=.01,clamping_identity=True,source_prefix_rows=40,source_new_mpc_rows=21,source_common_rows=12,
        initial_hold_rows=1,clock_measurements_verified=True,source_provenance_validation=previous,
        inherited_startup_original_observation_l2=float(np.linalg.norm(obs[0,:60].astype(float)-obs[1,:60])),
        inherited_startup_augmented_observation_l2=float(np.linalg.norm(obs[0].astype(float)-obs[1])),
        production_eligible=False,policy_only_success_claim=False,main_demonstrations=0)
    write(D/'dataset_validation.json',validation)
    protocol=json.loads((OLD/'protocol.json').read_text())
    protocol.update(purpose='One artifact-local elapsed-time representation diagnostic on identical73 authentic pairs',
        policy_interface=interface,observation_dimension=61,fresh_network=True,import_60D_weights=False,
        initialization_caveat='Extra input changes architecture and RNG consumption; same seed is not a pure causal control',
        dataset_sha256=sha(D/'supervised_pairs.npz'),dataset_validation_sha256=sha(D/'dataset_validation.json'),
        source60D_summary_sha256=sha(OLD/'summary.json'))
    write(D/'protocol.json',protocol)
    assert all(sha(OLD/name)==digest for name,digest in manifest.items())
    print(json.dumps(dict(pairs=73,dimension=61,actions_bitexact=True,measured_clock_float32_exact=True,
        elapsed_error_seconds=max(abs(x['measured_elapsed_seconds']-x['control_index_elapsed_seconds']) for x in checks),
        dataset_sha256=sha(D/'supervised_pairs.npz'))))

if __name__=='__main__':main()
