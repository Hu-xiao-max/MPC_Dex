from pathlib import Path
import hashlib,json,shutil
root=Path('/home/autel001/code/mpc_dex/artifacts');old=root/'bootstrap_0005_frozen_critic_calibration_0001';new=root/'bootstrap_0005_centered_critic_calibration_0001'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old_manifest=json.loads((old/'manifest.json').read_text());assert all(sha(old/p)==h for p,h in old_manifest.items())
new.mkdir(exist_ok=False)
for name in ('source','inputs'):shutil.copytree(old/name,new/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name in ('reconstructed_preupdate_controls.npz','source_hashes.json','input_bindings.json'):shutil.copyfile(old/name,new/name);assert sha(old/name)==sha(new/name)
shutil.copyfile(old/'manifest.json',new/'original_calibration_manifest.json')
config=json.loads((old/'config.json').read_text());config['environment']['initial_grasp']=str(new/'inputs/initial_grasp.npz');(new/'config.json').write_text(json.dumps(config,indent=2)+'\n')
protocol=json.loads((old/'protocol.json').read_text());protocol.update(status='data_verified',custom_critic_calibration_updates=0,paired_original_artifact=str(old),paired_original_manifest_sha256=sha(old/'manifest.json'),preparation_reuses_verified_original_data=True,new_physics_controls=0)
for key in ('actor_trunk_mean_std_alpha_bitexact','calibrated_checkpoint_sha256'):protocol.pop(key,None)
(new/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
source=(old/'calibrate.py').read_text().replace("OUT=ROOT/'artifacts/bootstrap_0005_frozen_critic_calibration_0001'","OUT=ROOT/'artifacts/bootstrap_0005_centered_critic_calibration_0001'")
needle=" actor_before={k:v.detach().clone() for k,v in agent.actor.state_dict().items()};alpha_before=agent.log_alpha.detach().clone()"
replacement=""" center=float(np.mean(data['empirical_soft_q_targets'][indices]))
 with torch.no_grad():
  for family in ('critics','target_critics'):
   for network in getattr(agent,family).networks:network[1].bias.add_(center)
 agent.save(OUT/'centered_initial.pt')
 write_checkpoint_metadata(OUT/'centered_initial.pt',ec,agent.config,stage='diagnostic-train-target-centered-critic-initialization',policy_interface={'kind':'absolute_joint_target'})
 protocol['training_target_mean_added_to_output_bias']=center
 protocol['center_uses_heldout_data']=False
 write(OUT/'centering.json',dict(empirical_training_target_mean=center,initial_checkpoint_sha256=sha(OUT/'inputs/initial_agent.pt'),centered_initial_sha256=sha(OUT/'centered_initial.pt'),changed_parameters=['critics.networks.0.1.bias','critics.networks.1.1.bias','target_critics.networks.0.1.bias','target_critics.networks.1.1.bias']))
"""+needle
assert needle in source;source=source.replace(needle,replacement)
source=source.replace("rng=np.random.default_rng(protocol['fit_numpy_seed']);metrics=[]", "rng=np.random.default_rng(protocol['fit_numpy_seed']);metrics=[];batch_indices=[]")
source=source.replace("chosen=rng.integers(len(indices),size=256);selection=", "chosen=rng.integers(len(indices),size=256);batch_indices.append(chosen.copy());selection=")
source=source.replace("after=assess();write(OUT/'assessment_after.json',after);write(OUT/'training_metrics.json',metrics)","after=assess();write(OUT/'assessment_after.json',after);write(OUT/'training_metrics.json',metrics);np.save(OUT/'minibatch_indices.npy',np.asarray(batch_indices))")
source=source.replace("stage='diagnostic-frozen-policy-empirical-return-critic-calibration'","stage='diagnostic-frozen-policy-centered-empirical-return-critic-calibration'")
source=source.replace("reexecuted_source_controls_for_integrity=256","reexecuted_source_controls_for_integrity=0,original_verified_reexecution_controls=256")
(new/'calibrate.py').write_text(source);Path('/tmp/frozen_critic_calibration_centered.py').write_text(source)
shutil.copyfile(__file__,new/'prepare_paired.py')
m={str(p.relative_to(new)):sha(p) for p in sorted(new.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p!=new/'manifest.json'}
(new/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
print('prepared centered pair',len(m))
