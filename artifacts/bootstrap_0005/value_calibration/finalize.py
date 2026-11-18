from pathlib import Path
import json,hashlib,shutil
import numpy as np
root=Path('/home/autel001/code/mpc_dex/artifacts/bootstrap_0005/value_calibration')
run=root.parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=[]
for name,gen,log in [('initial_0001','checkpoint_00000001','initial'),('mid_1000_0001','checkpoint_00000003','mid'),('final_2000_0001','checkpoint_00000006','final')]:
 d=root/name;r=json.loads((d/'report.json').read_text());bound={}
 for source,dest in [(run/'checkpoints'/gen/'archive_manifest.json','source_archive_manifest.json'),(run/'runtime.json','source_run_runtime.json'),(run/'inputs_manifest.json','source_run_inputs_manifest.json')]:
  before=sha(source);shutil.copyfile(source,d/dest);assert sha(source)==sha(d/dest)==before;bound[dest]=dict(source_path=str(source),sha256=before)
 shutil.copyfile('/tmp/bootstrap0005_value_'+log+'.log',d/'run.log')
 (d/'additional_provenance.json').write_text(json.dumps(bound,indent=2)+'\n')
 m={str(p.relative_to(d)):sha(p) for p in sorted(d.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p!=d/'manifest.json'}
 (d/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 ss=r['episodes'][1:];q=np.asarray([x['value_calibration']['critic_estimates_for_executed_initial_action'] for x in ss])
 z=np.load(d/'policy_probes.npz');base=np.load(d/'baseline_policy_probes.npz')
 row=dict(artifact=name,checkpoint_sha256=r['checkpoint_sha256'],manifest_sha256=sha(d/'manifest.json'),manifest_files=len(m),online_updates=r['runtime']['online_updates'],
  deterministic=r['episodes'][0],stochastic_successes=r['stochastic_sample_successes'],stochastic_drops=r['stochastic_sample_drops'],stochastic_palm_violations=r['stochastic_sample_palm_violations'],
  stochastic_mean_q=q.mean(axis=0).tolist(),stochastic_mean_discounted_reward=float(np.mean([x['value_calibration']['realized_discounted_reward'] for x in ss])),
  stochastic_mean_soft_q_return=float(np.mean([x['value_calibration']['realized_soft_q_return'] for x in ss])),
  stochastic_mean_critic_bias=r['mean_matched_action_critic_bias_vs_soft_q'],stochastic_max_force_qualified_angle=max(x['max_force_qualified_angle'] for x in ss),
  policy_movement=r['policy_movement'],cold_reset_mean_action_max_abs_change=float(np.max(abs(z['mean_action'][0]-base['mean_action'][0]))),
  source_success_export_review=r['runtime']['source_success_export_review'])
 rows.append(row)
summary=dict(status='complete',stage='read-only frozen checkpoint value calibration',updates_performed_by_audit=0,checkpoints=rows,
 seeds=list(range(17041,17046)),deterministic_seed=42,all_18_episodes_genuinely_terminal=True,independent_reconstruction='Every actual action, logprob, both critics, both target critics, final CPU/all-CUDA RNG, discounted return, and physics/control state match verified.',
 interpretation='Five fixed-seed stochastic diagnostics per freeze; not a robustness rate, policy-value expectation estimate, or causal identification of actor drift. No successful SAC-updated episodes in these audits or trainer exports.')
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
shutil.copyfile('/tmp/verify_frozen_value_artifacts.py',root/'verify_values.py')
shutil.copyfile('/tmp/bootstrap0005_value_verification.log',root/'verification.json')
shutil.copyfile(__file__,root/'finalize.py')
lines=['# Frozen bootstrap_0005 value calibration','',
'The imported deterministic BC mean completed a strict full turn at initialization. After 1,000 and 2,000 environment interactions, the deterministic policies failed. All fifteen prescribed stochastic audit episodes failed. No SAC-success claim is supported by these audits or the trainer exports.','',
'Each checkpoint was copied from an immutable, hash-verified committed generation. Audits used the unchanged frozen actor, both critics, learned alpha, and learned standard deviation: singleton float32 CUDA, one Torch thread, one interop thread, TF32 disabled. No parameter, optimizer, reward, action mapping, or physics setting was changed; no learning occurred in these audits.','',
'| Environment steps | Online SAC updates | Deterministic strict outcome | Stochastic outcomes, five seeds | Maximum stochastic force-qualified angle |',
'|---:|---:|---|---|---:|']
for step,row in zip((0,1000,2000),rows):
 e=row['deterministic'];outcome=('BC-origin success' if e['finger_supported_rotation_success'] else e['environment_failure_reason'])+f" at {e['steps']} controls; {e['max_force_qualified_angle']:.6f} rad max"
 lines.append(f"| {step} | {row['online_updates']} | {outcome} | {row['stochastic_drops']} drops; {row['stochastic_palm_violations']} palm violations; 0 full turns | {row['stochastic_max_force_qualified_angle']:.6f} rad |")
lines+=['',
'The five seeds are 17041–17045 at each freeze. These are fixed-seed diagnostics, not a robustness-rate estimate. Initial success has zero online updates and retains its BC origin. Trainer checkpoint state reports no successful trajectory exports at any audited freeze, so there are no exports to promote as learned SAC primitives.','',
'Critic calibration uses **the actual first action of each episode**. For frozen alpha and discount gamma=.995, the sampled soft-Q return is `sum(gamma**t * reward[t]) - alpha * sum(t>=1, gamma**t * logpi[t])`. The initial action entropy term is excluded to match the implemented SAC Q backup. Full state-value return including `-alpha*logpi[0]` is saved separately. Log probabilities come directly from `actor.sample`; there is no inverse-tanh reconstruction of rounded actions.','',
'| Environment steps | Mean initial Q1 | Mean initial Q2 | Mean discounted reward | Mean realized soft-Q return | Mean Q1 bias | Mean Q2 bias |',
'|---:|---:|---:|---:|---:|---:|---:|']
for step,row in zip((0,1000,2000),rows):
 values=[*row['stochastic_mean_q'],row['stochastic_mean_discounted_reward'],row['stochastic_mean_soft_q_return'],*row['stochastic_mean_critic_bias']]
 lines.append('| '+str(step)+' | '+' | '.join(f'{x:.6f}' for x in values)+' |')
lines+=['',
'Positive initial-state Q estimates overpredict these sampled soft returns. Bias does not grow monotonically: the five-sample means remain roughly +68 to +75. At 1,000 steps, initial Q estimates peak near +34 while sampled soft returns average −34.09; at 2,000, Q means are +14.42/+16.13 while sampled soft returns average −53.63. This differs from training-log averages over mixed replay states, which may be negative.','',
'All 18 episodes end with genuine terminal flags; there is no missing time-limit continuation here. For any future nonterminal horizon cutoff, the omitted `gamma**T * V_soft(s_T)` tail must remain explicit rather than be treated as zero. Deterministic returns are behavior diagnostics and are not stochastic-policy Q samples. Ensemble min/mean/max values are always for the same executed action; the audit performs no maximization over actions.','',
'Policy movement is evaluated on ten fixed observations: the cold reset plus the nine archived common-BC successful observations. Initialization exactly preserves the imported mean/trunk; only configured standard deviation changes from .3 to .05. Compared with the initial training checkpoint:',
'',
'| Environment steps | Cold-reset latent std range | Fixed-observation latent std range | Max normalized mean-action change | Cold-reset max mean-action change |',
'|---:|---|---|---:|---:|']
for step,row in zip((1000,2000),rows[1:]):
 m=row['policy_movement'];lines.append(f"| {step} | {m['cold_reset_std_min']:.6f}–{m['cold_reset_std_max']:.6f} | {m['latent_std_min']:.6f}–{m['latent_std_max']:.6f} | {m['mean_action_max_abs_change']:.6f} | {row['cold_reset_mean_action_max_abs_change']:.6f} |")
lines+=['',
'The configured 5,000-update actor warmup ramps the learning rate from a small positive value; it does not freeze the actor. These observations establish parameter/behavior drift, but do not isolate a single cause.','',
'Independent reconstruction from saved initial RNG states reproduces every recorded action, actual log probability, both critic/target-critic outputs, and final CPU/CUDA RNG states bit-for-bit. Discounted-return calculations were separately checked. Every control-boundary qpos/qvel/integration state exactly matches its corresponding 2 ms physics record. The initial deterministic transition archive also matches the original successful BC audit bit-for-bit. All files are retained, including checkpoint/config/source copies, runtime, observations, actions, rewards, terminal flags, means/stds/logprobs/Q estimates, complete physics, and RNG snapshots.']
(root/'findings.md').write_text('\n'.join(lines)+'\n')
manifest={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p!=root/'manifest.json'}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
assert all(sha(root/k)==v for k,v in manifest.items())
print(json.dumps({'files':len(manifest),'manifest_sha256':sha(root/'manifest.json'),'checkpoint_manifests':{x['artifact']:x['manifest_sha256'] for x in rows}},indent=2))
