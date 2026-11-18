from pathlib import Path
import hashlib,json,shutil
root=Path('/home/autel001/code/mpc_dex');out=root/'artifacts/bootstrap_0005_next_step_review_0001';run=root/'artifacts/bootstrap_0005'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
g=json.loads((out/'gradient_analysis.json').read_text())
for p,h in g['input_sha256'].items():assert sha(p)==h
(out/'sources').mkdir()
for source,name in [(run/'source/mpc_dex/learning.py','learning.py'),(run/'source/scripts/train_bootstrap.py','train_bootstrap.py'),(run/'source/mpc_dex/replay.py','replay.py'),(run/'config.json','run_config.json'),(run/'run/metrics.jsonl','metrics.jsonl'),(root/'artifacts/sources/project.html','main_project.html'),(root/'artifacts/sources/planner-project.html','companion_project.html')]:shutil.copyfile(source,out/'sources'/name)
shutil.copyfile('/tmp/bootstrap0005_loss_review.log',out/'analysis.log')
shutil.copyfile(__file__,out/'finalize.py')
text='''# Bootstrap0005: preserve the actor while testing critic learning

No concrete sign, terminal-mask, action-density, or warmup-implementation bug was found in the reviewed update path. The observed loss of the BC skill is consistent with the actual objective and schedule. The strongest next experiment is a **true frozen-actor, fixed-temperature critic-only phase**, not another learning-rate warmup or noise sweep. Nothing was trained or changed during this review.

## Evidence and mechanisms

1. **Warmup permits updates immediately.** `learning.py:269–276` sets actor LR to `3e-4 * min((online_updates+1)/5000,1)` and always calls the actor optimizer. It updates the shared trunk, mean head, and std head. The first LR is 6e-8; at update245 it is1.47e-5, and the deterministic policy already fails at the500-interaction checkpoint. By update1745, LR is1.047e-4 and the sum of ramp fractions equals304.677 full-rate updates. This equivalence describes cumulative LR, not identical Adam dynamics. Temperature updates are never warmed up. No freeze was implemented or promised by this parameter.

2. **The entropy objective strongly increases small initial std.** A read-only autograd decomposition on actual recorded observations found initial std-head entropy-gradient norm0.75131 versus Q-gradient norm0.002515, about299×. Every entropy gradient on std-head bias is negative, approximately−0.05, so gradient descent increases log std. This follows the correct reparameterized Gaussian entropy gradient; negative differential entropy is not itself an error. On the inspected initial batch, entropy is−40.20 against target−16. The temperature objective therefore increases alpha, as logged from0.05 to0.05803. At the end, cold-reset std is0.05569–0.36185 and fixed-reference std reaches0.59550. No claim is made that entropy alone caused all mean drift.

3. **Fresh and imperfect critics immediately influence the imported mean.** There is no BC loss, policy-distance penalty, frozen trunk, or demonstrated-skill constraint in `SACAgent.update`. At initialization, the inspected mean-head Q-gradient norm is0.776 versus entropy-gradient0.612. At the1,000- and2,000-interaction snapshots, Q-gradient norms are13.49× and21.01× the corresponding entropy-gradient norms on the same fixed observation batch. These are fresh frozen-loss probes, not gradients recorded during the historical optimizer steps. Independently matched-action rollouts show positive initial-state Q values despite mostly negative realized soft returns; the mean critic bias is about+68 to+75 across checkpoints. Value bias does not prove the action gradient's direction, but it does not justify trusting that gradient to preserve the skill.

4. **The seed does not anchor the actor.** The nine genuine MPC transitions enter ordinary uniform replay, with BC0 and critic-pretrain0. Their expected share is9/265=3.40% at the first update and9/2009=0.448% at the end: about8.69 and1.15 rows per256-row batch. Critic training uses the recorded actions; the actor instead samples its own actions on those observations. Thus replaying a demonstration row does not instruct the actor to reproduce its action. The reconstructed first observation batch contains nine seed rows. No new successful training trajectories replenish this coverage.

5. **The original skill has a narrow tested tolerance.** The unchanged deterministic mean succeeds, but all five prescribed stochastic samples fail at both .05 and the existing1e-5 minimum. The same-noise open-loop counterfactual avoids immediate palm/drop but completes no full turn in nine controls. Those diagnostics limit what a small LR alone can protect; they are samples, not a robustness rate or proof that stochastic learning is impossible.

The inspected target is `reward + gamma*(min(target_Q)-alpha*logpi)` only for nonterminal rows; the actor minimizes `alpha*logpi - mean(Q)`. The squashed-Gaussian Jacobian and temperature sign are consistent with this objective. All18 calibration episodes truly terminated, so omitted horizon tails do not explain their critic discrepancy. The update's actor-critic ensemble mean is an explicit local choice, not a newly discovered coding error.

## One proposed bounded learning experiment

Start again from the exact successful imported mean and fresh two critics, with the same strict40mm environment, nine-row seed, initial std.05, alpha.05, interaction seed17031, batch256, learning_starts256, UTD1, and2000-interaction cap. **Freeze every actor tensor, including the shared trunk and both heads, and freeze alpha for the entire run.** Train only the critic optimizer and Polyak targets using the current SAC Bellman backup and actual environment transitions. This requires an explicit critic-only path: setting a tiny actor LR, calling the current `update`, or merely freezing the mean head is insufficient. Do not automatically unfreeze afterward.

Record `critic_only_updates` separately: expected1745; actor/temperature updates remain0. Assert exact actor and alpha hashes at each checkpoint and reproduce the original deterministic nine-control success. The phase must not create SAC-actor success/primitive eligibility from its unchanged BC policy. Preserve the complete online replay and real episode boundaries.

Use the same held-out seeds17041–17045 before and after to compare Q on exactly matched first actions against the same frozen-policy soft returns. Freeze means/std/alpha so this is a controlled calibration comparison, unlike bootstrap0005's changing policies. Inspect both critics, terminal calibration error, and a separate BC-mean action check; do not use falling TD loss or critic agreement alone as the release criterion. The result can show whether real critic learning becomes trustworthy while preserving the demonstrated mean. It does not promise exploration support or a successful stochastic policy; those remain unresolved. Stop at2000 without extending or enabling actor updates.

## Source fidelity and limitations

The companion publishes a simulation-trained SAC primitive policy on the40mm AF hexagon without domain randomization, with five trajectories and an ambiguous“~1min” extraction duration. It does not publish this bootstrap optimizer, entropy, warmup, replay, or initialization recipe (`companion_project.html`, Implementation Details → Rotation primitive source; companion paper III-B1, `S3.SS2.SSS1.p3.1`). The proposed freeze phase is a disclosed local diagnostic, not an author requirement.

The main real-world method separately specifies a200k linear actor warmup, ten critics,50k frozen-actor Soft-SARSA updates,30% offline replay, and MPC scaffolding (`main_project.html:260–288`, Real-world SAC with MPC scaffolding). Bootstrap0005 used5000 warmup, two fresh critics, no critic pretraining, one uniformly mixed seed episode, and no MPC assistance. These are explicit local prerequisite choices; importing the main recipe wholesale would not establish companion fidelity. The current main-stage gates remain unchanged.

`gradient_analysis.json` records all frozen probes, gradients, input hashes, and the exact observation batch reconstructed from saved runner/replay RNG and actual data. `analyze_frozen_losses.py` performs autograd only: no optimizer step, environment execution, or source mutation. Cached primary tables and reviewed numerical files are copied under `sources/`.
'''
(out/'findings.md').write_text(text)
recommendation=dict(status='complete',code_bug_found=False,no_training_or_source_changes=True,
 proposed_experiment='Frozen actor and fixed alpha;2000 actual strict interactions, fresh critics only,1745 expected critic-only updates; no automatic unfreeze',
 method_status='Explicit local bootstrap diagnostic, not a published companion recipe',
 inputs_sha256=g['input_sha256'],gradient_probe_seed=g['gradient_seed'],source_hashes={str(p.relative_to(out)):sha(p) for p in sorted((out/'sources').rglob('*')) if p.is_file()})
(out/'summary.json').write_text(json.dumps(recommendation,indent=2)+'\n')
manifest={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
assert all(sha(out/p)==h for p,h in manifest.items())
print(len(manifest),sha(out/'manifest.json'))
