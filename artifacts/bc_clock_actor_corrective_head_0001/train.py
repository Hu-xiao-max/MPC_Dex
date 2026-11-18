"""One fresh61-input elapsed-time BC diagnostic on86 unchanged corrective labels."""
from pathlib import Path
from dataclasses import asdict, replace
import hashlib, json, sys, time
D=Path(__file__).resolve().parent
sys.path.insert(0,str(D/'source'))
import numpy as np
import torch
from mpc_dex.env import EnvConfig
from mpc_dex.learning import SACAgent,LearnerConfig
from mpc_dex.contracts import sha256,write_checkpoint_metadata
from scripts.audit_policy import write_json
from audit import run_audit,full_turn_events


def module_hash(module):
    digest=hashlib.sha256()
    for key,tensor in sorted(module.state_dict().items()):
        digest.update(key.encode());digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def invariant_hashes(agent):
    return dict(critics=module_hash(agent.critics),target_critics=module_hash(agent.target_critics),
        log_alpha=hashlib.sha256(agent.log_alpha.detach().cpu().numpy().tobytes()).hexdigest())


@torch.no_grad()
def singleton_values(agent,observations):
    features=[];means=[];predictions=[]
    for observation in observations:
        feature=agent.actor.trunk(torch.as_tensor(observation,device=agent.device,dtype=torch.float32))
        features.append(feature.cpu().numpy())
        means.append(agent.actor.mean_head(feature).cpu().numpy())
        predictions.append(agent.act(observation,deterministic=True))
    return tuple(np.asarray(x) for x in (features,means,predictions))


def grouped_metrics(means,predictions,targets,actions):
    groups={}
    for name,s in [('all86',slice(None)),('original_common12',slice(0,12)),
                   ('original_teacher_prefix6',slice(0,6)),('original_mpc6',slice(6,12)),
                   ('prior_noisy_prefix40',GROUP_MASKS['prefix']),('prior_mpc_corrections21',GROUP_MASKS['mpc']),
                   ('new_actor_prefix8',GROUP_MASKS['actor_prefix']),('new_actor_state_mpc5',GROUP_MASKS['actor_mpc'])]:
        delta=predictions[s].astype(np.float64)-actions[s].astype(np.float64)
        latent=means[s].astype(np.float64)-targets[s].astype(np.float64)
        groups[name]=dict(action_mse=float(np.mean(delta**2)),action_max_abs_error=float(abs(delta).max()),
            action_max_abs_error_by_row=abs(delta).max(1).tolist(),latent_mse=float(np.mean(latent**2)),
            latent_max_abs_error=float(abs(latent).max()))
    return groups


def evaluate(agent,ec,out,kind):
    result,executed,_=run_audit(agent,ec,out,100)
    result.update(policy_kind=kind,full_turn_events=full_turn_events(out,ec))
    write_json(out/'report.json',result)
    data=dict(strict=result)
    return data


def main():
    global GROUP_MASKS
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available()
    inputs=json.loads((D/'inputs_manifest.json').read_text())
    assert all(sha256(D/p)==v for p,v in inputs.items())
    validation=json.loads((D/'dataset_validation.json').read_text())
    assert validation['status']=='complete' and validation['pairs']==86
    ec=EnvConfig(**json.loads((D/'strict_config.json').read_text())['environment'])
    with np.load(D/'supervised_pairs.npz') as a:
        obs,actions=a['obs'].copy(),a['actions'].copy()
        origins=a['origin'].copy()
    GROUP_MASKS={'prefix':origins=='recorded_fixed_mean_noisy_prefix','mpc':origins=='actual_corrected_native_mpc',
        'actor_prefix':origins=='recorded_actual_clock_actor_prefix','actor_mpc':origins=='actual_clock_actor_state_mpc_correction'}
    assert GROUP_MASKS['prefix'].sum()==40 and GROUP_MASKS['mpc'].sum()==21
    assert GROUP_MASKS['actor_prefix'].sum()==8 and GROUP_MASKS['actor_mpc'].sum()==5
    assert obs.shape==(86,61) and actions.shape==(86,16)
    learner=LearnerConfig(hidden_sizes=(256,256),ensemble_size=2,online_target_critics=2,bc_action_epsilon=.01)
    assert learner.bc_lr==.003 and learner.bc_action_epsilon==.01
    torch.manual_seed(17001);rng=np.random.default_rng(17001)
    agent=SACAgent(61,16,learner,device='cuda')
    previous=torch.load(D/'inputs/clock73/bc/initial_fresh.pt',map_location='cpu',weights_only=True)
    for group in ['actor','critics','target_critics']:
        assert all(torch.equal(v,getattr(agent,group).state_dict()[k].detach().cpu()) for k,v in previous[group].items())
    assert torch.equal(torch.as_tensor(actions),torch.as_tensor(actions).clamp(-.99,.99))
    out=D/'bc';out.mkdir()
    agent.save(out/'initial_fresh.pt')
    frozen=invariant_hashes(agent)
    targets=torch.atanh(torch.as_tensor(actions,device=agent.device).clamp(-.99,.99)).cpu().numpy()
    report=dict(status='running',purpose='Fresh61-input clock diagnostic on86 unchanged pairs; no SAC/primitive or60D interface claim',
        production_eligible=False,learner_config=asdict(learner),source_script_sha256=sha256(__file__),
        policy_interface=json.loads((D/'policy_interface.json').read_text()),
        input_data_sha256=sha256(D/'supervised_pairs.npz'),initial_checkpoint_sha256=sha256(out/'initial_fresh.pt'),
        fresh_initialization_bitexact_to_prior61D=True,
        source_settings_sha256=sha256(D/'source/project.html'),invariant_initial=frozen,
        settings=dict(fresh_actor_and_optimizers=True,seed=17001,updates=12000,lr=.003,epsilon=.01,source_main_epsilon=.02,batch_size=256,
            mean_only=True,sampling='Uniform over all86 pairs with replacement',evaluation_updates=[3000,6000,12000],
            evaluation_seed=42,evaluation_max_controls=100,std_after_evaluation_checkpoint=.3,
            device='cuda',torch_threads=1,interop_threads=1,tf32=False,
            diagnostic_unused_critics=2,paper_critics=10,critic_updates=0,online_updates=0),results=[])
    write_json(out/'report.json',report)
    batches=np.empty((12000,256),dtype=np.uint8);start=time.monotonic()
    for update in range(1,12001):
        indices=rng.integers(len(obs),size=256);batches[update-1]=indices
        training=agent.bc_step(dict(obs=obs[indices],actions=actions[indices]))
        if update%250==0:
            item=dict(update=update,**training,wall_seconds=time.monotonic()-start)
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')
            print(json.dumps(item),flush=True)
        if update not in report['settings']['evaluation_updates']:continue
        agent.initialize_std();checkpoint=out/f'checkpoint_{update:05d}.pt';agent.save(checkpoint)
        metadata=write_checkpoint_metadata(checkpoint,ec,learner,stage='diagnostic-clock-actor-corrective86-bc-only',
            offline_replay_sha256=report['input_data_sha256'],
            policy_interface=report['policy_interface'])
        metadata.update(production_eligible=False,actor_kind='Fresh61D clockBC on86 authentic pairs; epsilon.01 preserveslabels; noSAC, incompatible main60D interface')
        write_json(str(checkpoint)+'.metadata.json',metadata)
        features,means,predictions=singleton_values(agent,obs)
        fit=grouped_metrics(means,predictions,targets,actions)
        np.savez_compressed(out/f'fit_{update:05d}.npz',observations=obs,actions=actions,
            features=features,latent_targets=targets,means=means,predicted_actions=predictions)
        evaluation=evaluate(agent,ec,out/f'audit_{update:05d}','diagnostic-clock-actor-corrective86-bc-only')
        row=dict(updates=update,training=training,fit=fit,**evaluation,checkpoint_sha256=sha256(checkpoint))
        report['results'].append(row)
        assert invariant_hashes(agent)==frozen and agent.pretrain_updates==agent.online_updates==0
        write_json(out/'report.json',report)
        print(json.dumps(dict(updates=update,fit=fit,evaluation=evaluation)),flush=True)
    np.savez_compressed(out/'sampled_batch_indices.npz',indices=batches)
    assert all(sha256(D/p)==v for p,v in inputs.items())
    report.update(status='complete',wall_seconds=time.monotonic()-start,invariant_final=invariant_hashes(agent),
        inputs_unchanged=True,head_polish_executed=False,
        final_max_action_fit_error=report['results'][-1]['fit']['all86']['action_max_abs_error'])
    write_json(out/'report.json',report)


if __name__=='__main__':main()
