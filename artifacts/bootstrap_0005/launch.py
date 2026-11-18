"""Run the fixed 2000-interaction protocol with immutable committed snapshots.

Only orchestration and archival I/O are local here. The frozen, reviewed
BootstrapRunner and all numerical/environment methods are used unchanged.
"""
from pathlib import Path
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json, os, shutil, signal, sys, threading, time, traceback

D=Path(__file__).resolve().parent
sys.path.insert(0,str(D/'source'))
import numpy as np
import torch
import mujoco
from mpc_dex.learning import LearnerConfig
from mpc_dex.contracts import sha256,validate_checkpoint,metadata_path
from scripts.train_bootstrap import BootstrapRunner,parser,read_environment,inference_runtime,load_bootstrap_seed,write_json

ARGV=['--config',str(D/'config.json'),'--output',str(D/'run'),
      '--initial-actor',str(D/'inputs/initial_actor.pt'),
      '--steps','2000','--device','cuda:0','--num-envs','1','--seed','17031',
      '--batch-size','256','--updates-per-step','1','--learning-starts','256',
      '--eval-every','500','--checkpoint-every','500','--log-every','100',
      '--replay-capacity','1000000','--torch-threads','1']


def archive_current():
    current=D/'run/latest.json'
    if not current.exists():return None
    committed=json.loads(current.read_text())
    name=committed['checkpoint_directory']
    source=D/'run'/name
    target=D/'checkpoints'/name
    if target.exists():return target
    pending=target.with_name(name+'.copying')
    shutil.copytree(source,pending)
    state=json.loads((pending/'state.json').read_text())
    assert 0 <= state['environment_steps'] <= 2000
    metadata=json.loads(metadata_path(pending/'agent.pt').read_text())
    assert metadata['checkpoint_sha256']==sha256(pending/'agent.pt')
    identities={str(p.relative_to(pending)):sha256(p) for p in sorted(pending.rglob('*')) if p.is_file()}
    for rel,digest in identities.items():
        assert sha256(source/rel)==digest,(name,rel)
    proof=dict(status='complete',source_generation=name,environment_steps=state['environment_steps'],
        state_status=state['status'],online_updates=state['online_updates'],
        bc_updates=state['bc_updates'],pretrain_updates=state['pretrain_updates'],
        checkpoint_sha256=metadata['checkpoint_sha256'],file_hashes=identities,
        copied_utc=datetime.now(timezone.utc).isoformat())
    write_json(pending/'archive_manifest.json',proof)
    pending.replace(target)
    print(json.dumps(dict(event='immutable_checkpoint_ready',directory=str(target),**{
        k:proof[k] for k in ['environment_steps','online_updates','checkpoint_sha256','state_status']})),flush=True)
    return target


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available()
    args=parser().parse_args(ARGV)
    config,ec=read_environment(args.config)
    source=validate_checkpoint(args.initial_actor,ec)
    assert source['policy_interface']=={'kind':'absolute_joint_target'}
    seed,metadata,episodes=load_bootstrap_seed(D/'inputs/replay.npz',ec,source['policy_interface'])
    assert len(seed)==9 and len(episodes)==1 and metadata['demonstration_successes']==1
    defaults=dict(ensemble_size=2,online_target_critics=2,actor_warmup_updates=0,critic_lr=3e-4,initial_std=.05)
    defaults.update(config['bootstrap']['learner'])
    learner=LearnerConfig(**defaults)
    runtime=dict(pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat(),executable=sys.executable,
        inference_runtime=inference_runtime(args.device),mujoco_version=mujoco.__version__,numpy_version=np.__version__,
        packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']},
        shell_command='.venv/bin/python artifacts/bootstrap_0005/launch.py',trainer_arguments=ARGV,
        launcher_sha256=sha256(__file__),resolved_learner=asdict(learner),environment=asdict(ec),
        immutable_input_actor_sha256=sha256(args.initial_actor),immutable_seed_sha256=sha256(D/'inputs/replay.npz'),
        no_automatic_extension=True,maximum_online_interactions=2000,
        evaluation_note='Five repeated cold-reset deterministic episodes, not an independent robustness-rate estimate',
        success_provenance_note='One seed MPC demo is separate; any completed policy episode with online_updates=0 is BC-origin and ineligible for SAC primitives')
    write_json(D/'runtime.json',runtime)
    inputs={str(p.relative_to(D)):sha256(p) for p in sorted(D.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p.name not in ('training.log','inputs_manifest.json')}
    write_json(D/'inputs_manifest.json',inputs)
    (D/'checkpoints').mkdir()
    runner=BootstrapRunner(args)
    assert runner.agent.bc_updates==12000 and runner.agent.pretrain_updates==runner.agent.online_updates==0
    assert runner.successes==runner.eval_successes==runner.environment_steps==0
    assert runner.demonstration_successes==1
    archive_current()  # Initial generation is immutable before the first interaction.
    errors=[];finished=threading.Event()

    def archive_watcher():
        while not finished.wait(.05):
            try:archive_current()
            except BaseException:
                errors.append(traceback.format_exc());runner.stop_requested=True
                return

    watcher=threading.Thread(target=archive_watcher,name='committed-checkpoint-archive',daemon=True)
    watcher.start()
    old={}

    def request_stop(signum,frame):runner.stop_requested=True

    for sig in [signal.SIGINT,signal.SIGTERM]:old[sig]=signal.signal(sig,request_stop)
    code=None
    try:
        code=runner.run()
    finally:
        finished.set();watcher.join()
        for sig,handler in old.items():signal.signal(sig,handler)
        archive_current()
        write_json(D/'execution_result.json',dict(exit_code=code,archive_errors=errors,
            environment_steps=runner.environment_steps,online_updates=runner.agent.online_updates,
            bc_updates=runner.agent.bc_updates,pretrain_updates=runner.agent.pretrain_updates,
            demonstration_successes=runner.demonstration_successes,
            online_episode_successes=runner.successes,evaluation_successes=runner.eval_successes,
            main_stage_updates=0,maximum_online_interactions=2000,
            stopped_at_budget_or_fault=True,utc=datetime.now(timezone.utc).isoformat()))
    assert not errors,errors
    assert code==0 and runner.environment_steps==2000,(code,runner.environment_steps)
    assert all(sha256(D/path)==digest for path,digest in inputs.items()),'Archived inputs changed during training'
    for step in [0,500,1000,1500,2000]:
        assert any(json.loads((p/'state.json').read_text())['environment_steps']==step
                   for p in (D/'checkpoints').iterdir() if p.is_dir() and not p.name.endswith('.copying'))
    return code


if __name__=='__main__':raise SystemExit(main())
