"""Independently replay and audit diagnostic absolute-target controls.

This does not promote partial or exploratory data into production primitives.
Optional hold commands test what happens after the recorded motion stops.
"""
import argparse
import json
import shutil
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
from mpc_dex.cli import read_config
from mpc_dex.contracts import environment_contract,sha256
from mpc_dex.env import AllegroRotationEnv
from scripts.audit_policy import PHYSICS_COLUMNS, audit_episode


class RecordedPolicy:
    def __init__(self,actions):
        self.actions=actions
        self.index=0

    def act(self,observation,deterministic=True):
        action=self.actions[self.index]
        self.index+=1
        return action


def audit(source,config,output,hold_steps=0):
    source,output=Path(source),Path(output)
    if output.exists():raise FileExistsError(output)
    if hold_steps<0:raise ValueError('Hold length must be nonnegative')
    cfg,ec=read_config(config)
    with np.load(source,allow_pickle=False) as archive:
        actions=archive['actions'].copy()
    if actions.ndim!=2 or actions.shape[1]!=16 or len(actions)==0:
        raise ValueError('Expected nonempty absolute normalized controls')
    recorded=len(actions)
    if hold_steps:
        actions=np.vstack((actions,np.repeat(actions[-1:],hold_steps,axis=0)))
    output.mkdir(parents=True)
    sources=output/'source';sources.mkdir()
    for relative in ['scripts/audit_recorded_controls.py','scripts/audit_policy.py','mpc_dex/env.py']:
        path=ROOT/relative;shutil.copyfile(path,sources/path.name)
    shutil.copyfile(config,sources/'config.json')
    env=AllegroRotationEnv(ec)
    try:
        report,physics,commands=audit_episode(env,RecordedPolicy(actions),0,len(actions))
        report.update(source=str(source.resolve()),source_sha256=sha256(source),
                      interface='absolute_normalized_joint_target',
                      recorded_commands=recorded,requested_hold_commands=hold_steps,
                      environment_contract=environment_contract(ec),config_sha256=sha256(config),
                      audit_source_sha256=sha256(__file__),observer_source_sha256=sha256(ROOT/'scripts/audit_policy.py'),
                      production_eligible=False,hardware_verified=False)
        np.savez_compressed(output/'physics.npz',samples=physics,columns=np.asarray(PHYSICS_COLUMNS))
        (output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        return report
    finally:env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--config',default=str(ROOT/'configs/local_sim.json'))
    parser.add_argument('--hold-steps',type=int,default=0)
    args=parser.parse_args()
    print(json.dumps(audit(args.source,args.config,args.output,args.hold_steps),indent=2))
