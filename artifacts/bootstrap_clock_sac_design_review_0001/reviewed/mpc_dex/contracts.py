"""Bind artifacts to actual simulation assets and the observation/action contract."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .env import ASSET_ROOT, JOINT_NAMES, EnvConfig, make_model


def canonical(value):
    return json.loads(json.dumps(value,sort_keys=True,allow_nan=False))


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):digest.update(chunk)
    return digest.hexdigest()


def environment_contract(config: EnvConfig):
    model=make_model(config)
    parameters=asdict(config)
    grasp=parameters.pop('initial_grasp')
    # Render resolution and file locations do not affect the physics contract.
    parameters.pop('render_width');parameters.pop('render_height')
    return canonical({'schema_version':1,'environment':parameters,
        'grasp_sha256':sha256(grasp) if grasp else None,
        'robot_xml_sha256':sha256(ASSET_ROOT/'right_hand.xml'),
        'robot_mesh_sha256':{p.name:sha256(p) for p in sorted((ASSET_ROOT/'assets').glob('*.stl'))},
        'environment_source_sha256':sha256(Path(__file__).with_name('env.py')),
        'joint_order':JOINT_NAMES,'action_low':model.actuator_ctrlrange[:,0].tolist(),
        'action_high':model.actuator_ctrlrange[:,1].tolist(),
        'observation':'q16,object_position_in_palm3,object_quaternion_in_palm_wxyz4,principal_angle1,previous_angle1,finite_difference_qvel16,finite_difference_object_vel3,previous_normalized_action16; reorientation appends goal1',
        'target':'local-simulation'})


def metadata_path(checkpoint):
    p=Path(checkpoint)
    return p.with_name(p.name+'.metadata.json')


def write_checkpoint_metadata(checkpoint, config, learner_config, *, stage, offline_replay_sha256=None, policy_interface=None):
    metadata={'schema_version':1,'implementation':'independent-paper-reproduction',
        'stage':stage,'environment_contract':environment_contract(config),
        'learner_config':canonical(asdict(learner_config)),
        'checkpoint_sha256':sha256(checkpoint),'offline_replay_sha256':offline_replay_sha256,
        'policy_interface':policy_interface or {'kind':'absolute_joint_target'}}
    path=metadata_path(checkpoint);temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n');temp.replace(path)
    return metadata


def validate_checkpoint(checkpoint, config):
    path=metadata_path(checkpoint)
    if not path.is_file():raise ValueError(f'Checkpoint is missing its environment contract: {path}')
    metadata=json.loads(path.read_text())
    if metadata.get('schema_version')!=1:raise ValueError('Unsupported checkpoint metadata schema')
    if metadata.get('checkpoint_sha256')!=sha256(checkpoint):raise ValueError('Checkpoint hash mismatch')
    if metadata.get('environment_contract')!=environment_contract(config):
        raise ValueError('Checkpoint environment/action contract mismatch; geometry transfer requires an explicit adaptation path')
    return metadata
