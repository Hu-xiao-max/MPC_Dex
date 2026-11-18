"""The policy auditor reads physics without altering it and preserves evidence."""
import hashlib
import json
from pathlib import Path

import numpy as np
import mujoco
import pytest

from mpc_dex.bootstrap_env import BootstrapControlConfig, BootstrapRotationEnv
from mpc_dex.env import AllegroRotationEnv, EnvConfig
from scripts.audit_policy import PHYSICS_COLUMNS, audit_episode, snapshot_checkpoint

GRASP=str(Path(__file__).resolve().parents[1]/'assets/initial_grasp.npz')


@pytest.mark.parametrize('incremental',[False,True])
def test_audit_samples_every_physics_step_without_changing_contact_dynamics(incremental):
    config=EnvConfig(initial_grasp=GRASP,horizon_steps=20)
    def make():
        return (BootstrapRotationEnv(config,BootstrapControlConfig()) if incremental else AllegroRotationEnv(config))
    observed,reference=make(),make()
    class Hold:
        def act(self,obs,deterministic=True):
            return np.zeros(16,dtype=np.float32) if incremental else obs[44:60].copy()
    agent=Hold()
    try:
        result,physics,commands=audit_episode(observed,agent,7,20)
        obs,_=reference.reset(seed=7);total=0.
        for _ in range(20):
            obs,reward,done,truncated,info=reference.step(agent.act(obs));total+=reward
        np.testing.assert_array_equal(observed.data.qpos,reference.data.qpos)
        np.testing.assert_array_equal(observed.data.qvel,reference.data.qvel)
        np.testing.assert_array_equal(observed.data.ctrl,reference.data.ctrl)
        assert result['return_value']==total
        assert result['physics_steps']==20*observed.substeps==len(physics)
        assert result['truncated'] and not result['audit_limit_reached']
        assert not result['independent_rotation_success']
        assert not result['finger_supported_rotation_success']
        assert result['max_raw_angle_disagreement']<1e-12
        assert result['max_qualified_angle_disagreement']<1e-12
        assert result['max_contact_force_balance_error']<1e-10
        assert result['no_external_support']
        assert result['mean_finger_support_weight_ratio']==pytest.approx(1.,abs=.03)
        assert result['final_force_bearing_fingers']==4
        assert len(commands)==20 and len(PHYSICS_COLUMNS)==physics.shape[1]
        assert observed.finger_contacts.__func__ is AllegroRotationEnv.finger_contacts
        assert result['max_qualified_angle']>=result['final_qualified_angle']
    finally:
        observed.close();reference.close()


def test_audit_budget_is_not_an_environment_terminal_or_success():
    env=AllegroRotationEnv(EnvConfig(initial_grasp=GRASP,horizon_steps=20))
    class Hold:
        def act(self,obs,deterministic=True):return obs[44:60].copy()
    try:
        result,_,_=audit_episode(env,Hold(),0,2)
        assert result['audit_limit_reached']
        assert not result['terminated'] and not result['truncated']
        assert not result['environment_success'] and not result['independent_rotation_success']
    finally:env.close()


def test_audit_independently_rejects_palm_supported_initial_grasp(tmp_path):
    fixture=AllegroRotationEnv(EnvConfig(initial_grasp=GRASP))
    try:
        fixture.reset()
        fixture.data.qpos[fixture.object_qadr:fixture.object_qadr+3]=(
            fixture.data.xpos[fixture.palm_id] +
            fixture.data.xmat[fixture.palm_id].reshape(3,3)@np.array([.03,0.,-.01]))
        fixture.data.qvel[:]=0.
        mujoco.mj_forward(fixture.model,fixture.data)
        grasp=tmp_path/'palm_contact.npz'
        np.savez(grasp,qpos=fixture.data.qpos,qvel=fixture.data.qvel,ctrl=fixture.data.ctrl)
    finally:fixture.close()
    config=EnvConfig(initial_grasp=str(grasp),horizon_steps=20)
    observed,reference=AllegroRotationEnv(config),AllegroRotationEnv(config)
    class Hold:
        def act(self,obs,deterministic=True):return obs[44:60].copy()
    try:
        result,physics,_=audit_episode(observed,Hold(),7,20)
        obs,_=reference.reset(seed=7)
        _,reward,done,_,info=reference.step(Hold().act(obs))
        np.testing.assert_array_equal(observed.data.qpos,reference.data.qpos)
        np.testing.assert_array_equal(observed.data.qvel,reference.data.qvel)
        assert done and info['palm_support_violation'] and result['terminated']
        assert result['independent_palm_support_violation']
        assert result['initial_palm_normal_force']>1e-3
        assert result['first_palm_support_seconds']==0.
        assert result['environment_failure_reason']=='palm_support'
        assert result['return_value']==reward
        assert not result['no_external_support']
        assert not result['independent_rotation_success']
        assert not result['finger_supported_rotation_success']
        assert result['max_qualified_angle_disagreement']<1e-12
        assert len(physics)==observed.substeps
    finally:observed.close();reference.close()


def test_snapshot_remains_usable_after_training_removes_original_generation(tmp_path):
    run=tmp_path/'run';generation=run/'checkpoint_00000002';generation.mkdir(parents=True)
    payload=b'checkpoint fixture bytes';checkpoint=generation/'agent.pt';checkpoint.write_bytes(payload)
    metadata={'checkpoint_sha256':hashlib.sha256(payload).hexdigest()}
    (generation/'agent.pt.metadata.json').write_text(json.dumps(metadata))
    (run/'latest.json').write_text(json.dumps({'checkpoint_directory':generation.name,'environment_steps':5000,'status':'running'}))
    output=tmp_path/'audit';copied,record=snapshot_checkpoint(output,run=run)
    checkpoint.unlink();(generation/'agent.pt.metadata.json').unlink();generation.rmdir()
    assert copied.read_bytes()==payload
    assert json.loads((output/'checkpoint.pt.metadata.json').read_text())==metadata
    assert record['run_manifest']['environment_steps']==5000
    assert json.loads((output/'snapshot.json').read_text())==record
    with pytest.raises(FileExistsError):snapshot_checkpoint(output,run=run)


def test_snapshot_rejects_a_checkpoint_metadata_hash_mismatch(tmp_path):
    checkpoint=tmp_path/'agent.pt';checkpoint.write_bytes(b'changed')
    (tmp_path/'agent.pt.metadata.json').write_text(json.dumps({'checkpoint_sha256':'wrong'}))
    with pytest.raises(ValueError,match='metadata hash'):
        snapshot_checkpoint(tmp_path/'audit',checkpoint=checkpoint)
