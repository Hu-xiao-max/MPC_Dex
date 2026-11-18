"""Exact compiled transitions must preserve physics, rejection and ownership."""
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mpc_dex.env import AllegroRotationEnv, EnvConfig
from mpc_dex.rollouts import MujocoRolloutEvaluator, RolloutCostConfig
from mpc_dex.native_rollouts import (NativeBackendUnavailable, NativeMujocoRolloutEvaluator,
                                     _array_pointer, _owner_pointer)


GRASP = str(Path(__file__).resolve().parents[1] / 'assets/initial_grasp.npz')


class Recording:
    def _simulate_candidates(self, worker, candidates, sequences, snapshot,
                             initial_contactless, states, finger_counts, step_counts, reasons):
        super()._simulate_candidates(worker, candidates, sequences, snapshot,
            initial_contactless, states, finger_counts, step_counts, reasons)
        self.recorded = (states, finger_counts, step_counts, reasons)


class RecordingPython(Recording, MujocoRolloutEvaluator):
    pass


class RecordingNative(Recording, NativeMujocoRolloutEvaluator):
    pass


def integration_state(env, data=None):
    result = np.empty(mujoco.mj_stateSize(env.model, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(env.model, env.data if data is None else data, result,
                      mujoco.mjtState.mjSTATE_INTEGRATION)
    return result


@pytest.mark.parametrize('workers', [1, 3])
@pytest.mark.parametrize('scenario', ['stable', 'airborne', 'inherited_timer', 'table', 'workspace', 'divergence'])
def test_native_predictions_and_rejections_equal_python_bit_for_bit(workers, scenario):
    env = AllegroRotationEnv(EnvConfig(initial_grasp=GRASP))
    env.reset(seed=0)
    env._contactless_seconds = .048 if scenario in {'stable', 'inherited_timer'} else 0.
    env.data.qfrc_applied[env.object_dadr] = .003
    env.data.xfrc_applied[env.object_id, 1] = .001
    if scenario in {'airborne', 'inherited_timer'}:
        env.data.qpos[env.object_qadr] += .1
        env.data.qvel[env.object_dadr + 4] = 100.
    elif scenario == 'table':
        env.data.qpos[env.object_qadr + 2] = .01
    elif scenario == 'workspace':
        env.data.qpos[env.object_qadr] += .3
    mujoco.mj_forward(env.model, env.data)
    if scenario == 'divergence':
        env.data.time = 1.
        env.data.qvel[env.object_dadr] = 1e12
    before = integration_state(env)
    # A strided caller array is intentionally accepted and copied at the API.
    sequences = np.broadcast_to(env.data.ctrl, (6, 10, 16))
    python = RecordingPython(env, .01, RolloutCostConfig(workers=workers, rotation_progress='retained'))
    try:
        native = RecordingNative(env, .01, RolloutCostConfig(workers=workers, rotation_progress='retained'),
                                 require_native=True)
    except NativeBackendUnavailable as error:
        python.close()
        env.close()
        pytest.skip(str(error))
    try:
        expected = python(sequences)
        actual = native(sequences)
        np.testing.assert_array_equal(actual, expected)
        for native_array, python_array in zip(native.recorded[:3], python.recorded[:3]):
            np.testing.assert_array_equal(native_array, python_array)
        assert native.recorded[3] == python.recorded[3]
        for native_data, python_data in zip(native.worker_data, python.worker_data):
            np.testing.assert_array_equal(integration_state(env, native_data), integration_state(env, python_data))
        np.testing.assert_array_equal(integration_state(env), before)
        assert native.last_diagnostics['backend'] == 'exact_native_step_threads'
        assert len(native.native_provenance['binary_sha256']) == 64
        expected_steps = {'stable': 50, 'airborne': 25, 'inherited_timer': 1,
                          'table': 1, 'workspace': 1, 'divergence': 1}
        expected_reason = {'stable': None, 'airborne': 'contact_loss',
                           'inherited_timer': 'contact_loss', 'table': 'table_contact',
                           'workspace': 'workspace', 'divergence': 'divergence'}
        assert native.last_diagnostics['candidate_physics_steps'] == [expected_steps[scenario]]*6
        assert native.last_diagnostics['rejection_reasons'] == [expected_reason[scenario]]*6
    finally:
        native.close()
        python.close()
        env.close()


def test_native_unavailable_falls_back_to_exact_python(monkeypatch):
    import mpc_dex.native_rollouts as module
    def unavailable(*args, **kwargs):
        raise NativeBackendUnavailable('compiler intentionally unavailable')
    monkeypatch.setattr(module, '_load_library', unavailable)
    env = AllegroRotationEnv(EnvConfig(initial_grasp=GRASP))
    env.reset(seed=0)
    fallback = NativeMujocoRolloutEvaluator(env, config=RolloutCostConfig(workers=1))
    reference = MujocoRolloutEvaluator(env, config=RolloutCostConfig(workers=1))
    try:
        sequences = env.data.ctrl[None, None, :].copy()
        np.testing.assert_array_equal(fallback(sequences), reference(sequences))
        assert fallback.last_diagnostics['backend'] == 'exact_step_threads'
        assert 'compiler' in fallback.last_diagnostics['native_unavailable_reason']
        with pytest.raises(NativeBackendUnavailable, match='compiler'):
            NativeMujocoRolloutEvaluator(env, require_native=True)
    finally:
        fallback.close()
        reference.close()
        env.close()


@pytest.mark.parametrize('progress', ['raw', 'retained'])
def test_native_varied_controls_and_single_finger_counts(progress):
    env = AllegroRotationEnv(EnvConfig(initial_grasp=GRASP))
    env.reset(seed=3)
    env.data.qpos[env.object_qadr+1] += .08
    mujoco.mj_forward(env.model, env.data)
    assert env.finger_contacts() == 1
    env.initial_position = env.object_position().copy()
    sequences = np.clip(env.data.ctrl + np.random.default_rng(4).normal(0, .08, (7, 10, 16)),
                        env.low, env.high)
    python = RecordingPython(env, .01, RolloutCostConfig(workers=3, rotation_progress=progress))
    try:
        native = RecordingNative(env, .01, RolloutCostConfig(workers=3, rotation_progress=progress),
                                 require_native=True)
    except NativeBackendUnavailable as error:
        python.close()
        env.close()
        pytest.skip(str(error))
    try:
        np.testing.assert_array_equal(native(sequences), python(sequences))
        for actual, expected in zip(native.recorded[:3], python.recorded[:3]):
            np.testing.assert_array_equal(actual, expected)
        assert native.recorded[3] == python.recorded[3]
        assert (native.recorded[1][:, 0] == 1).any()
        assert native.recorded[1][:, 0].max() == 1
        assert len(np.unique(native.recorded[0][:, 0, 1])) > 1
    finally:
        native.close()
        python.close()
        env.close()


def test_native_pointer_guards_reject_unowned_and_wrong_model_memory():
    model = mujoco.MjModel.from_xml_string('<mujoco/>')
    other = mujoco.MjModel.from_xml_string('<mujoco/>')
    with pytest.raises(NativeBackendUnavailable, match='Expected exact'):
        _owner_pointer(123456, mujoco.MjModel)
    with pytest.raises(NativeBackendUnavailable, match='different'):
        _owner_pointer(mujoco.MjData(other), mujoco.MjData, model)
    with pytest.raises(ValueError, match='contiguous'):
        _array_pointer(np.zeros((4, 4))[:, ::2], (4, 2), np.float64)
    with pytest.raises(ValueError, match='float64'):
        _array_pointer(np.zeros(3, dtype=np.float32), (3,), np.float64)


@pytest.mark.parametrize('direction', [-1, 0, 1])
def test_native_exact_workspace_boundary_uses_python_comparison(direction):
    env = AllegroRotationEnv(EnvConfig(initial_grasp=GRASP))
    env.reset(seed=0)
    trial = mujoco.MjData(env.model)
    mujoco.mj_copyData(trial, env.model, env.data)
    # Measure the boundary for the target the action interface actually sends.
    trial.ctrl[:] = env.to_joint_targets(env.to_action(env.data.ctrl))
    mujoco.mj_step(env.model, trial)
    mujoco.mj_forward(env.model, trial)
    distance = np.linalg.norm(env.object_position(trial)-env.initial_position)
    radius = np.nextafter(distance, -np.inf if direction < 0 else np.inf) if direction else distance
    config = RolloutCostConfig(workers=2, workspace_radius=radius)
    python = RecordingPython(env, .002, config)
    try:
        native = RecordingNative(env, .002, config, require_native=True)
    except NativeBackendUnavailable as error:
        python.close()
        env.close()
        pytest.skip(str(error))
    try:
        sequences = np.broadcast_to(env.data.ctrl, (5, 1, 16)).copy()
        np.testing.assert_array_equal(native(sequences), python(sequences))
        assert native.last_diagnostics['native_boundary_fallback_candidates'] == 5
        for actual, expected in zip(native.recorded[:3], python.recorded[:3]):
            np.testing.assert_array_equal(actual, expected)
        assert native.recorded[3] == python.recorded[3]
        for actual, expected in zip(native.worker_data, python.worker_data):
            np.testing.assert_array_equal(integration_state(env, actual), integration_state(env, expected))
    finally:
        native.close()
        python.close()
        env.close()


def test_native_callbacks_are_rejected_before_compilation(monkeypatch):
    monkeypatch.setattr(mujoco, 'get_mjcb_control', lambda: lambda model, data: None)
    env = AllegroRotationEnv(EnvConfig(initial_grasp=GRASP))
    try:
        with pytest.raises(NativeBackendUnavailable, match='callbacks'):
            NativeMujocoRolloutEvaluator(env, require_native=True)
    finally:
        env.close()


def _move_to_palm(env, supported):
    local = np.array([.015, 0., .005] if supported else [.025, 0., .01])
    env.data.qpos[env.object_qadr:env.object_qadr+3] = (
        env.data.xpos[env.palm_id] + env.data.xmat[env.palm_id].reshape(3, 3) @ local)
    env.data.qvel[:] = 0.
    mujoco.mj_forward(env.model, env.data)
    contacts = [contact for contact in env.data.contact if env.object_geom in contact.geom
                and env.model.geom_bodyid[contact.geom2 if contact.geom1 == env.object_geom
                                          else contact.geom1] == env.palm_id]
    assert contacts
    assert (env.palm_support_force() > env.config.palm_support_force_tolerance) == supported


@pytest.mark.parametrize('scenario', ['supported', 'zero_force_proximity', 'sticky_history',
                                     'supported_workspace', 'reorientation'])
def test_native_force_based_palm_rejection_matches_python(scenario):
    task = 'reorientation' if scenario == 'reorientation' else 'rotation'
    env = AllegroRotationEnv(EnvConfig(initial_grasp=GRASP, task=task))
    env.reset(seed=0)
    if scenario != 'sticky_history':
        _move_to_palm(env, supported=scenario != 'zero_force_proximity')
    else:
        assert env.palm_support_force() == 0.
    env._palm_support_violation = scenario in {'sticky_history', 'reorientation'}
    if scenario == 'supported_workspace':
        env.initial_position += np.array([.3, 0., 0.])
    before = integration_state(env)
    python = RecordingPython(env, .002, RolloutCostConfig(workers=2))
    try:
        native = RecordingNative(env, .002, RolloutCostConfig(workers=2), require_native=True)
    except NativeBackendUnavailable as error:
        python.close()
        env.close()
        pytest.skip(str(error))
    try:
        sequences = np.broadcast_to(env.data.ctrl, (5, 1, 16)).copy()
        np.testing.assert_array_equal(native(sequences), python(sequences))
        for actual, expected in zip(native.recorded[:3], python.recorded[:3]):
            np.testing.assert_array_equal(actual, expected)
        assert native.recorded[3] == python.recorded[3]
        rejected = scenario in {'supported', 'supported_workspace', 'sticky_history'}
        assert native.last_diagnostics['rejection_reasons'] == ['palm_support' if rejected else None]*5
        assert native.last_diagnostics['candidate_physics_steps'] == [0 if scenario == 'sticky_history' else 1]*5
        for actual, expected in zip(native.worker_data, python.worker_data):
            np.testing.assert_array_equal(integration_state(env, actual), integration_state(env, expected))
        np.testing.assert_array_equal(integration_state(env), before)
    finally:
        native.close()
        python.close()
        env.close()


@pytest.mark.parametrize('direction', [-1, 0, 1])
def test_native_palm_force_threshold_comparison_is_bit_exact(direction):
    env = AllegroRotationEnv(EnvConfig(initial_grasp=GRASP))
    env.reset(seed=0)
    _move_to_palm(env, supported=True)
    trial = mujoco.MjData(env.model)
    mujoco.mj_copyData(trial, env.model, env.data)
    trial.ctrl[:] = env.to_joint_targets(env.to_action(env.data.ctrl))
    mujoco.mj_step(env.model, trial)
    mujoco.mj_forward(env.model, trial)
    force = env.palm_support_force(trial)
    assert force > 1.
    env.config.palm_support_force_tolerance = (
        np.nextafter(force, -np.inf if direction < 0 else np.inf) if direction else force)
    python = RecordingPython(env, .002, RolloutCostConfig(workers=1))
    try:
        native = RecordingNative(env, .002, RolloutCostConfig(workers=1), require_native=True)
    except NativeBackendUnavailable as error:
        python.close()
        env.close()
        pytest.skip(str(error))
    try:
        sequences = env.data.ctrl[None, None, :].copy()
        np.testing.assert_array_equal(native(sequences), python(sequences))
        assert native.last_diagnostics['rejection_reasons'] == ['palm_support' if direction < 0 else None]
        for actual, expected in zip(native.recorded[:3], python.recorded[:3]):
            np.testing.assert_array_equal(actual, expected)
    finally:
        native.close()
        python.close()
        env.close()
