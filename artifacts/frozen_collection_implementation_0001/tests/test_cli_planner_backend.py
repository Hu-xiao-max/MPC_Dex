"""Planner backend integration checks; no fabricated rotation demonstrations."""
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from mpc_dex import cli
from mpc_dex.env import AllegroRotationEnv, EnvConfig
from mpc_dex.native_rollouts import NativeBackendUnavailable, NativeMujocoRolloutEvaluator
from mpc_dex.primitives import PrimitiveBasis
from mpc_dex.rollouts import MujocoRolloutEvaluator


GRASP = str(Path(__file__).resolve().parents[1] / 'assets/initial_grasp.npz')


@pytest.fixture
def planning_inputs(tmp_path, monkeypatch):
    # Backend tests use numerical fixtures; source qualification is tested
    # separately and no fixture represents a successful SAC primitive.
    monkeypatch.setattr(cli, 'validate_primitive_source', lambda primitive: None)
    env = AllegroRotationEnv(EnvConfig(initial_grasp=GRASP, horizon_steps=1))
    env.reset(seed=0)
    # Arbitrary numerical PCA data exercises interfaces only. It is never
    # labeled, exported or admitted as a successful rotation demonstration.
    frames = env.data.ctrl + np.random.default_rng(9).normal(0, .01, (30, 16))
    primitive = tmp_path / 'interface_test_primitive.npz'
    PrimitiveBasis.from_trajectories(frames, provenance={
        'source': 'numerical backend interface test only',
        'successful_policy': False}).save(primitive)
    yield env, primitive
    env.close()


def integration_state(env, data=None):
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    result = np.empty(mujoco.mj_stateSize(env.model, spec))
    mujoco.mj_getState(env.model, env.data if data is None else data, result, spec)
    return result


@pytest.mark.parametrize('backend', ['unknown', None, [], 1])
def test_planner_factory_rejects_unknown_backend_before_loading_primitive(planning_inputs, backend):
    env, primitive = planning_inputs
    with pytest.raises(ValueError, match='planner_backend'):
        cli.planner_for(env, {'planner_backend': backend}, primitive.with_name('missing.npz'), 2)


def test_default_python_and_explicit_native_match_costs_and_full_states(planning_inputs):
    env, primitive = planning_inputs
    config = {'planner_cost': {'workers': 1, 'rotation_progress': 'retained'}}
    python_planner, python = cli.planner_for(env, config, primitive, 7)
    try:
        try:
            native_planner, native = cli.planner_for(env, dict(config, planner_backend='native'), primitive, 7)
        except NativeBackendUnavailable as error:
            pytest.skip(str(error))
        try:
            assert type(python) is MujocoRolloutEvaluator
            assert type(native) is NativeMujocoRolloutEvaluator and native.require_native
            assert asdict(native_planner.config) == asdict(python_planner.config)
            assert asdict(native.config) == asdict(python.config)
            assert native_planner.config.samples == 350
            assert native_planner.config.iterations == 3
            env.data.qfrc_applied[env.object_dadr] = .003
            env._contactless_seconds = .048
            before = integration_state(env)
            sequences = np.broadcast_to(env.data.ctrl, (3, 2, 16)).copy()
            np.testing.assert_array_equal(native(sequences), python(sequences))
            np.testing.assert_array_equal(integration_state(env, native.worker_data[0]),
                                          integration_state(env, python.worker_data[0]))
            np.testing.assert_array_equal(integration_state(env), before)
            assert native.last_diagnostics['candidate_physics_steps'] == [100]*3
            assert native.last_diagnostics['rejection_reasons'] == [None]*3
            assert env._contactless_seconds == .048
        finally:
            native.close()
    finally:
        python.close()


@pytest.mark.parametrize('backend', ['python', 'native'])
def test_collect_commits_runtime_provenance_before_first_plan(planning_inputs, tmp_path, monkeypatch, backend):
    env, primitive = planning_inputs
    config = {'environment': asdict(env.config), 'planner_backend': backend,
              'planner': {'horizon_seconds': .1, 'timestep_seconds': .1,
                          'samples': 4, 'elites': 2, 'iterations': 1, 'knots': 2},
              'planner_cost': {'workers': 1, 'rotation_progress': 'retained'}}
    configuration = tmp_path/'config.json'
    configuration.write_text(json.dumps(config))
    output = tmp_path/'collect'
    original = cli.PrimitiveCEM.plan
    observed = []
    def checked_plan(planner, evaluator):
        state = json.loads((output/'run.json').read_text())
        runtime = state['planner_runtime']
        assert state['status'] == 'running'
        assert state['planner_backend'] == runtime['backend'] == backend
        assert runtime['planner'] == asdict(planner.config)
        assert runtime['cost'] == asdict(evaluator.config)
        assert runtime['seed'] == 11
        for name, digest in runtime['numerical_source_sha256'].items():
            assert cli.hash_file(cli.ROOT/name) == digest
        if backend == 'native':
            assert runtime['require_native']
            assert runtime['evaluator_backend'] == 'exact_native_step_threads'
            assert runtime['native_build']['transition_abi_version'] == 2
            assert runtime['native_build']['binary_sha256'] == cli.hash_file(runtime['native_build']['binary'])
        else:
            assert runtime['evaluator_backend'] == 'exact_step_threads'
            assert not runtime['require_native'] and runtime['native_build'] is None
        observed.append(runtime)
        return original(planner, evaluator)
    monkeypatch.setattr(cli.PrimitiveCEM, 'plan', checked_plan)
    args = SimpleNamespace(config=str(configuration), primitive=str(primitive),
                           output=str(output), seed=11, episodes=1, max_attempts=1)
    try:
        assert cli.collect(args) == 2  # A single real control interval is not a full turn.
    except NativeBackendUnavailable as error:
        if backend == 'native':
            pytest.skip(str(error))
        raise
    state = json.loads((output/'run.json').read_text())
    assert len(observed) == 1
    assert state['successful_episodes'] == 0
    assert state['planner_runtime'] == observed[0]
    assert not (output/'successful_joint_trajectories.npz').exists()


def test_training_initializer_persists_runtime_without_claiming_training(planning_inputs, tmp_path):
    env, primitive = planning_inputs
    config = {'planner_cost': {'workers': 1}}
    run, state = cli.new_run(tmp_path/'initialization_only', 'train', config, [primitive])
    _, evaluator = cli._planner_for_run(run, state, env, config, primitive, 3)
    try:
        saved = json.loads((run/'run.json').read_text())
        assert saved['command'] == 'train'
        assert saved['planner_backend'] == 'python'
        assert saved['planner_runtime']['backend'] == 'python'
        assert 'mpc_dex/learning.py' in saved['planner_runtime']['numerical_source_sha256']
        assert 'successful_training_episodes' not in saved
        assert evaluator.last_diagnostics == {}
    finally:
        evaluator.close()


def test_explicit_native_failure_never_falls_back_and_marks_initialization_failed(
        planning_inputs, tmp_path, monkeypatch):
    import mpc_dex.native_rollouts as module
    env, primitive = planning_inputs
    def unavailable(*args, **kwargs):
        raise NativeBackendUnavailable('compiler deliberately unavailable')
    monkeypatch.setattr(module, '_load_library', unavailable)
    config = {'planner_backend': 'native', 'planner_cost': {'workers': 1}}
    with pytest.raises(NativeBackendUnavailable, match='compiler'):
        cli.planner_for(env, config, primitive, 0)
    run, state = cli.new_run(tmp_path/'unavailable', 'collect', config, [primitive])
    closed = []
    original_close = env.close
    def tracked_close():
        closed.append(True)
        original_close()
    monkeypatch.setattr(env, 'close', tracked_close)
    with pytest.raises(NativeBackendUnavailable, match='compiler'):
        cli._planner_for_run(run, state, env, config, primitive, 0)
    saved = json.loads((run/'run.json').read_text())
    assert saved['status'] == 'failed'
    assert saved['planner_backend'] == 'native'
    assert 'compiler' in saved['planner_initialization_error']
    assert 'planner_runtime' not in saved
    assert closed == [True]
