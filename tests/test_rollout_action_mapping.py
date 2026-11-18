"""Arbitrary CEM targets must predict the actual normalized-action execution."""
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mpc_dex.env import AllegroRotationEnv, EnvConfig
from mpc_dex.rollouts import MujocoRolloutEvaluator, RolloutCostConfig
from mpc_dex.native_rollouts import NativeBackendUnavailable, NativeMujocoRolloutEvaluator


@pytest.mark.parametrize('backend', ['python', 'native'])
def test_unrounded_joint_targets_predict_executed_float32_actions_exactly(backend):
    env = AllegroRotationEnv(EnvConfig(
        initial_grasp=str(Path(__file__).resolve().parents[1] / 'assets/initial_grasp.npz')))
    env.reset(seed=4)
    evaluator = None
    try:
        cls = NativeMujocoRolloutEvaluator if backend == 'native' else MujocoRolloutEvaluator
        options = {'require_native': True} if backend == 'native' else {}
        try:
            evaluator = cls(env, .1, RolloutCostConfig(workers=1), **options)
        except NativeBackendUnavailable as error:
            pytest.skip(str(error))
        # CEM produces float64 joint commands, not pre-rounded actions. These
        # real commands stay close to the verified grasp while exercising the
        # lossy normalization boundary on every control interval.
        commands = np.stack([np.clip(env.data.ctrl + np.linspace(-1e-4, 1e-4, 16) * k,
                                    env.low, env.high) for k in (1, 2, 3)])
        actions = env.to_action(commands)
        executed = env.to_joint_targets(actions)
        assert actions.dtype == np.float32
        assert not np.array_equal(commands, executed)
        original_commands = commands.copy()
        before = env.data.__getstate__()
        costs = evaluator(commands[None])
        assert np.isfinite(costs[0])
        assert env.data.__getstate__() == before
        np.testing.assert_array_equal(commands, original_commands)
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        predicted = np.empty(mujoco.mj_stateSize(env.model, spec))
        mujoco.mj_getState(env.model, evaluator.worker_data[0], predicted, spec)
        predicted_qacc = evaluator.worker_data[0].qacc.copy()
        predicted_contacts = evaluator.worker_data[0].contact.geom.copy()
        prediction = evaluator.last_diagnostics
        for action in actions:
            _, _, terminated, truncated, _ = env.step(action)
            assert not (terminated or truncated)
        # Integration inputs include warmstart/controls/applied forces, unlike
        # a qpos-only check. Compare computed acceleration and contacts too;
        # profiling timers and unused scratch memory are not physical state.
        actual = np.empty_like(predicted)
        mujoco.mj_getState(env.model, env.data, actual, spec)
        np.testing.assert_array_equal(actual, predicted)
        np.testing.assert_array_equal(env.data.qacc, predicted_qacc)
        np.testing.assert_array_equal(env.data.contact.geom, predicted_contacts)
        assert prediction['joint_target_execution'] == 'float32_normalized_action'
        assert prediction['max_target_mapping_delta_radians'] == float(np.max(np.abs(executed-commands)))
    finally:
        if evaluator is not None:
            evaluator.close()
        env.close()
