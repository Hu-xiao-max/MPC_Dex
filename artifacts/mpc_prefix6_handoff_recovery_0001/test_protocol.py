"""Synthetic unit fixtures only: no environment, planner, physics or success proof."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('prefix6_recovery_review', HERE / 'run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def synthetic_source(steps=9, terminal_last=True):
    source = dict(obs=np.arange(steps * 60, dtype=np.float32).reshape(steps, 60),
                  actions=np.linspace(-.5, .5, steps * 16, dtype=np.float32).reshape(steps, 16),
                  rewards=np.arange(steps, dtype=np.float64), next_obs=np.ones((steps, 60), np.float32),
                  terminated=np.zeros(steps, bool), truncated=np.zeros(steps, bool),
                  qpos=np.zeros((steps, 23)), qvel=np.zeros((steps, 22)), targets=np.zeros((steps, 16)),
                  integration_state=np.zeros((steps, 244)), initial_obs=np.zeros(60, np.float32),
                  initial_qpos=np.zeros(23), initial_integration_state=np.zeros(244))
    source['terminated'][-1] = terminal_last
    return source


def test_select_prefix_keeps_initial_fields_and_first_controls_only():
    prefix = runner.select_prefix(synthetic_source(), 6)
    assert len(prefix['actions']) == 6 and len(prefix['obs']) == 6
    np.testing.assert_array_equal(prefix['actions'], synthetic_source()['actions'][:6])
    assert prefix['initial_qpos'].shape == (23,) and prefix['initial_integration_state'].shape == (244,)
    assert not prefix['terminated'].any()


def test_select_prefix_rejects_whole_trajectory_and_terminated_prefix():
    with pytest.raises(ValueError):
        runner.select_prefix(synthetic_source(), 9)
    source = synthetic_source()
    source['terminated'][3] = True
    with pytest.raises(ValueError):
        runner.select_prefix(source, 6)


def test_correction_pairs_exclude_prefix_rows_and_keep_row_identity():
    steps = 11
    trace = {key: value for key, value in synthetic_source(steps, False).items() if not key.startswith('initial_')}
    trace['control_times'] = np.arange(steps + 1) * .1
    pairs = runner.correction_pairs(trace, 6)
    assert len(pairs['actions']) == 5
    np.testing.assert_array_equal(pairs['actions'], trace['actions'][6:])
    np.testing.assert_array_equal(pairs['source_control_1based'], [7, 8, 9, 10, 11])
    np.testing.assert_allclose(pairs['control_times'], trace['control_times'][7:])
    assert 'next_actions' not in pairs
    assert runner.correction_pairs(trace, 11) is None


def test_strict_success_requires_every_independent_condition():
    base = dict(environment_success=True, finger_supported_rotation_success=True, terminated=True, truncated=False,
                environment_drop=False, independent_drop=False, independent_palm_support_violation=False,
                no_external_support=True)
    assert runner.strict_force_success(base)
    for key, value in (('finger_supported_rotation_success', False), ('truncated', True), ('independent_drop', True),
                       ('independent_palm_support_violation', True), ('no_external_support', False), ('environment_success', False)):
        assert not runner.strict_force_success(dict(base, **{key: value}))
