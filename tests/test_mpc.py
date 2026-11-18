"""Numerical tests use synthetic arrays only; none are deployment primitives."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from mpc_dex.mpc import CEMConfig, NoFeasiblePlan, PrimitiveCEM
from mpc_dex.primitives import PrimitiveBasis


def fixture_trajectories():
    angles = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    q = np.full((len(angles), 16), 0.2)
    q[:, 0] += 0.2 * np.cos(angles)
    q[:, 1] += 0.15 * np.sin(angles)
    return q


def fixture_primitive():
    return PrimitiveBasis.from_trajectories(
        fixture_trajectories(), explained_variance=1.0,
        provenance={"source": "synthetic numerical test fixture", "behavior": "none"},
    )


class PrimitiveTests(unittest.TestCase):
    def test_pca_rank_variance_reconstruction_and_ranges(self):
        q = fixture_trajectories()
        prior = PrimitiveBasis.from_trajectories(
            q, explained_variance=0.9, provenance={"source": "synthetic test fixture"},
        )
        self.assertEqual(prior.components, 2)
        self.assertGreaterEqual(prior.explained_variance_ratio.sum(), 0.9)
        z = prior.project(q)
        np.testing.assert_allclose(prior.reconstruct(z), q, atol=1e-14)
        np.testing.assert_allclose(prior.coefficient_min, z.min(axis=0))
        np.testing.assert_allclose(prior.coefficient_max, z.max(axis=0))
        np.testing.assert_allclose(prior.basis.T @ prior.basis, np.eye(2), atol=1e-14)

    def test_serialization_without_pickle_and_provenance(self):
        prior = fixture_primitive()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prior.npz"
            prior.save(path)
            with np.load(path, allow_pickle=False) as contents:
                for name in contents.files:
                    self.assertFalse(contents[name].dtype.hasobject)
            restored = PrimitiveBasis.load(path)
        np.testing.assert_array_equal(restored.mean, prior.mean)
        np.testing.assert_array_equal(restored.basis, prior.basis)
        np.testing.assert_array_equal(restored.coefficient_min, prior.coefficient_min)
        np.testing.assert_array_equal(restored.coefficient_max, prior.coefficient_max)
        self.assertEqual(restored.metadata, prior.metadata)
        self.assertEqual(len(restored.metadata["extraction"]["trajectories_float64_le_sha256"]), 64)
        self.assertFalse(restored.basis.flags.writeable)

    def test_reject_degenerate_nonfinite_and_missing_provenance(self):
        for q in (np.full((32, 16), 0.7), np.full((2, 16), np.nan), np.zeros((2, 15))):
            with self.assertRaises(ValueError):
                PrimitiveBasis.from_trajectories(q, provenance={"source": "test"})
        with self.assertRaises(ValueError):
            PrimitiveBasis.from_trajectories(fixture_trajectories(), provenance={})


class CEMTests(unittest.TestCase):
    def make_planner(self, **kwargs):
        config = CEMConfig(samples=8, elites=2, **kwargs)
        return PrimitiveCEM(fixture_primitive(), np.full(16, -2.), np.full(16, 2.), config, seed=7)

    def test_published_floor_scales_and_equal_split(self):
        planner = self.make_planner()
        d = planner.primitive.components
        spread = planner.primitive.coefficient_max - planner.primitive.coefficient_min
        np.testing.assert_array_equal(planner.exploration_floor[:d], spread)
        np.testing.assert_array_equal(planner.refinement_floor[:d], 0.3 * spread)
        np.testing.assert_array_equal(planner.exploration_floor[d:], np.ones(16))
        np.testing.assert_array_equal(planner.refinement_floor[d:], np.full(16, 0.1))

        class UnitNoise:
            @staticmethod
            def standard_normal(shape):
                return np.ones(shape)

        planner.rng = UnitNoise()
        std = np.zeros_like(planner.std_knots)
        std[0, 0] = 2.5
        sampled = planner._sample(planner.nominal_knots, std)
        for candidate in sampled[:4]:
            np.testing.assert_array_equal(candidate, np.maximum(std, planner.exploration_floor))
        for candidate in sampled[4:]:
            np.testing.assert_array_equal(candidate, np.maximum(std, planner.refinement_floor))

    def test_bounds_and_only_finite_costs_enter_elites(self):
        planner = self.make_planner(iterations=1)
        recorded = {}

        def evaluate(sequences):
            self.assertEqual(sequences.shape, (8, 50, 16))
            self.assertTrue(np.all(sequences >= -2))
            self.assertTrue(np.all(sequences <= 2))
            recorded["chosen"] = sequences[3, 0].copy()
            return np.array([np.inf, np.nan, -np.inf, 50., np.inf, np.inf, np.inf, np.inf])

        action, diagnostics = planner.plan(evaluate)
        np.testing.assert_array_equal(action, recorded["chosen"])
        self.assertEqual(diagnostics["iterations"][0]["elite_count"], 1)
        self.assertEqual(diagnostics["valid_candidates"], 1)

    def test_no_feasible_plan_raises_without_changing_warm_start(self):
        planner = self.make_planner(iterations=2)
        planner.reset(np.full(16, 0.9))
        before = planner.nominal_knots.copy()
        with self.assertRaises(NoFeasiblePlan):
            planner.plan(lambda sequences: np.full(len(sequences), np.inf))
        np.testing.assert_array_equal(planner.nominal_knots, before)
        self.assertIsNone(planner.last_sequence)

    def test_preserve_best_when_later_iteration_worsens_or_has_no_valid_samples(self):
        planner = self.make_planner(iterations=3)
        calls = []

        def evaluate(sequences):
            calls.append(sequences.copy())
            if len(calls) == 1:
                return np.arange(8, dtype=float) + 0.5
            if len(calls) == 2:
                return np.arange(8, dtype=float) + 20
            return np.full(8, np.inf)

        action, diagnostics = planner.plan(evaluate)
        np.testing.assert_array_equal(action, calls[0][0, 0])
        np.testing.assert_array_equal(planner.last_sequence, calls[0][0])
        self.assertEqual(diagnostics["best_cost"], 0.5)
        self.assertEqual(diagnostics["iterations"][-1]["elite_count"], 0)

    def test_optimizer_improves_tracking_objective(self):
        planner = PrimitiveCEM(
            fixture_primitive(), np.full(16, -2.), np.full(16, 2.),
            CEMConfig(iterations=8), seed=8,
        )
        target = np.full(16, 0.7)
        initial_cost = float(np.mean((planner.primitive.mean - target) ** 2))
        _, diagnostics = planner.plan(lambda sequences: np.mean((sequences - target) ** 2, axis=(1, 2)))
        history = diagnostics["iterations"]
        self.assertLess(diagnostics["best_cost"], initial_cost)
        self.assertLess(history[-1]["best_cost"], 0.9 * history[0]["best_cost"])
        self.assertTrue(all(a["best_cost"] >= b["best_cost"] for a, b in zip(history, history[1:])))

    def test_warm_start_interpolates_by_elapsed_control_time(self):
        planner = self.make_planner(iterations=1, control_seconds=0.15)
        samples = np.zeros((8, *planner.nominal_knots.shape))
        samples[0] = np.arange(6)[:, None] * 0.1
        samples[1] = np.arange(6)[:, None] * 0.3
        planner._sample = lambda mean, std: samples.copy()
        planner.plan(lambda sequences: np.array([0., 1., np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]))
        expected = np.array([0.3, 0.5, 0.7, 0.9, 1.0, 1.0])
        np.testing.assert_allclose(planner.nominal_knots, np.broadcast_to(expected[:, None], planner.nominal_knots.shape))
        expected_std = np.array([0.15, 0.25, 0.35, 0.45, 0.5, 0.5])
        np.testing.assert_allclose(planner.std_knots, np.broadcast_to(expected_std[:, None], planner.std_knots.shape))

    def test_config_timestep_and_callback_contract(self):
        self.assertEqual(CEMConfig(timestep_seconds=0.002).horizon_steps, 250)
        with self.assertRaises(ValueError):
            CEMConfig(samples=7)
        with self.assertRaises(ValueError):
            CEMConfig(timestep_seconds=0.003)
        with self.assertRaises(ValueError):
            self.make_planner(iterations=1).plan(lambda sequences: 0.0)


if __name__ == "__main__":
    unittest.main()
