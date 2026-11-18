import json
import unittest

import numpy as np

from mpc_dex.scheduler import ChunkScheduler, SchedulerConfig


class ChunkSchedulerTests(unittest.TestCase):
    def test_published_ramp_and_endpoints(self):
        scheduler = ChunkScheduler()
        self.assertEqual(scheduler.policy_probability(0), .5)
        self.assertEqual(scheduler.policy_probability(3000), .75)
        self.assertEqual(scheduler.policy_probability(6000), 1.)
        self.assertEqual(scheduler.policy_probability(1_000_000), 1.)
        for seed in range(32):
            scheduler = ChunkScheduler(seed=seed)
            self.assertEqual(scheduler.select(6000), "policy")
            self.assertEqual(scheduler.select(6001), "policy")
            self.assertEqual(scheduler.select(100_000), "policy")

    def test_initial_probability_statistically(self):
        fraction = np.mean([ChunkScheduler(seed=seed).select(0) == "policy" for seed in range(4096)])
        self.assertLess(abs(fraction - .5), .03)

    def test_choice_held_inside_global_chunks_even_across_episode_resets(self):
        scheduler = ChunkScheduler(seed=32)
        for chunk_start in range(0, 6040, 40):
            expected = scheduler.select(chunk_start)
            self.assertEqual(scheduler.chunk_start_step, chunk_start)
            state = scheduler.state_dict()["rng_state"]
            for step in range(chunk_start, chunk_start + 40):
                scheduler.reset_episode()
                self.assertEqual(scheduler.select(step), expected)
                self.assertEqual(scheduler.select(step), expected)
                self.assertEqual(scheduler.state_dict()["rng_state"], state)

    def test_resume_json_mid_chunk_reproduces_future_draws(self):
        first = ChunkScheduler(seed=123)
        first.select(973)
        serialized = json.dumps(first.state_dict())
        resumed = ChunkScheduler(seed=999)
        resumed.load_state_dict(json.loads(serialized))
        self.assertEqual(resumed.select(973), first.select(973))
        for step in range(974, 7000):
            self.assertEqual(resumed.select(step), first.select(step))
        self.assertEqual(resumed.state_dict(), first.state_dict())

    def test_resume_before_first_selection(self):
        original = ChunkScheduler(seed=74)
        restored = ChunkScheduler()
        restored.load_state_dict(original.state_dict())
        self.assertEqual(original.select(0), restored.select(0))

    def test_skips_preserve_seeded_global_chunk_choices(self):
        dense = ChunkScheduler(seed=15)
        choices = {step: dense.select(step) for step in range(6040)}
        sparse = ChunkScheduler(seed=15)
        for step in (13, 401, 1801, 5999, 6000, 6039):
            self.assertEqual(sparse.select(step), choices[step])
        self.assertEqual(sparse.state_dict(), dense.state_dict())

    def test_invalid_and_out_of_order_steps(self):
        scheduler = ChunkScheduler()
        for step in (-1, True, .5, "1"):
            with self.assertRaises(ValueError):
                scheduler.select(step)
        scheduler.select(80)
        with self.assertRaises(ValueError):
            scheduler.select(79)
        scheduler.reset_episode()
        with self.assertRaises(ValueError):
            scheduler.select(0)

    def test_invalid_resume_is_atomic(self):
        scheduler = ChunkScheduler(seed=5)
        scheduler.select(6030)
        original = scheduler.state_dict()
        changed = scheduler.state_dict()
        changed["controller"] = "mpc"
        with self.assertRaises(ValueError):
            scheduler.load_state_dict(changed)
        self.assertEqual(scheduler.state_dict(), original)
        with self.assertRaises(ValueError):
            ChunkScheduler(SchedulerConfig(chunk_steps=20)).load_state_dict(original)


if __name__ == "__main__":
    unittest.main()
