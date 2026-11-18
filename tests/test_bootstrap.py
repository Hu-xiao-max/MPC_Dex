"""Exercise real contact simulation, persistence, and honest bootstrap outcomes."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mpc_dex.contracts import validate_checkpoint
from mpc_dex.replay import ReplayBuffer
from scripts.train_bootstrap import BootstrapRunner, parser, read_environment, sample_replay


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def small_torch_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def arguments(tmp_path, *, steps=3, num_envs=2, horizon=900, eval_every=1000, learning_starts=0):
    configuration = json.loads((ROOT / "configs/local_sim.json").read_text())
    configuration["environment"]["horizon_steps"] = horizon
    configuration["bootstrap"] = {"learner": {"hidden_sizes": [16, 16]}}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(configuration))
    return parser().parse_args(["--config", str(config_path), "--steps", str(steps),
                               "--output", str(tmp_path / "run"), "--num-envs", str(num_envs),
                               "--eval-every", str(eval_every), "--batch-size", "4",
                               "--learning-starts", str(learning_starts), "--replay-capacity", "64",
                               "--checkpoint-every", "1000", "--log-every", "1000"])


def committed(output):
    manifest = json.loads((Path(output) / "latest.json").read_text())
    directory = Path(output) / manifest["checkpoint_directory"]
    return directory, json.loads((directory / "state.json").read_text())


def test_actual_contact_steps_checkpoint_and_resume_open_episodes(tmp_path):
    args = arguments(tmp_path)
    runner = BootstrapRunner(args)
    initial_qpos = runner.envs[0].data.qpos.copy()
    assert runner.agent.bc_updates == runner.agent.pretrain_updates == 0
    assert runner.agent.config.ensemble_size == 2
    assert runner.agent.config.actor_warmup_updates == 0
    assert runner.run() == 0
    directory, state = committed(args.output)
    assert state["status"] == "budget_exhausted"
    assert state["environment_steps"] == state["online_updates"] == state["replay_transitions"] == 3
    assert state["completed_episodes"] == state["successful_trajectory_episodes"] == 0
    assert not (Path(args.output) / "successful_joint_trajectories.npz").exists()
    assert validate_checkpoint(directory / "agent.pt", runner.ec)["stage"] == "bootstrap"
    with np.load(directory / "environment_0.npz", allow_pickle=False) as archive:
        assert json.loads(str(archive["metadata"]))["steps"] == 2
        assert len(archive["joint_trace"]) == 3
    args.steps = 6
    resumed = BootstrapRunner(args)
    assert resumed.environment_steps == 3 and resumed.envs[0].steps == 2
    assert not np.array_equal(resumed.envs[0].data.qpos, initial_qpos)
    for buffer, observation in zip(resumed.buffers, resumed.observations):
        np.testing.assert_array_equal(buffer.transitions()["next_obs"][-1], observation)
    assert resumed.run() == 0
    directory, state = committed(args.output)
    assert state["environment_steps"] == state["online_updates"] == state["replay_transitions"] == 6
    assert state["completed_episodes"] == 0
    assert len(list(Path(args.output).glob("checkpoint_????????"))) == 2
    for i in range(2):
        replay = ReplayBuffer.load(directory / f"replay_{i}.npz")
        assert not replay.transitions()["terminated"].any()
        assert not replay.transitions()["truncated"].any()
        assert not replay.transitions()["next_action_valid"][-1]
        assert replay.transitions()["next_action_valid"][:-1].all()


def test_deterministic_evaluation_is_five_real_episodes_without_success_claim(tmp_path):
    args = arguments(tmp_path, steps=2, num_envs=1, horizon=2, eval_every=2, learning_starts=1000)
    runner = BootstrapRunner(args)
    stable = runner.envs[0].to_action(runner.envs[0].data.ctrl)
    np.testing.assert_allclose(runner.agent.act(runner.observations[0], True), stable, atol=1e-6)
    _, log_std = runner.agent.actor(torch.as_tensor(runner.observations[0]))
    torch.testing.assert_close(log_std.exp(), torch.full((16,), .05))
    assert runner.run() == 0
    records = [json.loads(line) for line in (Path(args.output) / "metrics.jsonl").read_text().splitlines()]
    evaluation = next(row for row in records if row["event"] == "evaluation")
    assert evaluation["complete"] and evaluation["episodes"] == evaluation["required_episodes"] == 5
    assert len(evaluation["results"]) == 5
    assert all(row["steps"] == 2 for row in evaluation["results"])
    assert evaluation["successes"] == 0
    assert not (Path(args.output) / "successful_joint_trajectories.npz").exists()
    _, state = committed(args.output)
    assert state["completed_episodes"] == 1 and state["online_updates"] == 0


def test_stop_request_commits_live_episode_without_labeling_it_completed(tmp_path, monkeypatch):
    args = arguments(tmp_path, steps=100, num_envs=1)
    runner = BootstrapRunner(args)
    real_update = runner.agent.update

    def update_and_stop(batch):
        metrics = real_update(batch)
        runner.stop_requested = True
        return metrics

    monkeypatch.setattr(runner.agent, "update", update_and_stop)
    assert runner.run() == 130
    directory, state = committed(args.output)
    assert state["status"] == "interrupted" and state["environment_steps"] == 1
    assert state["completed_episodes"] == state["successful_trajectory_episodes"] == 0
    replay = ReplayBuffer.load(directory / "replay_0.npz")
    assert not replay.transitions()["terminated"][0] and not replay.transitions()["truncated"][0]
    assert not json.loads((Path(args.output) / "run.json").read_text())["status"] == "complete"


def test_geometry_override_requires_matched_grasp(tmp_path):
    args = arguments(tmp_path)
    config = json.loads(Path(args.config).read_text())
    config["environment"]["object_diameter"] = .04
    Path(args.config).write_text(json.dumps(config))
    with pytest.raises(ValueError, match="matched grasp"):
        read_environment(args.config)


def test_success_gate_rejects_partial_rotation(tmp_path):
    args = arguments(tmp_path, steps=0, num_envs=1)
    runner = BootstrapRunner(args)
    try:
        with pytest.raises(ValueError, match="real completed"):
            runner.record_success(np.zeros((3, 16)), {"success": True, "drop": False, "angle": .5},
                                  "invalid partial trace", 0., 2)
        assert not runner.success_records
    finally:
        runner.close()


@pytest.mark.parametrize('support', [
    {'palm_support_violation': True, 'peak_palm_support_force': .4},
    {'palm_support_violation': False, 'peak_palm_support_force': .4},
    {'palm_support_violation': False, 'peak_palm_support_force': float('nan')},
    {},
])
def test_bootstrap_success_export_requires_no_palm_support_evidence(tmp_path, support):
    runner = BootstrapRunner(arguments(tmp_path, steps=0, num_envs=1))
    try:
        # A numerically full turn is insufficient without whole-episode
        # no-palm evidence. This synthetic record tests rejection only.
        info = dict(success=True, drop=False, angle=2*np.pi, retained_angle=2*np.pi,
                    finger_contacts=4, axis_tilt_radians=0., failure_reason=None, **support)
        with pytest.raises(ValueError, match='real completed'):
            runner.record_success(np.zeros((3, 16)), info, 'invalid synthetic claim', 0., 2)
        assert not runner.success_records and not runner.successful_traces
    finally:
        runner.close()


def test_uniform_sampling_weights_shards_by_transition_count():
    buffers = [ReplayBuffer(10, seed=1), ReplayBuffer(10, seed=2)]
    for i, length in enumerate((1, 9)):
        for step in range(length):
            buffers[i].add_transition([100 * i + step], [0.], 0., [100 * i + step + 1],
                                      terminated=step == length - 1)
    samples = sample_replay(buffers, 10_000, np.random.default_rng(8))
    assert np.mean(samples["obs"][:, 0] >= 100) == pytest.approx(.9, abs=.015)


@pytest.mark.parametrize('bc,online,origin',[(0,0,'untrained'),(12000,0,'BC-origin'),(12000,2,'SAC-trained')])
def test_diagnostic_success_exports_preserve_origin_without_primitive_eligibility(tmp_path,bc,online,origin):
    runner=BootstrapRunner(arguments(tmp_path,steps=0,num_envs=1))
    try:
        # Serialization fixture only: these counters and records claim no real
        # training or physics. Production extraction must reject this history.
        runner.agent.bc_updates=bc;runner.agent.online_updates=online
        info=dict(success=True,drop=False,angle=2*np.pi,retained_angle=2*np.pi,
                  finger_contacts=4,axis_tilt_radians=0.,failure_reason=None,
                  palm_support_violation=False,peak_palm_support_force=0.)
        runner.record_success(np.zeros((3,16)),info,'synthetic export unit test',0.,2)
        record=runner.success_records[0]
        assert record['policy_origin']==origin
        assert record['bc_updates']==bc and record['online_updates']==online
        assert record['eligible_for_primitive'] is False
        runner.checkpoint('complete')
        with np.load(runner.output/'successful_joint_trajectories.npz') as archive:
            metadata=json.loads(str(archive['metadata']))
        assert metadata['eligible_for_primitive'] is False
        assert metadata['frozen_checkpoint_collection'] is False
        assert 'learned bootstrap SAC' not in metadata['source']
    finally:runner.close()
