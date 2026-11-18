"""Train the independent simulation SAC prerequisite for rotation primitives.

This is bootstrap SAC, not MPC-scaffolded learning or the authors' code. It uses
a free object driven only by hand contacts. The optional incremental bootstrap
configuration supplies an explicitly separate shaping reward and partial seed.
The default 45 mm local object differs from the companion's 40 mm extraction
object. Another configuration requires a grasp synthesized for that geometry.

--steps is a cumulative transition budget, also when resuming an existing run.
Checkpoint generations retain live simulator and unfinished episode states.
Only complete, successful 2*pi episodes contribute primitive extraction data.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
import torch

from mpc_dex.env import AllegroRotationEnv, EnvConfig, JOINT_NAMES, ROTATION_VALIDATION
from mpc_dex.bootstrap_env import create_bootstrap_environment, policy_interface
from mpc_dex.contracts import environment_contract, validate_checkpoint, write_checkpoint_metadata
from mpc_dex.learning import LearnerConfig, SACAgent
from mpc_dex.replay import ReplayBuffer


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_environment(path):
    config = json.loads(Path(path).read_text())
    values = config.get("environment", {}).copy()
    if values.get("initial_grasp"):
        grasp = Path(values["initial_grasp"])
        values["initial_grasp"] = str(grasp if grasp.is_absolute() else ROOT / grasp)
    env_config = EnvConfig(**values)
    if env_config.task != "rotation" or not env_config.terminate_on_success:
        raise ValueError("bootstrap extraction requires rotation with terminate_on_success=True")
    if not env_config.initial_grasp:
        raise ValueError("bootstrap requires an independently verified initial grasp")
    with np.load(env_config.initial_grasp, allow_pickle=False) as archive:
        record = json.loads(str(archive["metadata_json"]))
    # Check the physical parameters that determine whether the solved grasp is
    # applicable. Initial_hand_q is superseded by the actual saved grasp state.
    for key in ("object_shape", "object_diameter", "object_length", "object_mass",
                "object_friction", "finger_friction", "timestep", "joint_kp",
                "joint_kv", "max_joint_force", "palm_height"):
        if record["config"][key] != getattr(env_config, key):
            raise ValueError(f"initial grasp has a different {key}; supply a matched grasp")
    return config, env_config


def sample_replay(buffers, batch_size, rng):
    """Uniform transition sampling across independent environment replay shards."""
    lengths = np.array([len(buffer) for buffer in buffers], dtype=np.float64)
    if lengths.sum() == 0:
        raise ValueError("cannot sample empty replay")
    counts = rng.multinomial(batch_size, lengths / lengths.sum())
    batches = [buffer.sample(int(count)) for buffer, count in zip(buffers, counts) if count]
    order = rng.permutation(batch_size)
    return {key: np.concatenate([batch[key] for batch in batches])[order] for key in batches[0]}


def load_bootstrap_seed(path, env_config, interface):
    """Validate seed identity and expose complete episodes without reset links."""
    path=Path(path)
    metadata=json.loads(path.with_suffix(path.suffix+'.metadata.json').read_text())
    kind=metadata.get('kind')
    if (kind not in {'partial-bootstrap-seed','verified-bootstrap-demonstrations'}
        or metadata.get('replay_sha256')!=sha256(path)
        or metadata.get('environment_contract')!=environment_contract(env_config)
        or metadata.get('policy_interface')!=interface):
        raise ValueError('Bootstrap seed has an incompatible kind, contract or hash')
    replay=ReplayBuffer.load(path)
    if not len(replay) or metadata.get('transitions')!=len(replay):
        raise ValueError('Bootstrap seed transition count disagrees with replay')
    rows=replay.transitions()
    observation_dim=60 if env_config.task=='rotation' else 61
    if any(rows[key].shape!=(len(replay),dimension) for key,dimension in
           [('obs',observation_dim),('next_obs',observation_dim),('actions',16),('next_actions',16)]):
        raise ValueError('Bootstrap seed observation/action dimensions disagree with the environment contract')
    ends=np.flatnonzero(rows['terminated'] | rows['truncated'])
    if not len(ends) or ends[-1]!=len(replay)-1 or len(ends)!=replay.episode_count:
        raise ValueError('Bootstrap seed needs complete explicit episode boundaries')
    episodes=[];start=0
    for end in ends:
        episode={key:value[start:end+1] for key,value in rows.items()}
        if (episode['next_action_valid'][-1] or not episode['next_action_valid'][:-1].all()
            or not np.array_equal(episode['next_actions'][:-1],episode['actions'][1:])):
            raise ValueError('Bootstrap seed next actions do not match executed episode successors')
        # Apply the replay's finite-value, continuity and action-bound checks.
        check=ReplayBuffer(len(episode['actions']));check.add_episode(episode)
        episodes.append(episode);start=end+1
    if kind=='partial-bootstrap-seed':
        if (len(episodes)!=1 or rows['terminated'].any() or not rows['truncated'][-1]
                or metadata.get('successful_full_turns')!=0):
            raise ValueError('Partial bootstrap seed must remain one non-success truncated episode')
    else:
        count=metadata.get('demonstration_successes')
        records=metadata.get('episodes')
        if (interface!={'kind':'absolute_joint_target'} or not env_config.terminate_on_success
            or metadata.get('rotation_validation')!=ROTATION_VALIDATION
            or metadata.get('demonstrator')!='MPC diagnostic (not SAC)'
            or metadata.get('primitive_eligible') is not False
            or metadata.get('main_mpc_dataset_eligible') is not False
            or type(count) is not int or count!=len(episodes)
            or metadata.get('successful_full_turns')!=count
            or not isinstance(records,list) or len(records)!=count
            or any(not isinstance(record,dict) for record in records)
            or rows['truncated'].any() or not rows['terminated'][ends].all()):
            raise ValueError('Verified demonstration metadata or terminal episode boundaries disagree')
        source_digests=[record.get('source_sha256') for record in records]
        if (any(not isinstance(digest,str) or len(digest)!=64
                or any(character not in '0123456789abcdef' for character in digest)
                for digest in source_digests)
            or len(set(source_digests))!=count):
            raise ValueError('Verified demonstrations must identify distinct source archives')
        for episode,record in zip(episodes,records):
            info=record.get('final_info',{});audit=record.get('audit',{})
            angle=info.get('angle',float('nan'));retained=info.get('retained_angle',float('nan'))
            tilt=info.get('axis_tilt_radians',float('nan'));peak=info.get('peak_palm_support_force',float('nan'))
            if (record.get('used_commands')!=len(episode['actions'])
                or type(record.get('source_commands')) is not int
                or type(record.get('unused_tail_commands')) is not int
                or record['unused_tail_commands']<0
                or record['source_commands']!=record['used_commands']+record['unused_tail_commands']
                or info.get('success') is not True or info.get('drop') is not False
                or info.get('palm_support_violation') is not False or info.get('failure_reason') is not None
                or not np.isfinite([angle,retained,tilt,peak]).all() or min(angle,retained)<2*np.pi
                or not 0<=tilt<=np.radians(env_config.success_axis_tolerance_degrees)
                or not 0<=peak<=env_config.palm_support_force_tolerance
                or info.get('finger_contacts',0)<2
                or audit.get('finger_supported_rotation_success') is not True
                or audit.get('independent_rotation_success') is not True
                or audit.get('environment_success') is not True
                or audit.get('steps')!=len(episode['actions'])
                or not np.isfinite(audit.get('final_force_qualified_angle',float('nan')))
                or audit.get('final_force_qualified_angle',float('-inf'))<2*np.pi
                or audit.get('no_external_support') is not True
                or audit.get('independent_palm_support_violation') is not False
                or audit.get('independent_drop') is not False):
                raise ValueError('Verified demonstration lacks full-turn no-palm force-audit evidence')
    return replay,metadata,episodes


_ENV_ARRAYS = ("initial_position", "initial_orientation", "last_action", "last_q", "last_p", "_last_object_quaternion")
_ENV_SCALARS = ("start_angle", "_last_wrapped", "angle", "previous_angle", "steps", "rotations",
                "dwell", "goal", "heading_error", "_episode_over", "_contactless_seconds", "_goal_rewarded", "retained_angle", "previous_retained_angle", "retained_positive_step", "retained_negative_step",
                "_palm_support_violation", "_peak_palm_support_force")
_STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


def save_environment(path, env, observation, episode_return, joint_trace):
    state = np.empty(mujoco.mj_stateSize(env.model, _STATE_SPEC))
    mujoco.mj_getState(env.model, env.data, state, _STATE_SPEC)
    metadata = {key: getattr(env, key) for key in _ENV_SCALARS}
    metadata.update(rng_state=env.np_random.bit_generator.state, episode_return=episode_return)
    with Path(path).open("wb") as stream:
        np.savez_compressed(stream, integration_state=state, observation=observation,
                            joint_trace=np.asarray(joint_trace).reshape(-1, 16),
                            metadata=np.asarray(json.dumps(metadata)),
                            **{key: getattr(env, key) for key in _ENV_ARRAYS})


def restore_environment(path, env):
    env.reset()
    with np.load(path, allow_pickle=False) as archive:
        state = archive["integration_state"]
        if state.shape != (mujoco.mj_stateSize(env.model, _STATE_SPEC),):
            raise ValueError("saved simulator state does not match the environment")
        mujoco.mj_setState(env.model, env.data, state, _STATE_SPEC)
        mujoco.mj_forward(env.model, env.data)
        metadata = json.loads(str(archive["metadata"]))
        for key in _ENV_ARRAYS:
            setattr(env, key, archive[key].copy())
        for key in _ENV_SCALARS:
            setattr(env, key, metadata[key])
        env.np_random.bit_generator.state = metadata["rng_state"]
        return archive["observation"].copy(), float(metadata["episode_return"]), list(archive["joint_trace"].copy())


class BootstrapRunner:
    def __init__(self, args):
        self.args = args
        if min(args.num_envs, args.batch_size, args.replay_capacity, args.checkpoint_every,
               args.log_every, args.eval_every) <= 0 or args.steps < 0 or args.learning_starts < 0:
            raise ValueError("invalid bootstrap run limits")
        if args.updates_per_step <= 0:
            raise ValueError("updates_per_step must be positive")
        if args.replay_capacity < args.num_envs:
            raise ValueError("replay capacity must cover every environment")
        self.output = Path(args.output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.config, self.ec = read_environment(args.config)
        self.stop_requested = False
        self.rng = np.random.default_rng(args.seed)
        self.pool = ThreadPoolExecutor(args.num_envs) if args.num_envs > 1 else None
        bootstrap_settings=self.config.get('bootstrap',{})
        self.policy_interface=policy_interface(bootstrap_settings)
        seed_settings=bootstrap_settings.get('seed',{})
        self.seed_replay=None
        self.seed_path=None
        self.seed_metadata=None
        self.seed_episodes=[]
        self.demonstration_successes=0
        if seed_settings:
            self.seed_path=Path(seed_settings['replay'])
            if not self.seed_path.is_absolute():self.seed_path=ROOT/self.seed_path
            self.seed_replay,self.seed_metadata,self.seed_episodes=load_bootstrap_seed(
                self.seed_path,self.ec,self.policy_interface)
            self.demonstration_successes=(self.seed_metadata['demonstration_successes']
                if self.seed_metadata['kind']=='verified-bootstrap-demonstrations' else 0)
            assigned=[sum(len(episode['actions']) for episode in self.seed_episodes[i::args.num_envs])
                      for i in range(args.num_envs)]
            if max(assigned)>args.replay_capacity//args.num_envs:
                raise ValueError('Seed episodes exceed the initial replay shard capacity')
            if min(seed_settings.get('bc_updates',0),seed_settings.get('critic_updates',0))<0:
                raise ValueError('Seed update counts must be nonnegative')
        self.envs = [create_bootstrap_environment(self.ec,bootstrap_settings) for _ in range(args.num_envs)]
        self.evaluation_env = create_bootstrap_environment(self.ec,bootstrap_settings)
        self.observations, self.episode_returns, self.joint_traces = [], [], []
        self.successful_traces = []
        self.success_records = []
        self.latest_metrics = {}
        self.environment_steps = self.completed_episodes = self.drops = self.successes = 0
        self.evaluations = self.eval_successes = self.generation = 0
        self.next_eval = args.eval_every
        self.next_checkpoint = args.checkpoint_every
        self.next_log = args.log_every
        self.elapsed_seconds = 0.0
        self.started = time.monotonic()
        self.provenance = {
            "implementation": "independent-paper-reproduction", "stage": "simulation-sac-bootstrap",
            "environment": asdict(self.ec), "initial_grasp_sha256": sha256(self.ec.initial_grasp),
            "environment_source_sha256": sha256(ROOT / "mpc_dex/env.py"),
            "learning_source_sha256": sha256(ROOT / "mpc_dex/learning.py"),
            "bootstrap_source_sha256": sha256(__file__), "mujoco_version": mujoco.__version__,
            "torch_version": str(torch.__version__), "numpy_version": np.__version__,
            "num_envs": args.num_envs, "seed": args.seed, "batch_size": args.batch_size,
            "learning_starts": args.learning_starts, "updates_per_step": args.updates_per_step,
            "replay_capacity": args.replay_capacity, "eval_every": args.eval_every,
            "bootstrap_settings": self.config.get("bootstrap", {}),
            "seed_replay_sha256": sha256(self.seed_path) if self.seed_path else None,
            "seed_metadata_sha256": sha256(self.seed_path.with_suffix(self.seed_path.suffix+'.metadata.json')) if self.seed_path else None,
            "demonstration_successes": self.demonstration_successes,
            "policy_interface": self.policy_interface,
            "reward": "bootstrap retained-rotation shaping" if bootstrap_settings.get('control') else "unchanged paper rotation reward", "hardware_verified": False,
            "paper_performance_verified": False, "companion_40mm_geometry": self.ec.object_diameter == .04,
        }
        manifest = self.output / "latest.json"
        if manifest.exists():
            self._restore(json.loads(manifest.read_text()))
        else:
            if (self.output / "run.json").exists():
                raise ValueError("run.json exists without a committed checkpoint; use a new output directory")
            torch.manual_seed(args.seed)
            defaults = dict(ensemble_size=2, online_target_critics=2, actor_warmup_updates=0,
                            critic_lr=3e-4, initial_std=.05)
            defaults.update(self.config.get("bootstrap", {}).get("learner", {}))
            self.agent = SACAgent(60, 16, LearnerConfig(**defaults), args.device)
            self.buffers = [ReplayBuffer(max(1, args.replay_capacity // args.num_envs), args.seed + i)
                            for i in range(args.num_envs)]
            for i, env in enumerate(self.envs):
                observation, _ = env.reset(seed=args.seed + i)
                self.observations.append(observation)
                self.episode_returns.append(0.)
                self.joint_traces.append([env.data.qpos[env.q_ids].copy()])
            # Constant initial mean at the physically verified grasp. Its head
            # and trunk remain trainable; this initialization is not BC.
            stable_action = (np.zeros(16) if self.policy_interface['kind']=='bootstrap_joint_increment'
                             else self.envs[0].to_action(self.envs[0].data.ctrl))
            with torch.no_grad():
                self.agent.actor.mean_head.weight.zero_()
                self.agent.actor.mean_head.bias.copy_(torch.as_tensor(
                    np.arctanh(np.clip(stable_action, -1 + 1e-6, 1 - 1e-6)), device=self.agent.device))
            self.agent.initialize_std()
            if self.seed_replay is not None:
                for _ in range(seed_settings.get('bc_updates',0)):
                    metrics=self.agent.bc_step(self.seed_replay.sample(args.batch_size))
                self.agent.initialize_std()
                for _ in range(seed_settings.get('critic_updates',0)):
                    metrics=self.agent.pretrain_critic_step(self.seed_replay.sample(args.batch_size))
                # Seeds have their own recorded episode boundaries and never
                # increment online steps, SAC successes or trajectory exports.
                for index,episode in enumerate(self.seed_episodes):
                    self.buffers[index%len(self.buffers)].add_episode(episode)
                self.log('partial_seed_initialized' if self.seed_metadata['kind']=='partial-bootstrap-seed'
                         else 'verified_demonstrations_initialized',transitions=len(self.seed_replay),
                         bc_updates=self.agent.bc_updates,critic_updates=self.agent.pretrain_updates,
                         successful_full_turns=0,demonstration_successes=self.demonstration_successes)
            self.provenance["created_at"] = datetime.now(timezone.utc).isoformat()
            self.checkpoint("running")

    def log(self, event, **values):
        row = dict(event=event, environment_steps=self.environment_steps,
                   wall_seconds=self.elapsed_seconds + time.monotonic() - self.started, **values)
        encoded = json.dumps(row, allow_nan=False)
        with (self.output / "metrics.jsonl").open("a") as stream:
            stream.write(encoded + "\n")
            stream.flush()
        print(encoded, flush=True)

    def _restore(self, manifest):
        directory = self.output / manifest["checkpoint_directory"]
        state = json.loads((directory / "state.json").read_text())
        saved = state["provenance"]
        for key, current in self.provenance.items():
            # JSON canonicalization equates EnvConfig tuple/list representations.
            if json.dumps(saved[key], sort_keys=True) != json.dumps(current, sort_keys=True):
                raise ValueError(f"resume configuration or source differs at {key}")
        self.provenance = saved
        validate_checkpoint(directory / "agent.pt", self.ec)
        self.agent = SACAgent.load(directory / "agent.pt", self.args.device)
        self.buffers = [ReplayBuffer.load(directory / f"replay_{i}.npz") for i in range(self.args.num_envs)]
        for i, env in enumerate(self.envs):
            obs, episode_return, trace = restore_environment(directory / f"environment_{i}.npz", env)
            self.observations.append(obs)
            self.episode_returns.append(episode_return)
            self.joint_traces.append(trace)
        for key in ("environment_steps", "completed_episodes", "drops", "successes", "evaluations",
                    "eval_successes", "generation", "next_eval", "next_checkpoint", "next_log", "elapsed_seconds"):
            setattr(self, key, state[key])
        self.rng.bit_generator.state = state["rng_state"]
        self.latest_metrics = state["latest_metrics"]
        rng = torch.load(directory / "torch_rng.pt", map_location="cpu", weights_only=True)
        torch.set_rng_state(rng["cpu"])
        if self.agent.device.type == "cuda" and "cuda" in rng:
            torch.cuda.set_rng_state_all(rng["cuda"])
        trajectory_path = directory / "successful_joint_trajectories.npz"
        if trajectory_path.exists():
            with np.load(trajectory_path, allow_pickle=False) as archive:
                self.successful_traces = [archive["joint_positions"].copy()]
                self.success_records = json.loads(str(archive["metadata"]))["episodes"]
        self.log("resumed", checkpoint_directory=manifest["checkpoint_directory"], status=state["status"])

    def checkpoint(self, status):
        self.generation += 1
        name = f"checkpoint_{self.generation:08d}"
        # A crash after generation rename but before manifest publication can
        # leave an uncommitted complete directory. Never reuse its name.
        while (self.output / name).exists():
            self.generation += 1
            name = f"checkpoint_{self.generation:08d}"
        temporary = self.output / (name + ".pending")
        directory = self.output / name
        temporary.mkdir(exist_ok=True)
        self.agent.save(temporary / "agent.pt")
        write_checkpoint_metadata(temporary / "agent.pt", self.ec, self.agent.config, stage="bootstrap",
                                  policy_interface=self.policy_interface)
        for i, (buffer, env) in enumerate(zip(self.buffers, self.envs)):
            buffer.save(temporary / f"replay_{i}.npz")
            save_environment(temporary / f"environment_{i}.npz", env, self.observations[i],
                             self.episode_returns[i], self.joint_traces[i])
        rng = {"cpu": torch.get_rng_state()}
        if self.agent.device.type == "cuda":
            rng["cuda"] = torch.cuda.get_rng_state_all()
        torch.save(rng, temporary / "torch_rng.pt")
        if self.success_records:
            metadata = dict(source="learned bootstrap SAC policy in free-object MuJoCo contact simulation",
                            successful_episodes=len(self.success_records), task="rotation", environment=asdict(self.ec),
                            episodes=self.success_records, joint_order=JOINT_NAMES, units="radians",
                            policy_only=True, domain_randomization=False,
                            rotation_validation=ROTATION_VALIDATION,
                            environment_contract=environment_contract(self.ec),
                            implementation="independent-paper-reproduction", provenance=self.provenance)
            np.savez_compressed(temporary / "successful_joint_trajectories.npz",
                                joint_positions=np.concatenate(self.successful_traces),
                                metadata=np.asarray(json.dumps(metadata)))
        state = {key: getattr(self, key) for key in ("environment_steps", "completed_episodes", "drops",
                 "successes", "evaluations", "eval_successes", "generation", "next_eval", "next_checkpoint", "next_log")}
        state.update(status=status, provenance=self.provenance, rng_state=self.rng.bit_generator.state,
                     elapsed_seconds=self.elapsed_seconds + time.monotonic() - self.started,
                     latest_metrics=self.latest_metrics, learner_config=asdict(self.agent.config),
                     successful_trajectory_episodes=len(self.success_records),
                     replay_transitions=sum(map(len, self.buffers)), online_updates=self.agent.online_updates,
                     demonstration_successes=self.demonstration_successes,
                     target_steps=self.args.steps)
        write_json(temporary / "state.json", state)
        temporary.replace(directory)
        # This manifest is the commit point. A process failure before it leaves
        # the preceding generation authoritative, never a mixed checkpoint.
        write_json(self.output / "latest.json", {"checkpoint_directory": name,
                   "environment_steps": self.environment_steps, "status": status})
        write_json(self.output / "run.json", state)
        if self.success_records:
            trajectory = self.output / "successful_joint_trajectories.npz"
            shutil.copyfile(directory / trajectory.name, trajectory.with_suffix(".tmp"))
            trajectory.with_suffix(".tmp").replace(trajectory)
        # Retain the latest two committed generations. All are this runner's
        # derived files; source assets and logs are never removed.
        generations = sorted(self.output.glob("checkpoint_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]"))
        for obsolete in generations[:-2]:
            shutil.rmtree(obsolete)

    def record_success(self, trace, info, source, episode_return, episode_steps):
        if (not info["success"] or info["drop"] or info["angle"] < 2 * math.pi
            or info.get("retained_angle",float('-inf'))<2*math.pi
            or info.get("palm_support_violation") is not False
            or info.get("failure_reason") is not None
            or not np.isfinite(info.get("peak_palm_support_force",float('nan')))
            or not 0<=info.get("peak_palm_support_force",float('nan'))<=self.ec.palm_support_force_tolerance
            or info.get("finger_contacts",0)<2
            or info.get("axis_tilt_radians",float('inf'))>math.radians(self.ec.success_axis_tolerance_degrees)):
            raise ValueError("primitive trajectories require a real completed 2*pi rotation without a drop")
        self.successful_traces.append(np.asarray(trace, dtype=np.float64))
        self.success_records.append(dict(source=source, environment_step=self.environment_steps,
                                        angle=float(info["angle"]), return_value=episode_return,
                                        steps=episode_steps, frames=len(trace),
                                        online_updates=self.agent.online_updates, success=True, drop=False,
                                        finger_contacts=info["finger_contacts"],axis_tilt_radians=info["axis_tilt_radians"],
                                        retained_angle=info["retained_angle"],
                                        palm_support_violation=False,
                                        peak_palm_support_force=float(info["peak_palm_support_force"]),
                                        failure_reason=None))

    def evaluate(self):
        results = []
        self.evaluations += 1
        for episode in range(5):
            if self.stop_requested:
                break
            obs, _ = self.evaluation_env.reset(seed=self.args.seed + 1_000_000 + episode)
            trace = [self.evaluation_env.data.qpos[self.evaluation_env.q_ids].copy()]
            total = 0.
            info = {"success": False, "drop": False, "angle": 0.}
            for _ in range(self.ec.horizon_steps):
                if self.stop_requested:
                    break
                obs, reward, terminated, truncated, info = self.evaluation_env.step(self.agent.act(obs, True))
                total += reward
                trace.append(self.evaluation_env.data.qpos[self.evaluation_env.q_ids].copy())
                if terminated or truncated:
                    break
            if self.stop_requested:
                break  # An interrupted evaluation is never a completed episode.
            success = bool(info["success"] and not info["drop"] and info["angle"] >= 2 * math.pi)
            result = dict(episode=episode, success=success, drop=bool(info["drop"]),
                          palm_support_violation=info['palm_support_violation'],
                          peak_palm_support_force=info['peak_palm_support_force'],
                          failure_reason=info['failure_reason'],
                          angle=float(info["angle"]), retained_angle=float(info['retained_angle']),
                          finger_contacts=info['finger_contacts'],axis_tilt_radians=info['axis_tilt_radians'],
                          return_value=total, steps=self.evaluation_env.steps)
            results.append(result)
            if success:
                self.eval_successes += 1
                self.record_success(trace, info, "deterministic evaluation", total, self.evaluation_env.steps)
        self.log("evaluation", evaluation=self.evaluations, episodes=len(results), required_episodes=5,
                 complete=len(results) == 5, successes=sum(row["success"] for row in results), results=results)
        if len(results) < 5:
            self.next_eval = self.environment_steps  # Retry interrupted evaluation after resuming.

    def run(self):
        try:
            while self.environment_steps < self.args.steps and not self.stop_requested:
                active = min(self.args.num_envs, self.args.steps - self.environment_steps)
                actions = self.agent.act(np.asarray(self.observations[:active]))
                if self.pool:
                    futures = [self.pool.submit(env.step, action) for env, action in zip(self.envs[:active], actions)]
                    results = [future.result() for future in futures]
                else:
                    results = [self.envs[0].step(actions[0])]
                for i, (next_obs, reward, terminated, truncated, info) in enumerate(results):
                    self.buffers[i].add_transition(self.observations[i], actions[i], reward, next_obs, terminated, truncated)
                    self.observations[i] = next_obs
                    self.episode_returns[i] += reward
                    self.joint_traces[i].append(self.envs[i].data.qpos[self.envs[i].q_ids].copy())
                    self.environment_steps += 1
                    if terminated or truncated:
                        self.completed_episodes += 1
                        self.drops += int(info["drop"])
                        success = bool(info["success"] and not info["drop"] and info["angle"] >= 2 * math.pi)
                        self.successes += int(success)
                        if success:
                            self.record_success(self.joint_traces[i], info, "stochastic training",
                                                self.episode_returns[i], self.envs[i].steps)
                        self.log("episode", environment=i, episode=self.completed_episodes, success=success,
                                 drop=bool(info["drop"]), angle=float(info["angle"]),
                                 palm_support_violation=info['palm_support_violation'],
                                 peak_palm_support_force=info['peak_palm_support_force'],
                                 failure_reason=info['failure_reason'],
                                 retained_angle=float(info['retained_angle']),
                                 finger_contacts=info['finger_contacts'],axis_tilt_radians=info['axis_tilt_radians'],
                                 return_value=self.episode_returns[i], steps=self.envs[i].steps)
                        self.observations[i], _ = self.envs[i].reset()
                        self.episode_returns[i] = 0.
                        self.joint_traces[i] = [self.envs[i].data.qpos[self.envs[i].q_ids].copy()]
                if self.environment_steps >= self.args.learning_starts:
                    for _ in range(active * self.args.updates_per_step):
                        self.latest_metrics = self.agent.update(sample_replay(self.buffers, self.args.batch_size, self.rng))
                        if self.stop_requested:
                            break
                if self.environment_steps >= self.next_log:
                    self.next_log = (self.environment_steps // self.args.log_every + 1) * self.args.log_every
                    self.log("training", completed_episodes=self.completed_episodes, drops=self.drops,
                             successes=self.successes, replay_transitions=sum(map(len, self.buffers)), **self.latest_metrics)
                if self.environment_steps >= self.next_eval and not self.stop_requested:
                    self.next_eval = (self.environment_steps // self.args.eval_every + 1) * self.args.eval_every
                    self.evaluate()
                if self.environment_steps >= self.next_checkpoint and not self.stop_requested:
                    self.next_checkpoint = (self.environment_steps // self.args.checkpoint_every + 1) * self.args.checkpoint_every
                    self.checkpoint("running")
            status = "interrupted" if self.stop_requested else "budget_exhausted"
            self.checkpoint(status)
            self.log("stopped", status=status, completed_episodes=self.completed_episodes,
                     successful_trajectory_episodes=len(self.success_records), online_updates=self.agent.online_updates)
            return 130 if self.stop_requested else 0
        except BaseException:
            # A partially completed batch may have advanced some environments;
            # only the last committed generation is safe after such an error.
            # Normal SIGTERM/SIGINT set a flag and commit at a consistent boundary.
            self.log("failed", resume_from_last_committed_checkpoint=True)
            raise
        finally:
            self.close()

    def close(self):
        if self.pool:
            self.pool.shutdown(wait=True)
        for env in self.envs:
            env.close()
        self.evaluation_env.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", default=str(ROOT / "configs/local_sim.json"))
    result.add_argument("--steps", type=int, required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--device", default="cpu")
    result.add_argument("--num-envs", type=int, default=8)
    result.add_argument("--eval-every", type=int, default=10_000)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--batch-size", type=int, default=256)
    result.add_argument("--learning-starts", type=int, default=1024)
    result.add_argument("--updates-per-step", type=int, default=1)
    result.add_argument("--replay-capacity", type=int, default=1_000_000)
    result.add_argument("--checkpoint-every", type=int, default=5000)
    result.add_argument("--log-every", type=int, default=100)
    result.add_argument("--torch-threads", type=int, default=1)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    torch.set_num_threads(args.torch_threads)
    runner = BootstrapRunner(args)
    previous = {}

    def request_stop(signum, frame):
        runner.stop_requested = True

    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.signal(signum, request_stop)
    try:
        return runner.run()
    except KeyboardInterrupt:
        # Covers a manually raised KeyboardInterrupt outside normal signal
        # delivery. The committed generation remains the resumable authority.
        return 130
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
