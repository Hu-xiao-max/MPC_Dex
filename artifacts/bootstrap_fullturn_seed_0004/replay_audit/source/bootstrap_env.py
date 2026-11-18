"""Explicit training choices for the unpublished prerequisite simulation policy.

The companion paper says its primitive comes from simulation SAC, but does not
supply that policy's reward or control preprocessing. This optional environment
uses bounded joint-target increments and a retained-rotation shaping reward.
It does not change the main MPC-scaffolded learner's reward or action mapping.
"""
from dataclasses import asdict, dataclass
import math
from pathlib import Path

import numpy as np

from .env import AllegroRotationEnv, EnvConfig


@dataclass(frozen=True)
class BootstrapControlConfig:
    action_mode: str = 'joint_target_increment'
    max_delta_radians: float = .02
    max_target_error_radians: float = .15
    retained_progress_coefficient: float = 20.
    max_rewarded_progress_radians: float = .1
    survival_reward: float = .02
    unretained_penalty: float = -.2
    drop_penalty: float = -20.
    tilt_penalty: float = 2.
    position_penalty: float = 100.
    action_penalty: float = .002
    success_bonus: float = 200.

    def __post_init__(self):
        if self.action_mode!='joint_target_increment':raise ValueError('Unknown bootstrap action mode')
        if not 0<self.max_delta_radians<=self.max_target_error_radians:
            raise ValueError('Invalid incremental joint-target bounds')
        if self.max_rewarded_progress_radians<=0 or self.drop_penalty>=0:
            raise ValueError('Invalid bootstrap reward settings')


class BootstrapRotationEnv(AllegroRotationEnv):
    """SAC actions are increments, observations retain actual executed targets.

    Since the previous executed target is already observed, the increment
    controller's state is observable. Data.ctrl is its only extra persistent
    state and is included in MuJoCo integration checkpoints. The two angle
    slots expose retained progress rather than raw twist so the policy can
    observe the progress that determines its full-turn completion.
    """
    def __init__(self, config: EnvConfig, bootstrap: BootstrapControlConfig, render_mode=None):
        self.bootstrap=bootstrap
        super().__init__(config,render_mode)

    def _observation(self, qdot, pdot):
        observation=super()._observation(qdot,pdot)
        observation[23:25]=[self.retained_angle,self.previous_retained_angle]
        return observation

    def joint_targets_for_increment(self, action):
        action=np.asarray(action,dtype=np.float64)
        if action.shape!=(16,) or not np.isfinite(action).all() or np.any(np.abs(action)>1.000001):
            raise ValueError('Bootstrap action must be a finite normalized 16-vector')
        current=self.data.ctrl.copy()
        measured=self.data.qpos[self.q_ids]
        # Intersection with the reachable target interval prevents winding up a
        # large spring error when a finger is blocked by the object.
        target=np.clip(current+self.bootstrap.max_delta_radians*np.clip(action,-1,1),
                       measured-self.bootstrap.max_target_error_radians,
                       measured+self.bootstrap.max_target_error_radians)
        # The rate limit remains absolute even when an existing target lags.
        target=np.clip(target,current-self.bootstrap.max_delta_radians,current+self.bootstrap.max_delta_radians)
        return np.clip(target,self.low,self.high)

    def step(self, action):
        target=self.joint_targets_for_increment(action)
        executed=self.to_action(target)
        obs,paper_reward,terminated,truncated,info=super().step(executed)
        retained=(not info['palm_support_violation'] and info['finger_contacts']>=2 and
                  info['axis_tilt_radians']<=math.radians(self.config.success_axis_tolerance_degrees))
        c=self.bootstrap
        # Only positive, physically retained progress is capped. All reverse
        # motion remains penalized, so forward/backward cycles cannot harvest
        # reward through clipping or brief loss of contact.
        rewarded_progress=(min(info['retained_positive_step'],c.max_rewarded_progress_radians)
                           +info['retained_negative_step'])
        if info['drop'] or info['palm_support_violation']:
            # Share the configured failure penalty while preserving the
            # environment's distinct palm-support versus drop diagnosis.
            reward=c.drop_penalty
        else:
            displacement=self.object_position()-self.initial_position
            reward=(c.retained_progress_coefficient*rewarded_progress
                    +(c.survival_reward if retained else c.unretained_penalty)
                    -c.tilt_penalty*info['axis_tilt_radians']**2
                    -c.position_penalty*float(displacement@displacement)
                    -c.action_penalty*float(np.mean(np.asarray(action)**2))
                    +c.success_bonus*int(info['success']))
        info=dict(info,paper_reward=paper_reward,bootstrap_reward=reward,
                  rewarded_progress_radians=rewarded_progress,retained=retained,
                  executed_normalized_action=executed.copy())
        return obs,float(reward),terminated,truncated,info


def create_bootstrap_environment(env_config, settings):
    control=settings.get('control')
    return (BootstrapRotationEnv(env_config,BootstrapControlConfig(**control))
            if control else AllegroRotationEnv(env_config))


def policy_interface(settings):
    control=settings.get('control')
    if not control:return {'kind':'absolute_joint_target'}
    from .contracts import sha256
    return {'kind':'bootstrap_joint_increment','control':asdict(BootstrapControlConfig(**control)),
            'observation_angles':'retained_angle,previous_retained_angle at slots 23,24; previous action remains executed absolute target',
            'implementation_sha256':sha256(Path(__file__))}


def environment_for_policy(env_config, interface):
    interface=interface or {'kind':'absolute_joint_target'}
    if interface['kind']=='absolute_joint_target':return AllegroRotationEnv(env_config)
    if interface['kind']!='bootstrap_joint_increment':raise ValueError('Unknown checkpoint action interface')
    from .contracts import sha256
    if interface.get('implementation_sha256')!=sha256(Path(__file__)):
        raise ValueError('Bootstrap action/reward implementation changed since checkpoint')
    return BootstrapRotationEnv(env_config,BootstrapControlConfig(**interface['control']))
