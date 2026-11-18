"""Contact-dynamics Allegro rotation environment for the independent reproduction.

The stock Menagerie robot is mounted palm down; the object is free and has no
weld, supporting joint, applied torque, or scripted motion. All choices absent
from the paper are exposed in EnvConfig and described in reproduction.md.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import math
import xml.etree.ElementTree as ET

import gymnasium as gym
import mujoco
import numpy as np

ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets" / "allegro"
JOINT_NAMES = [f"{f}j{j}" for f in ("ff", "mf", "rf", "th") for j in range(4)]
ROTATION_VALIDATION = 'incremental-principal-twist-with-grasp-retention-no-palm-v3'


@dataclass
class EnvConfig:
    object_shape: str = "hexagonal"
    object_diameter: float = 0.045  # across flats for hexagonal, diameter for circular
    object_length: float = 0.12
    object_mass: float = 0.15
    object_friction: float = 0.5
    finger_friction: float = 0.8
    timestep: float = 0.002
    control_seconds: float = 0.1
    horizon_steps: int = 900
    terminate_on_success: bool = True
    joint_kp: float = 3.0
    joint_kv: float = 0.1
    max_joint_force: float = 0.7
    contact_loss_seconds: float = 0.05
    success_axis_tolerance_degrees: float = 30.0
    palm_support_force_tolerance: float = 1e-6
    palm_height: float = 0.35
    initial_object_position: tuple[float, float, float] = (0.055, 0.0, 0.075)
    initial_hand_q: tuple[float, ...] = (
        0.0, 0.7, 0.7, 0.5, 0.0, 0.7, 0.7, 0.5,
        0.0, 0.7, 0.7, 0.5, 1.2, 0.3, 0.5, 0.5,
    )
    initial_grasp: str | None = None
    task: str = "rotation"
    goal_heading: float | None = None
    render_width: int = 640
    render_height: int = 480


def _values(values):
    return " ".join(f"{x:.12g}" for x in values)


def make_model(config: EnvConfig) -> mujoco.MjModel:
    if config.object_shape not in {"hexagonal", "circular"}:
        raise ValueError("Object shape must be hexagonal or circular")
    if config.task not in {"rotation", "reorientation"}:
        raise ValueError("Task must be rotation or reorientation")
    if min(config.object_diameter, config.object_length, config.object_mass,
           config.timestep, config.control_seconds) <= 0:
        raise ValueError("Object dimensions, mass, and timesteps must be positive")
    if not np.isfinite(config.palm_support_force_tolerance) or config.palm_support_force_tolerance < 0:
        raise ValueError('Palm support force tolerance must be finite and nonnegative')
    ratio = config.control_seconds / config.timestep
    if not np.isclose(ratio, round(ratio)):
        raise ValueError("Control interval must be an integer number of physics steps")
    tree = ET.parse(ASSET_ROOT / "right_hand.xml")
    root = tree.getroot()
    root.find("compiler").set("meshdir", str(ASSET_ROOT / "assets"))
    option = root.find("option")
    option.set("timestep", str(config.timestep))
    option.set("integrator", "implicitfast")
    option.set("gravity", "0 0 -9.81")
    option.set("iterations", "50")
    default = root.find("default/default")
    default.find("joint").set("damping", str(config.joint_kv))
    servo = default.find("position")
    servo.set("kp", str(config.joint_kp))
    servo.set("forcerange", _values((-config.max_joint_force, config.max_joint_force)))
    collision = default.find("default[@class='collision']/geom")
    collision.set("friction", _values((config.finger_friction, 0.005, 0.0001)))
    collision.set("condim", "4")
    world = root.find("worldbody")
    palm = world.find("body[@name='palm']")
    palm.set("pos", _values((0, 0, config.palm_height)))
    palm.set("quat", "0.7071067811865476 0 0.7071067811865476 0")
    for body_name, tip_z in (("ff_tip", .019), ("mf_tip", .019), ("rf_tip", .019), ("th_tip", .035)):
        ET.SubElement(palm.find(f".//body[@name='{body_name}']"), "site", name=body_name+"_site", pos=f"0 0 {tip_z}", size=".003")
    ET.SubElement(world, "light", pos="0.3 -0.4 1", dir="-0.3 0.4 -1")
    ET.SubElement(world, "geom", name="table", type="plane", size="1 1 .02", rgba=".18 .2 .23 1", friction=".5 .005 .0001")
    obj = ET.SubElement(world, "body", name="object")
    ET.SubElement(obj, "freejoint", name="object_joint")
    geom_args = dict(name="object_geom", mass=str(config.object_mass), rgba=".3 .8 .2 1", friction=_values((config.object_friction,.005,.0001)), condim="4")
    if config.object_shape == "hexagonal":
        radius = config.object_diameter / math.sqrt(3)
        vertices = [(radius*math.cos(a), y, radius*math.sin(a))
                    for y in (-config.object_length/2,config.object_length/2)
                    for a in np.arange(6)*math.pi/3]
        ET.SubElement(root.find("asset"), "mesh", name="hex_object", vertex=_values(np.array(vertices).ravel()))
        ET.SubElement(obj, "geom", type="mesh", mesh="hex_object", **geom_args)
    else:
        ET.SubElement(obj, "geom", type="cylinder", size=_values((config.object_diameter/2,config.object_length/2)), quat="0.7071067811865476 0.7071067811865476 0 0", **geom_args)
    # A visible stripe on the body, with no collision or mass contribution.
    ET.SubElement(obj, "site", name="rotation_marker", pos=_values((config.object_diameter/2,0,0)), type="sphere", size=".004", rgba="1 .2 .1 1")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def relative_quaternion(parent, child):
    inv = np.empty(4); result = np.empty(4)
    mujoco.mju_negQuat(inv, parent)
    mujoco.mju_mulQuat(result, inv, child)
    return result


def principal_angle(quaternion):
    """Projection of object x onto palm x/z; rotation about palm +y."""
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quaternion)
    return math.atan2(-matrix[6], matrix[0])


def wrap_angle(value):
    return (value + math.pi) % (2 * math.pi) - math.pi


def axial_increment(previous_quaternion, current_quaternion):
    """Body-principal-y twist increment, without projected Euler singularities.

    With 2 ms transitions the incremental quaternion has a unique short arc.
    Swing about perpendicular axes contributes no twist about the rod axis.
    """
    delta=relative_quaternion(previous_quaternion,current_quaternion)
    if delta[0]<0:delta=-delta
    return 2*math.atan2(delta[2],delta[0])


class AllegroRotationEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, config=None, render_mode=None):
        self.config = config or EnvConfig()
        self.model = make_model(self.config)
        self.data = mujoco.MjData(self.model)
        self.joint_ids = np.array([self.model.joint(n).id for n in JOINT_NAMES])
        self.q_ids = self.model.jnt_qposadr[self.joint_ids]
        self.v_ids = self.model.jnt_dofadr[self.joint_ids]
        self.object_qadr = int(self.model.joint("object_joint").qposadr[0])
        self.object_dadr = int(self.model.joint("object_joint").dofadr[0])
        self.palm_id = self.model.body("palm").id
        self.object_id = self.model.body("object").id
        self.object_geom = self.model.geom("object_geom").id
        self.table_geom = self.model.geom("table").id
        self.low = self.model.actuator_ctrlrange[:,0].copy()
        self.high = self.model.actuator_ctrlrange[:,1].copy()
        self.action_space = gym.spaces.Box(-1,1,shape=(16,),dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf,np.inf,shape=(60 if self.config.task=="rotation" else 61,),dtype=np.float32)
        self.render_mode = render_mode
        self._renderer = None
        self.substeps = round(self.config.control_seconds / self.config.timestep)
        self._episode_over = True

    def to_joint_targets(self, action):
        action = np.asarray(action)
        if action.shape[-1:] != (16,) or not np.isfinite(action).all():
            raise ValueError("Action must be finite with last dimension 16")
        if np.any(np.abs(action)>1.000001):
            raise ValueError("Normalized action outside [-1,1]")
        return self.low + (np.clip(action,-1,1)+1)*0.5*(self.high-self.low)

    def to_action(self, joint_target):
        return np.clip(2*(np.asarray(joint_target)-self.low)/(self.high-self.low)-1,-1,1).astype(np.float32)

    def object_position(self, data=None):
        d = self.data if data is None else data
        return d.xmat[self.palm_id].reshape(3,3).T @ (d.xpos[self.object_id]-d.xpos[self.palm_id])

    def object_quaternion(self, data=None):
        d = self.data if data is None else data
        return relative_quaternion(d.xquat[self.palm_id],d.xquat[self.object_id])

    def dropped(self, data=None):
        d = self.data if data is None else data
        for contact in d.contact:
            if {int(contact.geom1),int(contact.geom2)} == {self.object_geom,self.table_geom}:
                return True
        # Explicit reproduction choice: stop once the object leaves the hand workspace.
        return bool(np.linalg.norm(self.object_position(d)-self.initial_position)>0.18)

    def finger_contacts(self, data=None):
        d=self.data if data is None else data
        fingers=set()
        for contact in d.contact:
            if self.object_geom not in (contact.geom1,contact.geom2):continue
            other=contact.geom2 if contact.geom1==self.object_geom else contact.geom1
            body=int(self.model.geom_bodyid[other])
            name=mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_BODY,body) or ''
            if name[:2] in {'ff','mf','rf','th'}:fingers.add(name[:2])
        return len(fingers)

    def palm_support_force(self, data=None):
        """Sum positive object--palm normal forces in the forwarded state, in N."""
        d = self.data if data is None else data
        total = 0.0
        force = np.empty(6)
        for index, contact in enumerate(d.contact):
            if self.object_geom not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == self.object_geom else contact.geom1
            if self.model.geom_bodyid[other] == self.palm_id:
                mujoco.mj_contactForce(self.model, d, index, force)
                total += max(float(force[0]), 0.0)
        return total

    def _record_palm_support(self):
        force = self.palm_support_force()
        self._peak_palm_support_force = max(self._peak_palm_support_force, force)
        # The source requirement is specific to rotation; reorientation keeps
        # its existing task semantics while still exposing measured palm force.
        if self.config.task == 'rotation' and force > self.config.palm_support_force_tolerance:
            self._palm_support_violation = True

    def axis_tilt(self):
        current=np.empty(9);initial=np.empty(9)
        mujoco.mju_quat2Mat(current,self.object_quaternion())
        mujoco.mju_quat2Mat(initial,self.initial_orientation)
        return math.acos(np.clip(current.reshape(3,3)[:,1] @ initial.reshape(3,3)[:,1],-1,1))

    def _observation(self, qdot, pdot):
        obs = np.concatenate((self.data.qpos[self.q_ids], self.object_position(),
            self.object_quaternion(), [self.angle,self.previous_angle],qdot,pdot,self.last_action))
        if self.config.task=="reorientation": obs=np.append(obs,self.goal)
        return obs.astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model,self.data)
        self.data.qpos[self.q_ids]=self.config.initial_hand_q
        self.data.ctrl[:]=self.data.qpos[self.q_ids]
        mujoco.mj_forward(self.model,self.data)
        p = self.data.xpos[self.palm_id]+self.data.xmat[self.palm_id].reshape(3,3) @ np.array(self.config.initial_object_position)
        qa=self.object_qadr
        self.data.qpos[qa:qa+3]=p
        self.data.qpos[qa+3:qa+7]=self.data.xquat[self.palm_id]
        if self.config.initial_grasp:
            with np.load(self.config.initial_grasp,allow_pickle=False) as grasp:
                if grasp['qpos'].shape != self.data.qpos.shape: raise ValueError('Grasp state dimension mismatch')
                self.data.qpos[:]=grasp['qpos']
                self.data.qvel[:]=grasp['qvel']
                self.data.ctrl[:]=grasp['ctrl']
        mujoco.mj_forward(self.model,self.data)
        self.initial_position=self.object_position().copy()
        self.initial_orientation=self.object_quaternion().copy()
        self.start_angle=principal_angle(self.initial_orientation)
        self._last_wrapped=self.start_angle
        self._last_object_quaternion=self.object_quaternion().copy()
        self._contactless_seconds=0.0
        self._palm_support_violation=False
        self._peak_palm_support_force=0.0
        self._record_palm_support()
        self.angle=0.0; self.previous_angle=0.0; self.steps=0; self.rotations=0; self.dwell=0
        self.retained_angle=0.0;self.previous_retained_angle=0.0
        self.retained_positive_step=0.0;self.retained_negative_step=0.0
        self.goal=float(self.np_random.uniform(-np.pi/9,np.pi/9) if self.config.goal_heading is None else self.config.goal_heading)
        self.heading_error=abs(self.goal)
        self._goal_rewarded=False
        self.last_action=self.to_action(self.data.ctrl)
        self._episode_over=False
        self.last_q=self.data.qpos[self.q_ids].copy();self.last_p=self.object_position().copy()
        return self._observation(np.zeros(16),np.zeros(3)), self._info(False,False)

    def _info(self,drop,success):
        return {"angle": self.angle, "rotations":self.rotations,"drop":drop,"success":success,
                "palm_support_force":self.palm_support_force(),
                "peak_palm_support_force":self._peak_palm_support_force,
                "palm_support_violation":self._palm_support_violation,
                "failure_reason":('palm_support' if self._palm_support_violation else 'drop' if drop else None),
                "retained_angle":self.retained_angle,
                "retained_positive_step":self.retained_positive_step,
                "retained_negative_step":self.retained_negative_step,
                "finger_contacts":self.finger_contacts(),"axis_tilt_radians":self.axis_tilt(),
                "sim_time": self.steps*self.config.control_seconds,"object_position":self.object_position().copy(),
                "goal_heading":self.goal if self.config.task=="reorientation" else None}

    def step(self, action):
        if self._episode_over: raise RuntimeError('Call reset after episode termination')
        targets=self.to_joint_targets(action)
        if targets.shape!=(16,): raise ValueError('step requires one action')
        self.data.ctrl[:]=targets
        self.previous_angle=self.angle
        self.previous_retained_angle=self.retained_angle
        self.retained_positive_step=0.0;self.retained_negative_step=0.0
        drop=False
        # Track angle and constraints each physics transition to avoid wrap aliasing.
        for _ in range(self.substeps):
            mujoco.mj_step(self.model,self.data)
            mujoco.mj_forward(self.model,self.data)
            self._record_palm_support()
            quaternion=self.object_quaternion()
            increment=axial_increment(self._last_object_quaternion,quaternion)
            self.angle += increment
            self._last_object_quaternion=quaternion.copy()
            self._last_wrapped=principal_angle(quaternion)
            # Full-turn validation must not accumulate released/tumbling spins.
            # Count every reverse increment, but positive progress only while
            # the grasp is retained at this physics transition.
            retained=(not self._palm_support_violation and self.finger_contacts()>=2 and
                      self.axis_tilt()<=math.radians(self.config.success_axis_tolerance_degrees))
            qualified=increment if increment<0 or retained else 0.0
            self.retained_angle+=qualified
            self.retained_positive_step+=max(qualified,0.0)
            self.retained_negative_step+=min(qualified,0.0)
            self._contactless_seconds=(self._contactless_seconds+self.config.timestep
                                       if self.finger_contacts()==0 else 0.0)
            if self._contactless_seconds>=self.config.contact_loss_seconds:drop=True
            if self.dropped(): drop=True
        self.steps+=1
        nrot=max(0,int(min(self.angle,self.retained_angle)//(2*math.pi)))
        held=self.finger_contacts()>=2 and self.axis_tilt()<=math.radians(self.config.success_axis_tolerance_degrees)
        success=nrot>self.rotations and held and not self._palm_support_violation
        reward=20*(self.angle-self.previous_angle)-.3 + 200*success
        if success:self.rotations=max(self.rotations,nrot)
        if self.config.task=="reorientation":
            # About palm z, perpendicular to the object's principal y axis.
            rel=relative_quaternion(self.initial_orientation,self.object_quaternion())
            mat=np.empty(9);mujoco.mju_quat2Mat(mat,rel)
            heading=math.atan2(mat[3],mat[0])
            prev_error=getattr(self,'heading_error',abs(self.goal))
            self.heading_error=abs(wrap_angle(self.goal-heading))
            self.dwell=self.dwell+1 if self.heading_error<=math.radians(5) else 0
            success=self.dwell>=round(.5/self.config.control_seconds)
            reward=20*(prev_error-self.heading_error)-.3+200*(success and not self._goal_rewarded)
            if success:self._goal_rewarded=True
        # Applying the existing failure penalty to palm support is an explicit
        # local choice; support is a task violation, not an object drop.
        failed=drop or self._palm_support_violation
        if failed: reward=-50.0;success=False
        terminated=bool(failed or (success and self.config.terminate_on_success))
        truncated=bool(self.steps>=self.config.horizon_steps and not terminated)
        dt=self.config.control_seconds
        q=self.data.qpos[self.q_ids].copy();p=self.object_position().copy()
        self.last_action=np.asarray(action,dtype=np.float32).copy()
        obs=self._observation((q-self.last_q)/dt,(p-self.last_p)/dt)
        self.last_q=q;self.last_p=p;self._episode_over=terminated or truncated
        return obs,float(reward),terminated,truncated,self._info(drop,success)

    def render(self):
        if self._renderer is None:
            self._renderer=mujoco.Renderer(self.model,height=self.config.render_height,width=self.config.render_width)
        camera=mujoco.MjvCamera()
        camera.lookat[:]=[.05,0,self.config.palm_height-.035]
        camera.distance=.48;camera.azimuth=135;camera.elevation=-20
        self._renderer.update_scene(self.data,camera=camera)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:self._renderer.close();self._renderer=None
