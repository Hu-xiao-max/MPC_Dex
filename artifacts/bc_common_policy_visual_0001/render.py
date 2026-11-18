"""Render exact replay of recorded policy commands, including the later drop."""
from pathlib import Path
import json
import sys

D = Path(__file__).resolve().parent
ROOT = D.parents[1]
SOURCE = ROOT/'artifacts/bc_common_continuation_0001'
sys.path.insert(0, str(SOURCE/'source'))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from mpc_dex.env import AllegroRotationEnv
from mpc_dex.cli import read_config
from mpc_dex.contracts import sha256
from scripts.audit_policy import PhysicsObserver
from dataclasses import replace

trace = SOURCE/'head_polish/independent_audit_continuous_policy_2s/transitions.npz'
with np.load(trace, allow_pickle=False) as data:
    actions, qpos, qvel = (data[k].copy() for k in ('actions','qpos','qvel'))
_, ec = read_config(SOURCE/'strict_config.json')
env = AllegroRotationEnv(replace(ec, terminate_on_success=False))
env.reset(seed=42)
observer = PhysicsObserver(env)
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
frames = []

def frame(step):
    scene = Image.fromarray(env.render())
    out = Image.new('RGB',(scene.width,scene.height+76),'#101820')
    out.paste(scene,(0,76)); draw=ImageDraw.Draw(out)
    state = 'FULL TURN' if step==9 else 'DROPPED' if step==12 else 'continued policy' if step>9 else 'turning'
    draw.text((12,8),f'Recorded neural policy: {state}',font=font,fill='white')
    draw.text((12,33),f'{step/10:.1f} s | force-qualified turns {observer.force_qualified_angle/(2*np.pi):.3f}',font=small,fill='#c9d5df')
    draw.text((12,53),'Independent simulation; BC + head correction, no SAC. Playback slowed 2.5x.',font=small,fill='#ffc078')
    return out

try:
    frames.append(frame(0))
    for i, action in enumerate(actions):
        _, _, terminated, truncated, info = env.step(action)
        np.testing.assert_array_equal(env.data.qpos,qpos[i])
        np.testing.assert_array_equal(env.data.qvel,qvel[i])
        assert not (terminated or truncated) or i==len(actions)-1
        frames.append(frame(i+1))
    with np.load(SOURCE/'head_polish/independent_audit_continuous_policy_2s/physics.npz',allow_pickle=False) as data:
        np.testing.assert_array_equal(np.asarray(observer.rows),data['samples'])
    assert len(actions)==12 and info['drop'] and not info['palm_support_violation']
    frames[9].save(D/'policy_first_turn.png')
    frames[-1].save(D/'continued_policy_drop.png')
    frames[0].save(D/'policy_turn_then_drop.gif',save_all=True,append_images=frames[1:],duration=[250]*12+[1000],loop=0)
    (D/'manifest.json').write_text(json.dumps(dict(
        purpose='Exact physics replay of recorded neural-policy commands; first-turn success followed by actual policy drop',
        checkpoint_sha256=sha256(SOURCE/'head_polish/polished.pt'),
        source_transitions_sha256=sha256(trace),script_sha256=sha256(__file__),
        environment_source_sha256=sha256(SOURCE/'source/mpc_dex/env.py'),
        all_qpos_qvel_and_physics_samples_bitexact=True,frames=len(frames),physics_seconds=1.2,
        normal_frame_display_ms=250,final_frame_display_ms=1000,production_eligible=False,
        first_success_boundary_seconds=.9,actual_drop_seconds=1.2),indent=2)+'\n')
finally:
    observer.close();env.close()
