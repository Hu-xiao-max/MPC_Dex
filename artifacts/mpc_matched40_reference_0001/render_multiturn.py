"""Render and verify a physics replay of the recorded MPC controls."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from mpc_dex.cli import read_config
from mpc_dex.env import AllegroRotationEnv
from mpc_dex.contracts import sha256
from scripts.audit_policy import PhysicsObserver
HERE=Path(__file__).resolve().parent
_,config=read_config(HERE/'source/config.json')
with np.load(HERE/'transitions.npz',allow_pickle=False) as data:
    original=data['actions'].copy();qpos=data['qpos'].copy();qvel=data['qvel'].copy()
actions=np.vstack([original,np.repeat(original[-1:],20,axis=0)])
env=AllegroRotationEnv(config);env.reset();observer=PhysicsObserver(env)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
frames=[]
def frame(step):
    scene=Image.fromarray(env.render());out=Image.new('RGB',(scene.width,scene.height+72),'#101820');out.paste(scene,(0,72))
    draw=ImageDraw.Draw(out)
    draw.text((12,8),'Independent simulation: three turns, then a two-second hold',font=font,fill='white')
    draw.text((12,33),f'{step/10:.1f} s | qualified turns {observer.force_qualified_angle/(2*np.pi):.3f} | 40 mm object',font=small,fill='#c9d5df')
    draw.text((12,51),'MPC diagnostic; no palm/table support. Not a learned policy or hardware result.',font=small,fill='#ffc078')
    return out
try:
    frames.append(frame(0))
    for i,action in enumerate(actions):
        _,_,done,truncated,info=env.step(action)
        if i<len(original):
            np.testing.assert_array_equal(env.data.qpos,qpos[i]);np.testing.assert_array_equal(env.data.qvel,qvel[i])
        if done or truncated:raise RuntimeError(f'Unexpected replay boundary at {i+1}')
        frames.append(frame(i+1))
    frames[-1].save(HERE/'multiturn_final.png')
    frames[0].save(HERE/'verified_three_turns.gif',save_all=True,append_images=frames[1:],duration=100,loop=0)
    (HERE/'render_manifest.json').write_text(json.dumps({'source_transition_sha256':sha256(HERE/'transitions.npz'),'renderer_sha256':sha256(__file__),'purpose':'Actual replay matches all60 recorded qpos/qvel states bit-for-bit, then holds20furthercommands; quantitative independent audit saved separately.','frames':len(frames),'frame_interval_seconds':.1,'final_force_qualified_angle':observer.force_qualified_angle,'final_drop':info['drop'],'final_palm_violation':info['palm_support_violation']},indent=2)+'\n')
finally:observer.close();env.close()
