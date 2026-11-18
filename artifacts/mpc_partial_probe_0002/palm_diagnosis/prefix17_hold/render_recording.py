"""Render previously recorded physical states; this does not generate motion."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import mujoco,numpy as np
from PIL import Image,ImageDraw,ImageFont
from mpc_dex.cli import read_config
from mpc_dex.env import AllegroRotationEnv
from mpc_dex.contracts import sha256
HERE=Path(__file__).resolve().parent
_,config=read_config(HERE/'config.json');env=AllegroRotationEnv(config)
with np.load(HERE/'transitions.npz',allow_pickle=False) as data:
    qpos=np.vstack((data['initial_qpos'],data['qpos']))
    qvel=np.vstack((data['initial_qvel'],data['qvel']))
with np.load(HERE/'strict_audit/physics.npz',allow_pickle=False) as data:
    physics=data['samples'];columns=list(data['columns'])
angles=np.r_[0.,physics[49::50,columns.index('force_qualified_angle')]]
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
frames=[]
try:
    env.reset()
    for i,(position,velocity,angle) in enumerate(zip(qpos,qvel,angles)):
        env.data.qpos[:]=position;env.data.qvel[:]=velocity;mujoco.mj_forward(env.model,env.data)
        rendered=Image.fromarray(env.render());frame=Image.new('RGB',(rendered.width,rendered.height+72),'#101820')
        frame.paste(rendered,(0,72));draw=ImageDraw.Draw(frame)
        draw.text((12,8),'Independent simulation: verified finger-supported turn',font=font,fill='white')
        draw.text((12,33),f'Recorded state {i/10:.1f} s | qualified rotation {np.degrees(angle):.1f} deg | 45 mm object',font=small,fill='#c9d5df')
        draw.text((12,51),'Later fixed hold drops at 2.9 s; this is not sustained control or a learned policy.',font=small,fill='#ffc078')
        frames.append(frame)
    frames[-1].save(HERE/'verified_turn_final.png')
    frames[0].save(HERE/'verified_turn.gif',save_all=True,append_images=frames[1:],duration=100,loop=0)
    (HERE/'render_manifest.json').write_text(json.dumps({'source_transition_sha256':sha256(HERE/'transitions.npz'),'source_physics_sha256':sha256(HERE/'strict_audit/physics.npz'),'renderer_sha256':sha256(__file__),'purpose':'Visualization of saved physical states only; quantitative evidence comes from independent strict and hold audits.','frames':len(frames),'frame_interval_seconds':.1},indent=2)+'\n')
finally:env.close()
