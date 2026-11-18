"""Check local MuJoCo physics/rendering, not paper implementation readiness."""

import json
import os
from pathlib import Path
import platform
import subprocess
from datetime import datetime, timezone

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np


def main():
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Local simulation dependencies only; no paper controller or robot is deployed.",
        "python": platform.python_version(),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "render_backend": os.environ["MUJOCO_GL"],
        "cpu_logical_count": os.cpu_count(),
    }
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=True,
        )
        report["gpu"] = gpu.stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        report["gpu_query_error"] = str(error)

    # A falling sphere verifies the runtime; it is not a manipulation task.
    model = mujoco.MjModel.from_xml_string('''
        <mujoco><option timestep="0.002"/>
        <worldbody>
          <light pos="0 0 2"/>
          <geom type="plane" size="1 1 .1"/>
          <body pos="0 0 .5"><freejoint/>
            <geom type="sphere" size=".05" mass=".1" rgba="1 .2 .1 1"/>
          </body>
        </worldbody></mujoco>
    ''')
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data, nstep=500)
    physics_ok = bool(np.isfinite(data.qpos).all() and 0.04 < data.qpos[2] < 0.06)
    report["physics"] = {"passed": physics_ok, "steps": 500, "final_height_m": float(data.qpos[2])}
    try:
        with mujoco.Renderer(model, height=120, width=160) as renderer:
            renderer.update_scene(data)
            pixels = renderer.render()
            from OpenGL import GL
            renderer_name = GL.glGetString(GL.GL_RENDERER).decode()
            renderer_vendor = GL.glGetString(GL.GL_VENDOR).decode()
        render_ok = bool(pixels.shape == (120, 160, 3) and pixels.max() > pixels.min())
        report["rendering"] = {
            "passed": render_ok, "shape": list(pixels.shape),
            "renderer": renderer_name, "vendor": renderer_vendor,
        }
    except Exception as error:
        render_ok = False
        report["rendering"] = {"passed": False, "error": str(error)}
    report["runtime_ready"] = physics_ok and render_ok
    report["paper_deployed"] = False
    output = Path(__file__).resolve().parents[1] / "artifacts" / "runtime.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["runtime_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
