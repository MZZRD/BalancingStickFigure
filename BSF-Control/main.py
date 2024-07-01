import time
import numpy as np
import mujoco
from mujoco._structs import MjModel, MjData

from model import get_model_xml

# Load the model and data
model_xml: str = get_model_xml()
model: MjModel = mujoco.MjModel.from_xml_string(model_xml)
data: MjData = mujoco.MjData(model)

# Set the initial state
mujoco.mj_resetDataKeyframe(model, data, 0)

# Simulation parameters
duration:int = 3
time_scale: float = 0.1

# LQR controller variables
ctrl0 = [ 0.01812103, 0.01812103, -0.00049134, 0.01812103]

with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
  # Set camera options
  viewer.cam.distance = 0.5
  viewer.cam.elevation = -15

  # Close the viewer automatically after a duration
  start = time.time()
  while viewer.is_running() and time.time() - start < duration/time_scale:
    step_start = time.time()

    # LQR control law.
    data.ctrl = ctrl0

    # Step the simulation
    mujoco.mj_step(model, data)

    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      viewer.cam.lookat = data.body('torso').xpos
      
    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = (1/time_scale)*model.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)