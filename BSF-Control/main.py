import time
import numpy as np
import mujoco
from model import get_mjmodel_xml

# Load the model and data
mjmodel_xml = get_mjmodel_xml()
mjmodel = mujoco.MjModel.from_xml_string(mjmodel_xml)
mjdata = mujoco.MjData(mjmodel)

# Set the initial state
# mujoco.mj_resetDataKeyframe(mjmodel, mjdata, 0)

# Launch the viewer
with mujoco.viewer.launch_passive(mjmodel, mjdata, show_left_ui=False, show_right_ui=False) as viewer:
  # Set camera options
  viewer.cam.distance = 1
  viewer.cam.elevation = -15

  # PID variables
  Kp = 300
  Kd = 10
  Ki = 0
  com_ref = mjdata.contact[0].pos[0]
  integral = 0
  error_prior = 0

  # Close the viewer automatically after a duration
  start = time.time()
  while viewer.is_running():
    step_start = time.time()

    # pid control law.
    error = com_ref - mjdata.subtree_com[1, 0]
    integral += error * mjmodel.opt.timestep
    derivative = (error - error_prior) / mjmodel.opt.timestep
    error_prior = error
    mjdata.ctrl = Kp*error + Ki*integral + Kd*derivative

    # Step the simulation
    mujoco.mj_step(mjmodel, mjdata)

    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      viewer.cam.lookat = mjdata.body('torso').xpos
      
    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = mjmodel.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)