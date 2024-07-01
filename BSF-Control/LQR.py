import time
import numpy as np
import scipy.linalg
import mujoco
from mujoco._structs import MjModel, MjData
from model import get_model_xml
from utils import calc_vertical_offset

# Simulation parameters
KEYFRAME: int = 0
DURATION:int = 3
TIME_SCALE: float = 1.0

####################
## Load the model ##
####################

# Load mujoco model and data
model_xml: str = get_model_xml()
model: MjModel = mujoco.MjModel.from_xml_string(model_xml)
data: MjData = mujoco.MjData(model)


#########################################################
## Finding the control setpoint using inverse dynamics ##
#########################################################

# Accurately position the model on the ground floor
offset = calc_vertical_offset(model=model, data=data, keyframe=KEYFRAME)

# Find the desired forces on the joints
mujoco.mj_resetDataKeyframe(model, data, KEYFRAME)
mujoco.mj_forward(model, data)
data.qacc = 0
data.qpos[2] += offset
qpos0 = data.qpos.copy()
mujoco.mj_inverse(model, data)
qfrc0 = data.qfrc_inverse.copy()
print(f'desired forces: {qfrc0}\n')

print(f"Initial qpos: {qpos0}")

# Find forces to be produced by the actuators, depends on moment arms
ctrl0 = np.atleast_2d(qfrc0) @ np.linalg.pinv(data.actuator_moment)
ctrl0 = ctrl0.flatten()  # Save the ctrl setpoint.
print(f'control setpoint: {ctrl0}\n')

# Check if forces produced by the actuators approach the desired forces
data.ctrl = ctrl0
mujoco.mj_forward(model, data)
print(f'actuator forces: {data.qfrc_actuator}\n')

###################################
## Choosing the Q and R matrices ##
###################################
# Q is constructed as a sum of two terms
# First, a balancing cost that will keep the CoM over the ground contact point.
# Second, a cost for joints moving away from their initial configuration.

# Cost coefficients.
BALANCE_COST = 1000
BALANCE_JOINT_COST = 1
OTHER_JOINT_COST = 1
ACTUATOR_COST = 1

# Setting up aliases
nu = model.nu # number of actuators.
nv = model.nv  # number of DoFs.

# Define the R matrix
R = ACTUATOR_COST * np.eye(nu)


# Get dof indices into relevant sets of joints.
root_dofs = range(6)
body_dofs = range(6, nv)
balance_dofs = model.joint('left_leg').dofadr
other_dofs = np.setdiff1d(body_dofs, balance_dofs)

# Construct the Qjoint matrix.
Qjoint = np.eye(nv)
Qjoint[root_dofs, root_dofs] *= 0  # Don't penalize free joint directly.
Qjoint[balance_dofs, balance_dofs] *= BALANCE_JOINT_COST
Qjoint[other_dofs, other_dofs] *= OTHER_JOINT_COST

# Construct the Q matrix for position DoFs.
# Qpos = BALANCE_COST * Qbalance + Qjoint
Qpos = Qjoint

# No explicit penalty for velocities.
Q = np.block([[Qpos, np.zeros((nv, nv))],
            [np.zeros((nv, 2*nv))]]) 
# Q += np.eye(Q.shape[0]) * 1e-6 # make Q pos-def


#####################################
## Computing the LQR gain matrix K ##
#####################################

# Set the initial state and control.
mujoco.mj_resetData(model, data)
data.ctrl = ctrl0
data.qpos = qpos0

# Allocate the A and B matrices, compute them.
A = np.zeros((2*nv, 2*nv))
B = np.zeros((2*nv, nu))
epsilon = 1e-6
flg_centered = True
mujoco.mjd_transitionFD(model, data, epsilon, flg_centered, A, B, None, None)

# Solve discrete Riccati equation.
P = scipy.linalg.solve_discrete_are(A, B, Q, R)

# Compute the feedback gain matrix K.
K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A


############################
## Perform the simulation ##
############################

# Reset data, set initial pose.
mujoco.mj_resetData(model, data)
data.qpos = qpos0

# Allocate position difference dq.
dq = np.zeros(model.nv)

with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
    # Set camera options
    viewer.cam.distance = 0.5
    viewer.cam.elevation = -15

    # Close the viewer automatically after a duration
    start = time.time()
    while viewer.is_running() and time.time() - start < DURATION/TIME_SCALE:
        step_start = time.time()

        # Get state difference dx.
        mujoco.mj_differentiatePos(model, dq, 1, qpos0, data.qpos)
        dx = np.hstack((dq, data.qvel)).T

        # LQR control law.
        data.ctrl = ctrl0 - K @ dx
        
        # Step the simulation
        mujoco.mj_step(model, data)

        # Example modification of a viewer option: toggle contact points every two seconds.
        with viewer.lock():
            viewer.cam.lookat = data.body('torso').xpos
        
        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = (1/TIME_SCALE)*model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)