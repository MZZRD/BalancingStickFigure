from sourcetypes import xml
import mujoco
import mujoco.viewer

def main():
    # Sim/scene variables
    timestep = 0.01  # [s] simulation timestep
    gravity = -9.81  # [m/s^2] gravity in z-axis direction
    integrator = "RK4"  # simulation integrator
    
    # Body positions
    head_body_pos = "0 0 0.05752"
    torso_body_pos = "0 0 0.08004"
    left_arm_body_pos = "-0.02893 0 0.03009"
    right_arm_body_pos = "0.02893 0 0.03009"
    left_leg_body_pos = "-0.00964 0 -0.03009"
    right_leg_body_pos = "0.00964 0 -0.03009"
    
    # Body center of masses
    head_com = "0 0 0"
    torso_com = "0 0 0.00692"
    arm_com = "0 0 -0.02580032"
    leg_com = "0 0 -0.02580032"
    
    # Body masses
    head_mass = 0.02401526
    torso_mass = 0.07159599
    arm_mass = 0.07159599
    leg_mass = 0.07159599
    
    # Body diagonal inertias
    head_inertia = "0.00000268 0.00000402 0.00000268"
    torso_inertia = "0.00003631 0.00004998 0.00001708"
    arm_inertia = "0.000004 0.00000391 0.00000105"
    leg_inertia = "0.000004 0.00000391 0.00000105"
    
    # Define xml model description
    model: xml = rf"""
    <mujoco model="stick_figure">
        <option timestep="{timestep}" gravity="0 0 {gravity}" integrator="{integrator}"/>
        <compiler angle="degree" meshdir="meshes" texturedir="textures" />
        
        <visual>
            <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
            <rgba haze="0.15 0.25 0.35 1"/>
            <global azimuth="120" elevation="-20"/>
        </visual>

        <asset>
            <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
            <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
            <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
            <material name="body" rgba="1.0 0.8 0.0 1"/>
            <mesh name="head_mesh" file="head.stl"/>
            <mesh name="torso_mesh" file="torso.stl"/>
            <mesh name="arm_mesh" file="arm.stl"/>
            <mesh name="leg_mesh" file="leg.stl"/>
        </asset>

        <worldbody>
            <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
            <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
            <body name="torso" pos="{torso_body_pos}">
                <camera name="track_torso_com" mode="targetbodycom" target="torso" pos="0 -0.5 0"/>
                <joint type="free"/>
                <geom type="mesh" mesh="torso_mesh" material="body"/>
                <inertial pos="{torso_com}" mass="{torso_mass}" diaginertia="{torso_inertia}"/>
                <body name="head" pos="{head_body_pos}">
                    <geom type="mesh" mesh="head_mesh" material="body"/>
                    <inertial pos="{head_com}" mass="{head_mass}" diaginertia="{head_inertia}"/>
                </body>
                <body name="left_arm" pos="{left_arm_body_pos}">
                    <joint name="left_arm" type="hinge" pos="0 0 0" axis="0 1 0" limited="true" range="0 180"/>
                    <geom type="mesh" mesh="arm_mesh" material="body"/>
                    <inertial pos="{arm_com}" mass="{arm_mass}" diaginertia="{arm_inertia}"/>
                </body>
                <body name="right_arm" pos="{right_arm_body_pos}">
                    <joint name="right_arm" type="hinge" pos="0 0 0" axis="0 -1 0" limited="true" range="0 180"/>
                    <geom type="mesh" mesh="arm_mesh" material="body"/>
                    <inertial pos="{arm_com}" mass="{arm_mass}" diaginertia="{arm_inertia}"/>
                </body>
                <body name="left_leg" pos="{left_leg_body_pos}">
                    <joint name="left_leg" type="hinge" pos="0 0 0" axis="0 1 0" limited="true" range="0 90"/>
                    <geom type="mesh" mesh="leg_mesh" material="body"/>
                    <inertial pos="{leg_com}" mass="{leg_mass}" diaginertia="{leg_inertia}"/>
                </body>
                <body name="right_leg" pos="{right_leg_body_pos}">
                    <joint name="right_leg" type="hinge" pos="0 0 0" axis="0 -1 0" limited="true" range="0 90"/>
                    <geom type="mesh" mesh="leg_mesh" material="body"/>
                    <inertial pos="{leg_com}" mass="{leg_mass}" diaginertia="{leg_inertia}"/>
                </body>
            </body>
        </worldbody>

        <actuator>
            <position name="left_arm" joint="left_arm" inheritrange="1" kp="1"/>
            <position name="right_arm" joint="right_arm" inheritrange="1" kp="1"/>
            <position name="left_leg" joint="left_leg" inheritrange="1" kp="1"/>
            <position name="right_leg" joint="right_leg" inheritrange="1" kp="1"/>
        </actuator>
    </mujoco>
    """

    # Export the model xml
    with open("model.xml", "w") as file:
        file.write(model)

    return 0


if __name__ == "__main__":
    main()
    
    # Load the model in the viewer
    model = mujoco.MjModel.from_xml_path('model.xml')
    data = mujoco.MjData(model)
    mujoco.viewer.launch(model, data)