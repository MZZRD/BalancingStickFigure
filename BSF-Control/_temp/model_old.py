from sourcetypes import xml
import mujoco
import mujoco.viewer
from mujoco._structs import MjModel, MjData
import numpy as np


def get_model_xml(export: bool = False) -> str:
    # Sim/scene variables
    timestep = 0.003  # [s] simulation timestep
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
    # print(f"Total mass: {head_mass+torso_mass+2*arm_mass+2*leg_mass}")

    # Body diagonal inertias
    head_inertia = "0.00000268 0.00000402 0.00000268"
    torso_inertia = "0.00003631 0.00004998 0.00001708"
    arm_inertia = "0.000004 0.00000391 0.00000105"
    leg_inertia = "0.000004 0.00000391 0.00000105"

    # Actuator settings
    force_limited = "false"
    force_range = "-0.01 0.01"

    # Define xml model description
    model_xml: xml = rf"""
    <mujoco model="stick_figure">
        <option timestep="{timestep}" gravity="0 0 {gravity}" integrator="{integrator}"/>
        <compiler angle="degree" meshdir="meshes" texturedir="textures" />
        
        <visual>
            <map force="0.1" zfar="30"/>
            <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
            <rgba haze="0.15 0.25 0.35 1"/>
            <global offwidth="2560" offheight="1440" elevation="-20" azimuth="120"/>
        </visual>

        <asset>
            <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="32" height="512"/>
            <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
            <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
            <material name="body" rgba="1.0 0.8 0.0 1"/>
            <mesh name="head_mesh" file="head.stl"/>
            <mesh name="torso_mesh" file="torso.stl"/>
            <mesh name="arm_mesh" file="arm.stl"/>
            <mesh name="leg_mesh" file="leg.stl"/>
        </asset>

        <worldbody>
            <!-- <light pos="0 0 1.5" dir="0 0 -1" directional="true"/> -->
            <light name="spotlight" mode="targetbodycom" target="torso" diffuse=".8 .8 .8" specular="0.3 0.3 0.3" pos="0 -6 4" cutoff="30"/>
            <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" contype="1" conaffinity="1"/>
            <body name="torso" pos="{torso_body_pos}">
                <light name="top" pos="0 0 2" mode="trackcom"/>
                <joint name="torso" type="free"/>
                <geom type="mesh" mesh="torso_mesh" material="body" contype="2" conaffinity="1"/>
                <inertial pos="{torso_com}" mass="{torso_mass}" diaginertia="{torso_inertia}"/>
                <body name="head" pos="{head_body_pos}">"-0.01 0.01"
                    <geom type="mesh" mesh="head_mesh" material="body" contype="2" conaffinity="1"/>
                    <inertial pos="{head_com}" mass="{head_mass}" diaginertia="{head_inertia}"/>
                </body>
                <body name="left_arm" pos="{left_arm_body_pos}">
                    <joint name="left_arm" type="hinge" pos="0 0 0" axis="0 1 0" limited="true" range="0 180"/>
                    <geom type="mesh" mesh="arm_mesh" material="body" contype="2" conaffinity="1"/>
                    <inertial pos="{arm_com}" mass="{arm_mass}" diaginertia="{arm_inertia}"/>
                </body>
                <body name="right_arm" pos="{right_arm_body_pos}">
                    <joint name="right_arm" type="hinge" pos="0 0 0" axis="0 -1 0" limited="true" range="0 180"/>
                    <geom type="mesh" mesh="arm_mesh" material="body" contype="2" conaffinity="1"/>
                    <inertial pos="{arm_com}" mass="{arm_mass}" diaginertia="{arm_inertia}"/>
                </body>
                <body name="left_leg" pos="{left_leg_body_pos}">
                    <joint name="left_leg" type="hinge" pos="0 0 0" axis="0 1 0" limited="true" range="0 90"/>
                    <geom type="mesh" mesh="leg_mesh" material="body" contype="2" conaffinity="1"/>
                    <inertial pos="{leg_com}" mass="{leg_mass}" diaginertia="{leg_inertia}"/>
                </body>
                <body name="right_leg" pos="{right_leg_body_pos}">
                    <joint name="right_leg" type="hinge" pos="0 0 0" axis="0 -1 0" limited="true" range="0 90"/>
                    <geom type="mesh" mesh="leg_mesh" material="body" contype="2" conaffinity="1"/>
                    <inertial pos="{leg_com}" mass="{leg_mass}" diaginertia="{leg_inertia}"/>
                </body>
            </body>
        </worldbody>

        <actuator>
            <motor name="left_arm" joint="left_arm" forcelimited="{force_limited}" forcerange="{force_range}"/>
            <motor name="right_arm" joint="right_arm" forcelimited="{force_limited}" forcerange="{force_range}"/>
            <motor name="left_leg" joint="left_leg" forcelimited="{force_limited}" forcerange="{force_range}"/>
            <motor name="right_leg" joint="right_leg" forcelimited="{force_limited}" forcerange="{force_range}"/>
            <!-- <position name="left_arm" joint="left_arm" inheritrange="1" kp="0.1" kv="0.01"/>
            <position name="right_arm" joint="right_arm" inheritrange="1" kp="0.2" kv="0.01"/>
            <position name="left_leg" joint="left_leg" inheritrange="1" kp="0.2" kv="0.01"/>
            <position name="right_leg" joint="right_leg" inheritrange="1" kp="0.2" kv="0.01"/> -->
        </actuator>
        
        <keyframe>
            <key name="stand"
                 qpos="
                 0.0 0.0 0.08
                 0.0 0.0 0.0 0.0 
                 0.0 0.0 0.0 0.0"/>
            <key name="balance_on_left_leg_0"
                 qpos="
                 0.0 0.0 0.0805534
                 0.9659 0.0 -0.2588 0.0 
                 {np.deg2rad(120)} {np.deg2rad(60)} {np.deg2rad(30)} {np.deg2rad(60)}"/>
            <key name="balance_on_left_leg_1"
                 qpos="
                 0.0 0.0 0.0805534 
                 0.9659 0.0 -0.2588 0.0 
                 {np.deg2rad(90)} {np.deg2rad(90)} {np.deg2rad(30)} {np.deg2rad(0)}"/>
        </keyframe>
    </mujoco>
    """

    # Export the model xml
    if export:
        with open("model.xml", "w") as file:
            file.write(model_xml)

    return model_xml


if __name__ == "__main__":
    # Generate the model xml string
    model_xml: str = get_model_xml(export=True)

    # Load the model in the viewer
    model: MjModel = mujoco.MjModel.from_xml_string(model_xml)
    data: MjData = mujoco.MjData(model)
    mujoco.viewer.launch(model, data)
