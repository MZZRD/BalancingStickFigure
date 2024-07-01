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

    # Define xml model description
    model_xml: xml = rf"""
    <mujoco model="stick_figure">
        <option timestep="{timestep}" gravity="0 0 {gravity}" integrator="{integrator}"/>
        <compiler angle="radian" meshdir="meshes"/>
        
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
        </asset>

        <worldbody>
            <!-- <light name="spotlight" mode="targetbodycom" target="torso" diffuse=".8 .8 .8" specular="0.3 0.3 0.3" pos="0 -6 4" cutoff="30"/> -->
            <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" contype="1" conaffinity="1"/>
            <body name=
        </worldbody>

        <actuator>

        </actuator>
        
        <keyframe>
            
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
    model_xml: str = get_model_xml()

    # Load the model in the viewer
    model: MjModel = mujoco.MjModel.from_xml_string(model_xml)
    data: MjData = mujoco.MjData(model)
    mujoco.viewer.launch(model, data)
