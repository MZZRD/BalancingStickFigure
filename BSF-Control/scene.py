import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import mujoco.viewer


def arr2str(array) -> str:
    str_array = ' '.join(map(str, array))
    return str_array


def main() -> str:
    # Base dimensions
    h = 12e-3 # height limb (z-axis) 
    w = 4e-3 # width limb (x-axis)
    d = 4e-3 # depth limb (y-axis)
    tol = .250e-3 # tolerance
    a = 10 # scaling factor

    # Toros parameters
    head_size = a * np.array([w, d/2])
    head_pos = a * np.array([0, 0, w/2 + w])
    chest_size = a * np.array([(3*w+3*tol)/2, d/2, w/2])
    stomach_size = a * np.array([(2*w+tol)/2, d/2, (h-w/2)/2])
    stomach_pos = a * np.array([0, 0, -(w + 2*h)/4])

    # Joint parameters
    joint_size = a * np.array([w/2, d/2])
    sholder_right_pos = a * np.array([(3*w+3*tol)/2, 0, 0])
    sholder_left_pos = -sholder_right_pos
    hip_right_pos = a * np.array([(w+tol)/2, 0, -h])
    hip_left_pos = a * np.array([-(w+tol)/2, 0, -h])

    # Limb parameters (relative to relevant joint position)
    limb_size = a * np.array([w/2, d/2, (h-w)/2])
    limb_pos = a * np.array([0, 0, -w])
    limb_end_pos = a * np.array([0, 0, -2*w])

    # Create the root mujoco element
    mjmodel = ET.Element("mujoco", model="scene")

    # Set compiler & simulation parameters
    ET.SubElement(mjmodel, "compiler", angle="degree")
    ET.SubElement(mjmodel, "option", timestep="0.003", gravity="0 0 -9.81", integrator="RK4")

    # Add visual elements
    visual = ET.SubElement(mjmodel, "visual")
    ET.SubElement(visual, "headlight", diffuse="0.6 0.6 0.6", ambient="0.3 0.3 0.3", specular="0 0 0")
    ET.SubElement(visual, "rgba", haze="0.15 0.25 0.35 1")
    ET.SubElement(visual, "global", azimuth="150", elevation="-20")
    
    # Add asset elements
    asset = ET.SubElement(mjmodel, "asset")
    ET.SubElement(asset, "texture", type="skybox", builtin="gradient", rgb1="0.3 0.5 0.7", rgb2="0 0 0", width="512", height="3072")
    ET.SubElement(asset, "texture", type="2d", name="groundplane", builtin="checker", mark="edge", rgb1="0.2 0.3 0.4", rgb2="0.1 0.2 0.3", markrgb="0.8 0.8 0.8", width="300", height="300")
    ET.SubElement(asset, "material", name="groundplane", texture="groundplane", texuniform="true", texrepeat="5 5", reflectance="0.2")
    ET.SubElement(asset, "material", name="geom", rgba="1.0 0.8 0.0 1.0")

    default = ET.SubElement(mjmodel, "default")
    ET.SubElement(default, "geom", material="geom")

    # Add worldbody elements
    worldbody = ET.SubElement(mjmodel, "worldbody")
    ET.SubElement(worldbody, "light", pos="0 0 3", dir="0 0 -1", directional="false")
    ET.SubElement(worldbody, "geom", name="floor", size="0 0 .125", type="plane", material="groundplane")
    
    # The actual model (to seperate)
    torso = ET.SubElement(worldbody, "body", name="torso", pos="0 0 .1")
    ET.SubElement(torso, "joint", type="free")
    ET.SubElement(torso, "geom", name="chest", type="box", size=arr2str(chest_size))
    ET.SubElement(torso, "geom", name="right_sholder", type="cylinder", size=arr2str(joint_size), pos=arr2str(sholder_right_pos), euler="90 0 0")
    ET.SubElement(torso, "geom", name="left_sholder", type="cylinder", size=arr2str(joint_size), pos=arr2str(sholder_left_pos), euler="90 0 0")
    ET.SubElement(torso, "geom", name="stomach", type="box", size=arr2str(stomach_size), pos=arr2str(stomach_pos))
    ET.SubElement(torso, "geom", name="right_hip", type="cylinder", size=arr2str(joint_size), pos=arr2str(hip_right_pos), euler="90 0 0")
    ET.SubElement(torso, "geom", name="left_hip", type="cylinder", size=arr2str(joint_size), pos=arr2str(hip_left_pos), euler="90 0 0")

    head = ET.SubElement(torso, "body", name="head", pos=arr2str(head_pos))
    ET.SubElement(head, "geom", name="head", type="cylinder", size=arr2str(head_size), euler="90 0 0")

    right_arm = ET.SubElement(torso, "body", name="right_arm", pos=arr2str(sholder_right_pos), euler="0 -90 0")
    ET.SubElement(right_arm, "joint", name="right_arm", type="hinge", axis="0 -1 0", limited="true", range="-90 90")
    ET.SubElement(right_arm, "geom", name="right_arm", type="box", size=arr2str(limb_size), pos=arr2str(limb_pos))
    ET.SubElement(right_arm, "geom", name="right_hand", type="cylinder", size=arr2str(joint_size), pos=arr2str(limb_end_pos), euler="90 0 0")

    left_arm = ET.SubElement(torso, "body", name="left_arm", pos=arr2str(sholder_left_pos), euler="0 90 0")
    ET.SubElement(left_arm, "joint", name="left_arm", type="hinge", axis="0 1 0", limited="true", range="-90 90")
    ET.SubElement(left_arm, "geom", name="left_arm", type="box", size=arr2str(limb_size), pos=arr2str(limb_pos))
    ET.SubElement(left_arm, "geom", name="left_hand", type="cylinder", size=arr2str(joint_size), pos=arr2str(limb_end_pos), euler="90 0 0")

    right_leg = ET.SubElement(torso, "body", name="right_leg", pos=arr2str(hip_right_pos), euler="0 0 0")
    ET.SubElement(right_leg, "joint", name="right_leg", type="hinge", axis="0 -1 0", limited="true", range="-15 90")
    ET.SubElement(right_leg, "geom", name="right_leg", type="box", size=arr2str(limb_size), pos=arr2str(limb_pos))
    ET.SubElement(right_leg, "geom", name="right_foot", type="cylinder", size=arr2str(joint_size), pos=arr2str(limb_end_pos), euler="90 0 0")

    left_leg = ET.SubElement(torso, "body", name="left_leg", pos=arr2str(hip_left_pos), euler="0 0 0")
    ET.SubElement(left_leg, "joint", name="left_leg", type="hinge", axis="0 1 0", limited="true", range="-15 90")
    ET.SubElement(left_leg, "geom", name="left_leg", type="box", size=arr2str(limb_size), pos=arr2str(limb_pos))
    ET.SubElement(left_leg, "geom", name="left_foot", type="cylinder", size=arr2str(joint_size), pos=arr2str(limb_end_pos), euler="90 0 0")

    actuator = ET.SubElement(mjmodel, "actuator")
    ET.SubElement(actuator, "motor", name="left_arm", joint="left_arm")
    ET.SubElement(actuator, "motor", name="right_arm", joint="right_arm")
    ET.SubElement(actuator, "motor", name="left_leg", joint="left_leg")
    ET.SubElement(actuator, "motor", name="right_leg", joint="right_leg")

    # Convert the tree to a xml string
    mjmodel_xml = ET.tostring(mjmodel, encoding="utf-8", method="xml")

    return mjmodel_xml


if __name__ == "__main__":
    # Load the model in the viewer
    mjmodel_xml = main()
    mjmodel = mujoco.MjModel.from_xml_string(mjmodel_xml)
    mjdata = mujoco.MjData(mjmodel)
    mujoco.viewer.launch(mjmodel, mjdata)

# TODO: Create a custom generalized state for the model
# TODO: Given the generalized state, place the model neatly on the ground
# TODO: Greate a function that given a certain rotation of the torso decides on which limb (or even head) to balance and create the generalized state for that. that makes sure to have the CoM above the contact point


# Maybe create a class of this robot with the functions to change state or randomly reinitialize etc.
# state = [0, 0, 0, 0, 0] # angle of [torso, right_arm, left_arm, right_leg, left_leg]