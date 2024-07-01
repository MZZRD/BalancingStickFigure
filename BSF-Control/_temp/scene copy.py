import xml.etree.ElementTree as ET
import mujoco
import mujoco.viewer
from mujoco._structs import MjModel, MjData


def main() -> str:
    # Create the mujoco element
    mjmodel = ET.Element("mujoco", model="scene")

    # Set compiler & simulation parameters
    ET.SubElement(mjmodel, "compiler", angle="radian")
    ET.SubElement(
        mjmodel, "option", timestep="0.003", gravity="0 0 -9.81", integrator="RK4"
    )

    # Add visual element
    visual = ET.SubElement(mjmodel, "visual")

    # Add asset element
    asset = ET.SubElement(mjmodel, "asset")
    ET.SubElement(
        asset,
        "texture",
        type="skybox",
        builtin="gradient",
        rgb1=".3 .5 .7",
        rgb2="0 0 0",
        width="32",
        height="512",
    )
    ET.SubElement(
        asset,
        "texture",
        name="body",
        type="cube",
        builtin="flat",
        mark="cross",
        width="128",
        height="128",
        rgb1="0.8 0.6 0.4",
        rgb2="0.8 0.6 0.4",
        markrgb="1 1 1",
        random="0.01",
    )
    ET.SubElement(
        asset,
        "material",
        name="body",
        texture="body",
        texuniform="true",
        rgba="0.8 0.6 .4 1",
    )
    ET.SubElement(
        asset,
        "texture",
        name="grid",
        type="2d",
        builtin="checker",
        width="512",
        height="512",
        rgb1=".1 .2 .3",
        rgb2=".2 .3 .4",
    )
    ET.SubElement(
        asset,
        "material",
        name="grid",
        texture="grid",
        texrepeat="1 1",
        texuniform="true",
        reflectance=".2",
    )

    # Add worldbody element
    worldbody = ET.SubElement(mjmodel, "worldbody")
    ET.SubElement(
        worldbody,
        "geom",
        name="floor",
        size="0 0 0.05",
        type="plane",
        material="grid",
        condim="3",
    )
    

    # Convert the model tree to a string
    root_xml = ET.tostring(mjmodel, encoding="utf-8", method="xml")

    return root_xml


if __name__ == "__main__":
    # Generate the model xml string
    model_xml: str = main()

    # Load the model in the viewer
    model: MjModel = mujoco.MjModel.from_xml_string(model_xml)
    data: MjData = mujoco.MjData(model)
    mujoco.viewer.launch(model, data)


# How to continue?
## What I like:
### I like that this solution is vanilla python, no opening or loading of external file structures is happening which keeps things clean
### I like that I can include variables and other python structures, this allows for automatic keydata generation and correct placement of the model on the ground.
###     Next to that it also allows to for example rotate the main body and determine on which limb it should balance, making that a keydata 
###     or creating a function that creates correct initial positions based on main body rotation, etc.
## What I don't like:
### an xml file can get pretty big and tags can get a lot of options, so when running a formatter it can look pretty unintuitive and messy. it would be better if the model is seperated into static and dynamics things.
###     e.g. the actually robot model is dynamic because there are parameters that depend on the scaling factor, the envirionment is not dynamic and never changes so it can be seperated.
###     This would also allow for easier changing of environments for the robot, eventhough I don't think this will happen. The main environment is a flat floor, but who knows.

# TODO: Create the environment first in fully vanilla python
# TODO: Create a simple model script with a cube and make lighting work
# TODO: Start creating the Torso, and then the rest.