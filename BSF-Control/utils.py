import numpy as np
import matplotlib.pyplot as plt

import mujoco
from mujoco._structs import MjModel, MjData

from xml.dom.minidom import parseString


def export_xml(xml: str, file_name: str = "mjmodel.xml", pretty: bool = False) -> None:
    # Pretty-print the XML string
    if pretty:
        pretty_xml: str = parseString(xml).toprettyxml(indent="  ")

    # Write the pretty-printed XML to a file
    with open(file_name, "w") as f:
        f.write(pretty_xml)

    return None


def calc_vertical_offset(
    model: MjModel, data: MjData, keyframe: int = 0, plot: bool = False
) -> float:
    """Calculate and fine-tune the vertical height offset where the zero vertical
    acceleration can be entirely explained by internal joint forces, without
    resorting to "magical" external forces. It is assumed that the required offset
    lies within the range of +- 0.001. If not, roughly adjust the vertical position
    of the model and use this function to fine-tune it further.

        Args:
            model (MjModel): MuJoCo model data structure
            data (MjData): MuJoCo simulation state data structure
            keyframe (int, optional): model keyframe. Defaults to 0.
            plot (bool, optional): plot vertical force - height offset relationship figure. Defaults to False.

        Returns:
            float: vertical offset with smallest vertical force
    """

    # Determine point where the force on the 3rd degree-of-freedom (DoF) of the root joint is zero.
    height_offsets = np.linspace(-0.01, 0.01, 20001)
    vertical_forces = []
    for offset in height_offsets:
        mujoco.mj_resetDataKeyframe(model, data, keyframe)
        mujoco.mj_forward(model, data)
        data.qacc = 0
        # Offset the height by `offset`.
        data.qpos[2] += offset
        mujoco.mj_inverse(model, data)
        vertical_forces.append(data.qfrc_inverse[2])

    # Find the height-offset at which the vertical force is smallest.
    idx = np.argmin(np.abs(vertical_forces))
    best_offset = height_offsets[idx]

    if plot:
        # Plot the relationship.
        plt.figure(figsize=(10, 6))
        plt.plot(height_offsets * 1000, vertical_forces, linewidth=3)
        # Red vertical line at offset corresponding to smallest vertical force.
        plt.axvline(x=best_offset * 1000, color="red", linestyle="--")
        # Green horizontal line at the humanoid's weight.
        weight = model.body_subtreemass[1] * np.linalg.norm(model.opt.gravity)
        plt.axhline(y=weight, color="green", linestyle="--")
        plt.xlabel("Height offset (mm)")
        plt.ylabel("Vertical force (N)")
        plt.grid(which="major", color="#DDDDDD", linewidth=0.8)
        plt.grid(which="minor", color="#EEEEEE", linestyle=":", linewidth=0.5)
        plt.minorticks_on()
        plt.title(
            f"Smallest vertical force " f"found at offset {best_offset*1000:.4f}mm."
        )
        plt.show()

    return best_offset


def arr2str(array) -> str:
    str_array = ' '.join(map(str, array))
    return str_array


def euler2quat(radians: float, axis: list[float] = [0, 1, 0]):
    qw = np.cos(radians/2)
    qx = axis[0]*np.sin(radians/2)
    qy = axis[1]*np.sin(radians/2)
    qz = axis[2]*np.sin(radians/2)
    return [qw, qx, qy, qz]


def set_state(state: list[float], mjdata: MjData) -> None:
    # qpos = [x, y, z, qw, qx, qy, qz, qrh, qlh, qrl, qll]
    qpos = [0, 0, 0]
    qpos += euler2quat(state[0])
    qpos += state[1:]
    mjdata.qpos = qpos
    return None


if __name__ == "__main__":
    quat = euler2quat(np.deg2rad(90))
    print(quat)



# # Get gradient
#     max_iterations = 1000
#     h = 1e-2
#     lr = 1e-25
#     vertical_position = 0.2
#     for i in range(max_iterations):
#         mujoco.mj_resetData(mjmodel, mjdata)
#         mujoco.mj_forward(mjmodel, mjdata)
#         mjdata.qacc = 0
       
#         mjdata.qpos[2] = vertical_position
#         mujoco.mj_inverse(mjmodel, mjdata)
#         f0 = mjdata.qfrc_inverse[2]

#         mjdata.qpos[2] = vertical_position + h
#         mujoco.mj_inverse(mjmodel, mjdata)
#         f1 = mjdata.qfrc_inverse[2]

#         grad = (f1**2 - f0**2)/h

#         vertical_position = vertical_position + lr*grad

#         print(f"{f0=}\t\t{grad*lr=}\t\t{vertical_position=}")