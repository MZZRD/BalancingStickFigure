import mujoco

model = mujoco.MjModel.from_xml_path('model.xml')
data = mujoco.MjData(model)

while data.time < 1:
  mujoco.mj_step(model, data)
  print(data.geom_xpos)