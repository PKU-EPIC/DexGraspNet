# %%
import numpy as np
import pathlib

# data_path = pathlib.Path("../data/ROT_PROBE_1_2024-02-07_softball_0-051_5random/evaled_grasp_config_dicts/ddg-ycb_054_softball_0_0510.npy")
data_path = pathlib.Path(f"../data/ROT_PROBE_7_2024-02-07_50mugs_0-075_5random/evaled_grasp_config_dicts_train/core-mug-10f6e09036350e92b3f21f1137c3c347_0_0750.npy")
assert data_path.exists()

# %%
data_dict = np.load(data_path, allow_pickle=True).item()

# %%
print(f"data_dict.keys() = {data_dict.keys()}")

import matplotlib.pyplot as plt

# %%
passed_sim = data_dict["passed_simulation"]
passed_penetration = data_dict["passed_new_penetration_test"]
passed_eval = data_dict["passed_eval"]

label_to_use = passed_eval
label_name = "passed_eval"

# label_to_use = passed_sim
# label_name = "passed_sim"

# %%
plt.hist(label_to_use)

# %%
from scipy.spatial.transform import Rotation as R
rpy = R.from_matrix(data_dict["rot"]).as_euler('xyz', degrees=True)

# %%
# HACK: wrap to avoid discontinuity at 180 degrees.
# Make in [0, 360]
rpy = (rpy + 360) % 360
rpy[rpy < 50] += 360

# %%
# Color each point by whether it passed the simulation.
plt.scatter(rpy[:, 0], rpy[:, 1], s=1, c=label_to_use)
plt.xlabel('r')
plt.ylabel('p')
plt.title(label_name)
# Add a colorbar.
plt.colorbar()

# %%
import plotly.graph_objects as go
fig = go.Figure()
# fig.add_trace(go.Scatter(x=data_dict["trans"][:, 0], y=data_dict["trans"][:, 1], mode="markers", marker=dict(size=1, color=label_to_use)))
fig.add_trace(go.Scatter3d(x=rpy[:, 0], y=rpy[:, 1], z=label_to_use, mode="markers", marker=dict(size=1, color=label_to_use)))

fig.show()


# %%
