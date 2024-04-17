# %%
import numpy as np
import pathlib

data_path = pathlib.Path("../data/2024-02-07_softball_0-051_5random/evaled_grasp_config_dicts/ddg-ycb_054_softball_0_0510.npy")
# data_path = pathlib.Path(f"../data/2024-02-07_50mugs_0-075_5random/evaled_grasp_config_dicts_train/core-mug-10f6e09036350e92b3f21f1137c3c347_0_0750.npy")
assert data_path.exists()

# %%
data_dict = np.load(data_path, allow_pickle=True).item()

# %%
print(f"data_dict.keys() = {data_dict.keys()}")

import matplotlib.pyplot as plt

# %%
trans = data_dict["trans"]
passed_sim = data_dict["passed_simulation"]
plt.scatter(trans[:, 0], trans[:, 1], s=1, c=passed_sim)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Passed simulation')
plt.colorbar()

# %%
plt.scatter(trans[:, 0], trans[:, 1], s=1, c=passed_sim)
plt.xlim(-0.12, 0.12)
plt.ylim(-0.05, 0.15)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Passed simulation')
plt.colorbar()


# %%
from scipy.spatial.transform import Rotation as R
rpy = R.from_matrix(data_dict["rot"]).as_euler('xyz', degrees=True)

# %%
# HACK: wrap to avoid discontinuity at 180 degrees.
# Make in [0, 360]
rpy = (rpy + 360) % 360
rpy[rpy < 180] += 360

# %%
# Color each point by whether it passed the simulation.
plt.scatter(rpy[:, 0], rpy[:, 1], s=1, c=passed_sim)
plt.xlabel('r')
plt.ylabel('p')
plt.title('Passed simulation')
# Add a colorbar.
plt.colorbar()

# %%
import plotly.graph_objects as go
fig = go.Figure()
# fig.add_trace(go.Scatter(x=data_dict["trans"][:, 0], y=data_dict["trans"][:, 1], mode="markers", marker=dict(size=1, color=passed_sim)))
fig.add_trace(go.Scatter3d(x=rpy[:, 0], y=rpy[:, 1], z=passed_sim, mode="markers", marker=dict(size=1, color=passed_sim)))

fig.show()


# %%
