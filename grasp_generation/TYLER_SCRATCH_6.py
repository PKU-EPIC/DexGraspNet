# %%
import numpy as np
import pathlib

# data_path = pathlib.Path("../data/PROBE_2024-02-07_softball_0-051_5random/evaled_grasp_config_dicts/ddg-ycb_054_softball_0_0510.npy")
data_path = pathlib.Path("../data/2024-03-05_softballs_idx11_augmented_pose/evaled_grasp_config_dicts/ddg-ycb_054_softball_0_0350.npy")
# data_path = pathlib.Path(f"../data/PROBE_7_2024-02-07_50mugs_0-075_5random/evaled_grasp_config_dicts_train/core-mug-10f6e09036350e92b3f21f1137c3c347_0_0750.npy")
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

# %%
plt.hist(passed_sim)

# %%
# Color each point by whether it passed the simulation.
plt.scatter(data_dict["trans"][:, 0], data_dict["trans"][:, 1], s=1, c=passed_sim)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Passed simulation')
# Add a colorbar.
plt.colorbar()

# %%
import plotly.graph_objects as go
fig = go.Figure()
# fig.add_trace(go.Scatter(x=data_dict["trans"][:, 0], y=data_dict["trans"][:, 1], mode="markers", marker=dict(size=1, color=passed_sim)))
fig.add_trace(go.Scatter3d(x=data_dict["trans"][:, 0], y=data_dict["trans"][:, 1], z=passed_sim, mode="markers", marker=dict(size=1, color=passed_sim)))

fig.show()


# %%
from scipy.spatial.transform import Rotation as R
dist_from_base = np.linalg.norm(data_dict["trans"], axis=1)
rpy = R.from_matrix(data_dict["rot"]).as_euler('xyz', degrees=True)
rot_deg_from_base = np.linalg.norm(rpy, axis=1)
plt.scatter(dist_from_base, rot_deg_from_base, s=1, c=passed_sim)


# %%
XYZ = data_dict["trans"]
RPY = R.from_matrix(data_dict["rot"]).as_euler('xyz', degrees=True)

# %%
def trans_diff(xyz1: np.ndarray, xyz2: np.ndarray) -> np.ndarray:
    return np.linalg.norm(xyz1 - xyz2, axis=1)

def rot_diff(rpy1: np.ndarray, rpy2: np.ndarray) -> np.ndarray:
    q1 = R.from_euler('xyz', rpy1, degrees=True).as_quat()
    q2 = R.from_euler('xyz', rpy2, degrees=True).as_quat()
    dot_product = np.sum(q1 * q2, axis=1)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    theta = 2 * np.arccos(np.abs(dot_product))
    return np.degrees(theta)

# %%
my_trans_diff = trans_diff(XYZ[0:1].repeat(N, axis=0), XYZ)
my_rot_diff = rot_diff(RPY[0:1].repeat(N, axis=0), RPY)

# %%
import matplotlib.pyplot as plt
plt.hist(my_trans_diff)

# %%
plt.hist(my_rot_diff)