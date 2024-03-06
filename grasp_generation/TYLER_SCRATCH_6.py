# %%
import numpy as np
import pathlib

# data_path = pathlib.Path("../data/2024-03-05_softballs_idx11_augmented_pose/evaled_grasp_config_dicts/ddg-ycb_054_softball_0_0350.npy")
data_path = pathlib.Path("../data/2024-03-05_softballs_idx11_augmented_pose_HALTON/evaled_grasp_config_dicts/ddg-ycb_054_softball_0_0350.npy")
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
# Color each point by whether it passed the simulation.
plt.scatter(data_dict["trans"][:, 0], data_dict["trans"][:, 1], s=1, c=label_to_use)
plt.xlabel('x')
plt.ylabel('y')
plt.title(label_name)
# Add a colorbar.
plt.colorbar()

# %%
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter3d(x=data_dict["trans"][:, 0], y=data_dict["trans"][:, 1], z=label_to_use, mode="markers", marker=dict(size=1, color=label_to_use)))

fig.show()

# %%
fig = go.Figure()
thresholds = np.linspace(0, 1, 10)
colors = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "gray"]
for low_thresh, high_thresh, color in zip(thresholds[:-1], thresholds[1:], colors):
    print(f"low_thresh = {low_thresh}, high_thresh = {high_thresh}")
    in_range = np.logical_and(label_to_use > low_thresh, label_to_use <= high_thresh)
    fig.add_trace(go.Scatter3d(x=data_dict["trans"][in_range, 0],
                               y=data_dict["trans"][in_range, 1],
                               z=data_dict["trans"][in_range, 2],
                               mode="markers",
                               # marker=dict(size=1, color=label_to_use[in_range]),
                               marker=dict(size=1, color=color),
                               name=f"{low_thresh:.2f} to {high_thresh:.2f}"))
fig.show()
# %%
fig.add_trace(go.Scatter3d(x=data_dict["trans"][:, 0], y=data_dict["trans"][:, 1], z=data_dict["trans"][:, 2], mode="markers", marker=dict(size=1, color=label_to_use)))

fig.show()


# %%
from scipy.spatial.transform import Rotation as R
dist_from_base = np.linalg.norm(data_dict["trans"], axis=1)
rpy = R.from_matrix(data_dict["rot"]).as_euler('xyz', degrees=True)
rot_deg_from_base = np.linalg.norm(rpy, axis=1)
plt.scatter(dist_from_base, rot_deg_from_base, s=1, c=label_to_use)


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
N = XYZ.shape[0]
my_trans_diff = trans_diff(XYZ[0:1].repeat(N, axis=0), XYZ)
my_rot_diff = rot_diff(RPY[0:1].repeat(N, axis=0), RPY)

# %%
import matplotlib.pyplot as plt
plt.hist(my_trans_diff)

# %%
plt.hist(my_rot_diff)
# %%
plt.scatter(my_trans_diff, my_rot_diff, s=1, c=label_to_use)

# %%
fig = go.Figure()
fig.add_trace(
    go.Scatter3d(
        x=my_trans_diff,
        y=my_rot_diff,
        z=label_to_use,
        mode="markers",
        marker=dict(size=1, color=label_to_use)
    )
)
fig.show()

# %%
