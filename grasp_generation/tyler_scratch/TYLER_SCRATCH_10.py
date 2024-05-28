# %%
import numpy as np
import pathlib

data_path = pathlib.Path(
    "../data/2024-03-05_softballs_idx11_augmented_pose_HALTON_no-rot/evaled_grasp_config_dicts/ddg-ycb_054_softball_0_0350.npy"
)
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
z_min, z_max = np.min(data_dict["trans"][:, 2]), np.max(data_dict["trans"][:, 2])
n_plots = 10
z_list = np.linspace(z_min, z_max, n_plots + 1)
nrows = int(np.ceil(np.sqrt(n_plots)))
ncols = int(np.ceil(n_plots / nrows))
fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
axes = axes.flatten()
for i, (z_low, z_high) in enumerate(zip(z_list[:-1], z_list[1:])):
    points_to_plot = np.logical_and(data_dict["trans"][:, 2] > z_low, data_dict["trans"][:, 2] < z_high)
    axes[i].scatter(data_dict["trans"][points_to_plot, 0], data_dict["trans"][points_to_plot, 1], s=1, c=label_to_use[points_to_plot])
    axes[i].set_title(f"z in [{z_low:.2f}, {z_high:.2f}]")
fig.tight_layout()

# %%
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(
    go.Scatter3d(
        x=data_dict["trans"][:, 0],
        y=data_dict["trans"][:, 1],
        z=data_dict["trans"][:, 2],
        mode="markers",
        marker=dict(size=1, color=label_to_use),
    )
)

fig.show()


# %%
pass_points = label_to_use > 0.5

fig = go.Figure()
fig.add_trace(
    go.Scatter3d(
        x=data_dict["trans"][pass_points, 0],
        y=data_dict["trans"][pass_points, 1],
        z=data_dict["trans"][pass_points, 2],
        mode="markers",
        marker=dict(size=1, color=label_to_use),
    )
)

fig.show()


# %%

fig = go.Figure()
thresholds = np.linspace(0, 1, 10)
colors = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "gray"]
for low_thresh, high_thresh, color in zip(thresholds[:-1], thresholds[1:], colors):
    print(f"low_thresh = {low_thresh}, high_thresh = {high_thresh}")
    in_range = np.logical_and(label_to_use >= low_thresh, label_to_use < high_thresh)
    fig.add_trace(go.Scatter3d(x=data_dict["trans"][in_range, 0],
                               y=data_dict["trans"][in_range, 1],
                               z=data_dict["trans"][in_range, 2],
                               mode="markers",
                               # marker=dict(size=1, color=label_to_use[in_range]),
                               marker=dict(size=1, color=color),
                               name=f"{low_thresh:.2f} to {high_thresh:.2f}"))
fig.show()
# %%
