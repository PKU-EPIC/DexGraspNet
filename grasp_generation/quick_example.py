# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
import random
import pathlib
from utils.hand_model import HandModel
from utils.object_model import ObjectModel
import numpy as np
import torch
from utils.hand_model_type import handmodeltype_to_joint_names, HandModelType
from utils.qpos_pose_conversion import qpos_to_pose
from utils.seed import set_seed
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
from utils.joint_angle_targets import (
    OptimizationMethod,
    compute_optimized_joint_angle_targets_given_grasp_orientations,
)
from utils.parse_object_code_and_scale import parse_object_code_and_scale


# %% [markdown]
# ## PARAMS

# %%
meshdata_root_path = "../data/meshdata"
grasp_config_dicts_path = pathlib.Path("../data/2023-08-23_grasp_config_dicts")
hand_model_type = HandModelType.ALLEGRO_HAND
seed = 102
joint_angle_targets_optimization_method = (
    OptimizationMethod.DESIRED_DIST_TOWARDS_OBJECT_SURFACE_MULTIPLE_STEPS
)
should_canonicalize_hand_pose = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# ## Set Seed

# %%
set_seed(seed)

# %% [markdown]
# ## Grasp codes|

# %%
object_code_list, object_scale_list = [], []
for path in grasp_config_dicts_path.iterdir():
    object_code_and_scale_str = path.stem

    object_code, object_scale = parse_object_code_and_scale(object_code_and_scale_str)
    object_code_list.append(object_code)
    object_scale_list.append(object_scale)

# %% [markdown]
# ## Sample and read in data

# %%
random_file_idx = random.randint(0, len(object_code_list) - 1)
object_code = object_code_list[random_file_idx]
object_scale = object_scale_list[random_file_idx]
grasp_data_list = np.load(
    grasp_config_dicts_path
    / (f"{object_code}_{object_scale:.2f}".replace(".", "_") + ".npy"),
    allow_pickle=True,
)
print(f"Randomly sampled object_code = {object_code}")

random_grasp_index = random.randint(0, len(grasp_data_list) - 1)
qpos = grasp_data_list[random_grasp_index]["qpos"]
grasp_orientations = torch.tensor(grasp_data_list[random_grasp_index]["grasp_orientations"], device=device, dtype=torch.float)
print(f"Randomly sampled random_grasp_index = {random_grasp_index}")
print(f"object_scale = {object_scale}")

# %% [markdown]
# ## Object model

# %%
object_model = ObjectModel(
    meshdata_root_path=meshdata_root_path,
    batch_size_each=1,
    scale=object_scale,
    device=device,
)
object_model.initialize([object_code])

# %% [markdown]
# ## Hand model

# %%
joint_names = handmodeltype_to_joint_names[hand_model_type]
hand_model = HandModel(hand_model_type, device=device)

hand_pose = qpos_to_pose(
    qpos=qpos, joint_names=joint_names, unsqueeze_batch_dim=True
).to(device)
hand_model.set_parameters(hand_pose)

batch_idx = 0
hand_mesh = hand_model.get_trimesh_data(batch_idx)
object_mesh = object_model.object_mesh_list[batch_idx].copy().apply_scale(object_scale)

# %% [markdown]
# ## Visualize hand and object

# %%
(hand_mesh + object_mesh).show()

# %% [markdown]
# ## Visualize hand and object plotly

# %%
fig_title = f"Grasp Code: {object_code}, Index: {random_grasp_index}"
idx_to_visualize = batch_idx

fig = go.Figure(
    layout=go.Layout(
        scene=dict(
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z"),
            aspectmode="data",
        ),
        showlegend=True,
        title=fig_title,
        autosize=False,
        width=800,
        height=800,
    )
)
plots = [
    *hand_model.get_plotly_data(
        i=idx_to_visualize, opacity=1.0, with_contact_candidates=True
    ),
    *object_model.get_plotly_data(i=idx_to_visualize, opacity=0.5),
]
for plot in plots:
    fig.add_trace(plot)
fig.show()

# %% [markdown]
# ## Compute optimized joint angle targets

# %%
original_hand_pose = hand_model.hand_pose.detach().clone()
print(f"original_hand_pose[:, 9:] = {original_hand_pose[:, 9:]}")

# %%
assert grasp_orientations.shape == (hand_model.num_fingers, 3, 3)
(
    joint_angle_targets_to_optimize,
    debug_info,
) = compute_optimized_joint_angle_targets_given_grasp_orientations(
    joint_angles_start=original_hand_pose[:, 9:],
    hand_model=hand_model,
    grasp_orientations=grasp_orientations.unsqueeze(dim=0),
)
losses = debug_info["loss"]

# %%
fig = px.line(y=losses)
fig.update_layout(
    title=f"{joint_angle_targets_optimization_method} Loss vs. Iterations",
    xaxis_title="Iterations",
    yaxis_title="Loss",
)
fig.show()

# %%
print(f"joint_angle_targets_to_optimize = {joint_angle_targets_to_optimize}")


# %% [markdown]
# ## Visualize hand pose before and after optimization

# %%
# Plotly fig
hand_model.set_parameters(original_hand_pose)
old_hand_model_plotly = hand_model.get_plotly_data(
    i=idx_to_visualize, opacity=1.0, with_contact_candidates=True
)

fig = make_subplots(
    rows=1,
    cols=2,
    specs=[[{"type": "scene"}, {"type": "scene"}]],
    subplot_titles=("Original", "Optimized"),
)

plots = [
    *old_hand_model_plotly,
    *object_model.get_plotly_data(i=idx_to_visualize, opacity=0.5),
]

for plot in plots:
    fig.append_trace(plot, row=1, col=1)

# %%

new_hand_pose = original_hand_pose.detach().clone()
new_hand_pose[:, 9:] = joint_angle_targets_to_optimize
hand_model.set_parameters(new_hand_pose)
new_hand_model_plotly = hand_model.get_plotly_data(
    i=idx_to_visualize, opacity=1.0, with_contact_candidates=True
)

plots = [
    *new_hand_model_plotly,
    *object_model.get_plotly_data(i=idx_to_visualize, opacity=0.5),
]

for plot in plots:
    fig.append_trace(plot, row=1, col=2)

fig.update_layout(
    autosize=False,
    width=1600,
    height=800,
    title_text=f"Optimization Method: {joint_angle_targets_optimization_method.name}",
)
fig.show()
