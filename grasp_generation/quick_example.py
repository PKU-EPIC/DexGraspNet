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
    compute_optimized_joint_angle_targets,
    compute_optimized_joint_angle_targets_given_directions,
    compute_optimized_canonicalized_hand_pose,
)


# %% [markdown]
# ## PARAMS

# %%
mesh_path = "../data/meshdata"
data_path = "../data/graspdata_2023-05-24_allegro_distalonly/"
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
grasp_code_list = []
for filename in os.listdir(data_path):
    code = filename.split(".")[0]
    grasp_code_list.append(code)

# %% [markdown]
# ## Sample and read in data

# %%
grasp_code = random.choice(grasp_code_list)
grasp_data_list = np.load(os.path.join(data_path, grasp_code + ".npy"), allow_pickle=True)
print(f"Randomly sampled grasp_code = {grasp_code}")

index = random.randint(0, len(grasp_data_list) - 1)
qpos = grasp_data_list[index]["qpos"]
scale = grasp_data_list[index]["scale"]
print(f"Randomly sampled index = {index}")
print(f"scale = {scale}")

# %% [markdown]
# ## Object model

# %%
object_model = ObjectModel(
    meshdata_root_path=mesh_path,
    batch_size_each=1,
    scale=scale,
    device=device,
)
object_model.initialize([grasp_code])

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
object_mesh = object_model.object_mesh_list[batch_idx].copy().apply_scale(scale)

# %% [markdown]
# ## Visualize hand and object

# %%
(hand_mesh + object_mesh).show()

# %% [markdown]
# ## Visualize hand and object plotly

# %%
fig_title = f"Grasp Code: {grasp_code}, Index: {index}"
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
# ## Compute optimized canonicalized grasp

# %%
if should_canonicalize_hand_pose:
    (
        canonicalized_hand_pose,
        canonicalized_losses,
        canonicalized_debug_infos,
    ) = compute_optimized_canonicalized_hand_pose(
        hand_model=hand_model,
        object_model=object_model,
        device=device,
    )
    hand_model.set_parameters(canonicalized_hand_pose)

    fig_title = f"Canonicalized Grasp Code: {grasp_code}, Index: {index}"
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

    canonicalized_target_points = canonicalized_debug_infos[-1]["target_points"]
    canonicalized_contact_points_hand = canonicalized_debug_infos[-1]["contact_points_hand"]
    canonicalized_closest_points_object = (
        canonicalized_debug_infos[-1]['contact_points_hand']
        - canonicalized_debug_infos[-1]['contact_normals'] * canonicalized_debug_infos[-1]['contact_distances'][..., None]
    )

    plots = [
        *hand_model.get_plotly_data(
            i=idx_to_visualize, opacity=1.0, with_contact_candidates=True
        ),
        *object_model.get_plotly_data(i=idx_to_visualize, opacity=0.5),
        go.Scatter3d(
            x=canonicalized_target_points[batch_idx, :, 0].detach().cpu().numpy(),
            y=canonicalized_target_points[batch_idx, :, 1].detach().cpu().numpy(),
            z=canonicalized_target_points[batch_idx, :, 2].detach().cpu().numpy(),
            mode="markers",
            marker=dict(size=10, color="red"),
            name="target_points",
        ),
        go.Scatter3d(
            x=canonicalized_contact_points_hand[batch_idx, :, 0].detach().cpu().numpy(),
            y=canonicalized_contact_points_hand[batch_idx, :, 1].detach().cpu().numpy(),
            z=canonicalized_contact_points_hand[batch_idx, :, 2].detach().cpu().numpy(),
            mode="markers",
            marker=dict(size=10, color="green"),
            name="contact_points_hand",
        ),
        # Draw blue line between closest points and contact points
        *[
            go.Scatter3d(
                x=[
                    canonicalized_closest_points_object[batch_idx, i, 0].detach().cpu().numpy(),
                    canonicalized_contact_points_hand[batch_idx, i, 0].detach().cpu().numpy(),
                ],
                y=[
                    canonicalized_closest_points_object[batch_idx, i, 1].detach().cpu().numpy(),
                    canonicalized_contact_points_hand[batch_idx, i, 1].detach().cpu().numpy(),
                ],
                z=[
                    canonicalized_closest_points_object[batch_idx, i, 2].detach().cpu().numpy(),
                    canonicalized_contact_points_hand[batch_idx, i, 2].detach().cpu().numpy(),
                ],
                mode="lines",
                line=dict(color="blue", width=5),
                name="contact_point_to_closest_point",
            )
            for i in range(canonicalized_closest_points_object.shape[1])
        ],
    ]
    for plot in plots:
        fig.add_trace(plot)
    fig.show()

# %%
if should_canonicalize_hand_pose:
    fig = px.line(y=canonicalized_losses)
    fig.update_layout(
        title="Canonicalized Loss vs. Iterations", xaxis_title="Iterations", yaxis_title="Loss"
    )
    fig.show()



# %% [markdown]
# ## Compute optimized joint angle targets

# %%
original_hand_pose = hand_model.hand_pose.detach().clone()
print(f"original_hand_pose[:, 9:] = {original_hand_pose[:, 9:]}")

# %%
(
    joint_angle_targets_to_optimize,
    losses,
    debug_infos,
) = compute_optimized_joint_angle_targets(
    optimization_method=joint_angle_targets_optimization_method,
    hand_model=hand_model,
    object_model=object_model,
    device=device,
)
old_debug_info = debug_infos[0]
debug_info = debug_infos[-1]

# %%
fig = px.line(y=losses)
fig.update_layout(
    title=f"{joint_angle_targets_optimization_method} Loss vs. Iterations", xaxis_title="Iterations", yaxis_title="Loss"
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
old_target_points = old_debug_info["target_points"]
old_contact_points_hand = old_debug_info["contact_points_hand"]

plots = [
    *old_hand_model_plotly,
    *object_model.get_plotly_data(i=idx_to_visualize, opacity=0.5),
    go.Scatter3d(
        x=old_target_points[batch_idx, :, 0].detach().cpu().numpy(),
        y=old_target_points[batch_idx, :, 1].detach().cpu().numpy(),
        z=old_target_points[batch_idx, :, 2].detach().cpu().numpy(),
        mode="markers",
        marker=dict(size=10, color="red"),
        name="target_points",
    ),
    go.Scatter3d(
        x=old_contact_points_hand[batch_idx, :, 0].detach().cpu().numpy(),
        y=old_contact_points_hand[batch_idx, :, 1].detach().cpu().numpy(),
        z=old_contact_points_hand[batch_idx, :, 2].detach().cpu().numpy(),
        mode="markers",
        marker=dict(size=10, color="green"),
        name="contact_points_hand",
    ),
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

new_target_points = debug_info["target_points"]
new_contact_points_hand = debug_info["contact_points_hand"]

plots = [
    *new_hand_model_plotly,
    *object_model.get_plotly_data(i=idx_to_visualize, opacity=0.5),
    go.Scatter3d(
        x=new_target_points[batch_idx, :, 0].detach().cpu().numpy(),
        y=new_target_points[batch_idx, :, 1].detach().cpu().numpy(),
        z=new_target_points[batch_idx, :, 2].detach().cpu().numpy(),
        mode="markers",
        marker=dict(size=10, color="red"),
        name="new_target_points",
    ),
    go.Scatter3d(
        x=new_contact_points_hand[batch_idx, :, 0].detach().cpu().numpy(),
        y=new_contact_points_hand[batch_idx, :, 1].detach().cpu().numpy(),
        z=new_contact_points_hand[batch_idx, :, 2].detach().cpu().numpy(),
        mode="markers",
        marker=dict(size=10, color="green"),
        name="contact_points_hand",
    ),
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

# %% [markdown]
# ## Compute optimized joint angle targets given directions

# %%
hand_model.set_parameters(original_hand_pose)
print(f"original_hand_pose[:, 9:] = {original_hand_pose[:, 9:]}")

# %%
batch_size = hand_model.batch_size
num_fingers = hand_model.num_fingers
num_xyz = 3
GRASP_DIRS_ARRAY = torch.stack([
    2 * torch.rand(num_fingers, num_xyz, device=device) - 1
    for _ in range(batch_size)
], dim=0)

# %%
(
    joint_angle_targets_to_optimize,
    losses,
    debug_infos,
) = compute_optimized_joint_angle_targets_given_directions(
    hand_model=hand_model,
    grasp_dirs_array=GRASP_DIRS_ARRAY,
    dist_move_link=0.02,
)
old_debug_info = debug_infos[0]
debug_info = debug_infos[-1]

# %%
fig = px.line(y=losses)
fig.update_layout(
    title=f"{joint_angle_targets_optimization_method} Loss vs. Iterations", xaxis_title="Iterations", yaxis_title="Loss"
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
old_target_points = old_debug_info["target_points"]
old_contact_points_hand = old_debug_info["contact_points_hand"]

plots = [
    *old_hand_model_plotly,
    *object_model.get_plotly_data(i=idx_to_visualize, opacity=0.5),
    go.Scatter3d(
        x=old_target_points[batch_idx, :, 0].detach().cpu().numpy(),
        y=old_target_points[batch_idx, :, 1].detach().cpu().numpy(),
        z=old_target_points[batch_idx, :, 2].detach().cpu().numpy(),
        mode="markers",
        marker=dict(size=10, color="red"),
        name="target_points",
    ),
    go.Scatter3d(
        x=old_contact_points_hand[batch_idx, :, 0].detach().cpu().numpy(),
        y=old_contact_points_hand[batch_idx, :, 1].detach().cpu().numpy(),
        z=old_contact_points_hand[batch_idx, :, 2].detach().cpu().numpy(),
        mode="markers",
        marker=dict(size=10, color="green"),
        name="contact_points_hand",
    ),
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

new_target_points = debug_info["target_points"]
new_contact_points_hand = debug_info["contact_points_hand"]

plots = [
    *new_hand_model_plotly,
    *object_model.get_plotly_data(i=idx_to_visualize, opacity=0.5),
    go.Scatter3d(
        x=new_target_points[batch_idx, :, 0].detach().cpu().numpy(),
        y=new_target_points[batch_idx, :, 1].detach().cpu().numpy(),
        z=new_target_points[batch_idx, :, 2].detach().cpu().numpy(),
        mode="markers",
        marker=dict(size=10, color="red"),
        name="new_target_points",
    ),
    go.Scatter3d(
        x=new_contact_points_hand[batch_idx, :, 0].detach().cpu().numpy(),
        y=new_contact_points_hand[batch_idx, :, 1].detach().cpu().numpy(),
        z=new_contact_points_hand[batch_idx, :, 2].detach().cpu().numpy(),
        mode="markers",
        marker=dict(size=10, color="green"),
        name="contact_points_hand",
    ),
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

# %%
