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
from utils.joint_angle_targets import OptimizationMethod, compute_optimized_joint_angle_targets


# %% [markdown]
# ## PARAMS

# %%
mesh_path = "../data/meshdata"
data_path = "../data/graspdata_2023-05-24_allegro_distalonly/"
hand_model_type = HandModelType.ALLEGRO_HAND
seed = 42
joint_angle_targets_optimization_method = OptimizationMethod.DESIRED_DIST_MOVE_TOWARDS_CENTER_ONE_STEP
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
grasp_data = np.load(os.path.join(data_path, grasp_code + ".npy"), allow_pickle=True)
print(f"Randomly sampled grasp_code = {grasp_code}")

index = random.randint(0, len(grasp_data) - 1)
qpos = grasp_data[index]["qpos"]
scale = grasp_data[index]["scale"]
print(f"Randomly sampled index = {index}")
print(f"scale = {scale}")

# %% [markdown]
# ## Object model

# %%
object_model = ObjectModel(
    data_root_path=mesh_path,
    batch_size_each=1,
    device=device,
)
object_model.initialize([grasp_code])
object_model.object_scale_tensor = torch.tensor(
    scale, dtype=torch.float, device=device
).reshape(object_model.object_scale_tensor.shape)

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
from plotly.subplots import make_subplots
import plotly.graph_objects as go

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
# ## Compute optimized joint angle targets

# %%
# Try to optimize grasp so that all fingers are at a fixed distance from the object
from utils.joint_angle_targets import compute_loss_desired_penetration_dist

original_hand_pose = hand_model.hand_pose.detach().clone()
joint_angle_targets_to_optimize = (
    original_hand_pose.detach().clone()[:, 9:].requires_grad_(True)
)

losses = []
debug_infos = []
N_ITERS = 100
for i in range(N_ITERS):
    loss, debug_info = compute_loss_desired_penetration_dist(
        joint_angle_targets_to_optimize=joint_angle_targets_to_optimize,
        hand_model=hand_model,
        object_model=object_model,
        device=device,
        dist_thresh_to_move_finger=0.03,
        desired_penetration_dist=-0.02,
        return_debug_info=True,
    )
    grad_step_size = 50
    loss.backward(retain_graph=True)

    with torch.no_grad():
        joint_angle_targets_to_optimize -= (
            joint_angle_targets_to_optimize.grad * grad_step_size
        )
        joint_angle_targets_to_optimize.grad.zero_()
    losses.append(loss.item())
    debug_infos.append(debug_info)

# Update hand pose parameters
new_hand_pose = original_hand_pose.detach().clone()
new_hand_pose[:, 9:] = joint_angle_targets_to_optimize
hand_model.set_parameters(new_hand_pose)

# %%
# Plotly fig
hand_model.set_parameters(new_hand_pose)
TYLER_hand_model_plotly = hand_model.get_plotly_data(
    i=idx_to_visualize, opacity=1.0, with_contact_candidates=True
)

TYLER_target_points = debug_infos[-1]["target_points"]
TYLER_contact_points_hand = debug_infos[-1]["contact_points_hand"]
TYLER_closest_points = debug_infos[-1]['contact_points_hand'] - debug_infos[-1]['contact_normals'] * (
    debug_infos[-1]['contact_distances'][..., None]
)

plots = [
    *TYLER_hand_model_plotly,
    *object_model.get_plotly_data(i=idx_to_visualize, opacity=0.5),
    go.Scatter3d(
        x=TYLER_target_points[batch_idx, :, 0].detach().cpu().numpy(),
        y=TYLER_target_points[batch_idx, :, 1].detach().cpu().numpy(),
        z=TYLER_target_points[batch_idx, :, 2].detach().cpu().numpy(),
        mode="markers",
        marker=dict(size=10, color="red"),
        name="target_points",
    ),
    go.Scatter3d(
        x=TYLER_contact_points_hand[batch_idx, :, 0].detach().cpu().numpy(),
        y=TYLER_contact_points_hand[batch_idx, :, 1].detach().cpu().numpy(),
        z=TYLER_contact_points_hand[batch_idx, :, 2].detach().cpu().numpy(),
        mode="markers",
        marker=dict(size=10, color="green"),
        name="contact_points_hand",
    ),
    # Draw blue line between closest points and contact points
    *[
        go.Scatter3d(
            x=[
                TYLER_closest_points[batch_idx, i, 0].detach().cpu().numpy(),
                TYLER_contact_points_hand[batch_idx, i, 0].detach().cpu().numpy(),
            ],
            y=[
                TYLER_closest_points[batch_idx, i, 1].detach().cpu().numpy(),
                TYLER_contact_points_hand[batch_idx, i, 1].detach().cpu().numpy(),
            ],
            z=[
                TYLER_closest_points[batch_idx, i, 2].detach().cpu().numpy(),
                TYLER_contact_points_hand[batch_idx, i, 2].detach().cpu().numpy(),
            ],
            mode="lines",
            line=dict(color="blue", width=5),
            name="contact_point_to_closest_point",
        )
        for i in range(TYLER_closest_points.shape[1])
    ],
]

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
for plot in plots:
    fig.add_trace(plot)
fig.show()


# %%

import plotly.express as px

fig = px.line(y=losses)
fig.update_layout(
    title="Loss vs. Iterations",
    xaxis_title="Iterations",
    yaxis_title="Loss"
)
fig.show()
