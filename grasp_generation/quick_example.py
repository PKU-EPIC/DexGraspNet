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
from scripts.validate_grasps import set_seed
import os
import random
from utils.hand_model import HandModel
from utils.object_model import ObjectModel
import numpy as np
import transforms3d
import torch
import trimesh
from utils.hand_model_type import handmodeltype_to_joint_names, HandModelType
from utils.qpos_pose_conversion import qpos_to_pose


# %%
set_seed(42)

# %%
# PARAMS
mesh_path = "../data/meshdata"
data_path = "../data/dataset"
hand_model_type = HandModelType.SHADOW_HAND
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# Grasp codes
grasp_code_list = []
for filename in os.listdir(data_path):
    code = filename.split('.')[0]
    grasp_code_list.append(code)

# %%
# Sample and read in data
grasp_code = random.choice(grasp_code_list)
grasp_data = np.load(os.path.join(data_path, grasp_code+".npy"), allow_pickle=True)
print(f"Randomly sampled grasp_code = {grasp_code}")

index = random.randint(0, len(grasp_data) - 1)
qpos = grasp_data[index]['qpos']
scale = grasp_data[index]['scale']
print(f"Randomly sampled index = {index}")
print(f"scale = {scale}")

# %%
# Object model
object_model = ObjectModel(
    data_root_path=mesh_path,
    batch_size_each=1,
    device=device,
)
object_model.initialize([grasp_code])
object_model.object_scale_tensor = torch.tensor(
    scale, dtype=torch.float, device=device
).reshape(object_model.object_scale_tensor.shape)

# %%
# Hand model
joint_names = handmodeltype_to_joint_names[hand_model_type]
hand_model = HandModel(hand_model_type, device=device)

hand_pose = qpos_to_pose(qpos=qpos, joint_names=joint_names, unsqueeze_batch_dim=True).to(device)
hand_model.set_parameters(hand_pose)

batch_idx = 0
hand_mesh = hand_model.get_trimesh_data(batch_idx)
object_mesh = object_model.object_mesh_list[batch_idx].copy().apply_scale(scale)

# %%
(hand_mesh+object_mesh).show()

# %%
# import plotly to vis mesh
import plotly.graph_objects as go

# %%
# Plotly fig
fig_title = f"Grasp Code: {grasp_code}, Index: {index}"
idx_to_visualize = batch_idx

fig = go.Figure(
    layout=go.Layout(
        scene=dict(xaxis=dict(title="X"), yaxis=dict(title="Y"), zaxis=dict(title="Z"), aspectmode="data"),
        showlegend=True,
        title=fig_title,
        autosize=False,
        width=800,
        height=800,
    )
)
plots = [
    *hand_model.get_plotly_data(i=idx_to_visualize, opacity=1.0, with_contact_candidates=True),
    *object_model.get_plotly_data(i=idx_to_visualize, opacity=0.5),
]
for plot in plots:
    fig.add_trace(plot)
fig.show()

# %%
original_hand_pose = hand_model.hand_pose.detach().clone()
print(f"original_hand_pose[:, 9:] = {original_hand_pose[:, 9:]}")

# %%
joint_angle_targets_to_optimize = original_hand_pose[:, 9:].detach().clone().requires_grad_(True)

# %%
batch_size = 1
num_links = len(hand_model.mesh)
contact_points_hand = torch.zeros((batch_size, num_links, 3)).to(device)
contact_normals = torch.zeros((batch_size, num_links, 3)).to(device)
contact_distances = torch.zeros((batch_size, num_links)).to(device)

from utils.hand_model_type import handmodeltype_to_expectedcontactlinknames
expected_contact_link_names = handmodeltype_to_expectedcontactlinknames[hand_model_type]
dist_thresh_to_move_finger = 0.01
desired_penetration_dist = 0.003
grad_step_size = 50

current_status = hand_model.chain.forward_kinematics(
    joint_angle_targets_to_optimize
)
for i, link_name in enumerate(hand_model.mesh):
    surface_points = hand_model.mesh[link_name]["contact_candidates"]
    if len(surface_points) == 0:
        continue
    if link_name not in expected_contact_link_names:
        continue

    surface_points = (
        current_status[link_name]
        .transform_points(surface_points)
        .expand(batch_size, -1, 3)
    )
    surface_points = surface_points @ hand_model.global_rotation.transpose(
        1, 2
    ) + hand_model.global_translation.unsqueeze(1)

    # Interiors are positive dist, exteriors are negative dist
    # Normals point from object to hand
    distances, normals = object_model.cal_distance(surface_points)
    nearest_point_index = distances.argmax(dim=1)
    nearest_distances = torch.gather(distances, 1, nearest_point_index.unsqueeze(1))
    nearest_points_hand = torch.gather(
        surface_points, 1, nearest_point_index.reshape(-1, 1, 1).expand(-1, 1, 3)
    )
    nearest_normals = torch.gather(
        normals, 1, nearest_point_index.reshape(-1, 1, 1).expand(-1, 1, 3)
    )
    admited = -nearest_distances < dist_thresh_to_move_finger
    contact_distances[:, i : i + 1] = torch.where(
        admited, -nearest_distances, contact_distances[:, i : i + 1]
    )
    admited = admited.reshape(-1, 1, 1).expand(-1, 1, 3)
    contact_points_hand[:, i : i + 1, :] = torch.where(
        admited, nearest_points_hand, contact_points_hand[:, i : i + 1, :]
    )
    contact_normals[:, i : i + 1, :] = torch.where(
        admited, nearest_normals, contact_normals[:, i : i + 1, :]
    )

target_points = contact_points_hand - contact_normals * (contact_distances[..., None] + desired_penetration_dist)

loss = (target_points.detach().clone() - contact_points_hand).square().sum()
print(f"Before step, loss = {loss.item()}")
loss.backward(retain_graph=True)
with torch.no_grad():
    joint_angle_targets_to_optimize -= joint_angle_targets_to_optimize.grad * grad_step_size
# print(f"After step, loss = {loss.item()}")

# %%
print(f"joint_angle_targets_to_optimize = {joint_angle_targets_to_optimize}")


# %%
# Plotly fig
new_hand_pose = hand_model.hand_pose.detach().clone()
new_hand_pose[:, 9:] = joint_angle_targets_to_optimize
hand_model.set_parameters(new_hand_pose)
fig_title = f"Grasp Code: {grasp_code}, Index: {index}"
idx_to_visualize = batch_idx

fig = go.Figure(
    layout=go.Layout(
        scene=dict(xaxis=dict(title="X"), yaxis=dict(title="Y"), zaxis=dict(title="Z"), aspectmode="data"),
        showlegend=True,
        title=fig_title,
        autosize=False,
        width=800,
        height=800,
    )
)
plots = [
    *hand_model.get_plotly_data(i=idx_to_visualize, opacity=1.0, with_contact_candidates=True),
    *object_model.get_plotly_data(i=idx_to_visualize, opacity=0.5),
]
plots += [
    go.Scatter3d(
        x=target_points[batch_idx, :, 0].detach().cpu().numpy(),
        y=target_points[batch_idx, :, 1].detach().cpu().numpy(),
        z=target_points[batch_idx, :, 2].detach().cpu().numpy(),
        mode="markers",
        marker=dict(size=10, color="red"),
        name="target_points",
    ),
    go.Scatter3d(
        x=contact_points_hand[batch_idx, :, 0].detach().cpu().numpy(),
        y=contact_points_hand[batch_idx, :, 1].detach().cpu().numpy(),
        z=contact_points_hand[batch_idx, :, 2].detach().cpu().numpy(),
        mode="markers",
        marker=dict(size=10, color="green"),
        name="contact_points_hand",
    ),
]
for plot in plots:
    fig.add_trace(plot)
fig.show()

# %%
