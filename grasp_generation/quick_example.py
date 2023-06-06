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
data_path = "../data/graspdata_2023-05-24_allegro_distalonly/"
hand_model_type = HandModelType.ALLEGRO_HAND
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# Grasp codes
grasp_code_list = []
for filename in os.listdir(data_path):
    code = filename.split(".")[0]
    grasp_code_list.append(code)

# %%
# Sample and read in data
grasp_code = random.choice(grasp_code_list)
grasp_data = np.load(os.path.join(data_path, grasp_code + ".npy"), allow_pickle=True)
print(f"Randomly sampled grasp_code = {grasp_code}")

index = random.randint(0, len(grasp_data) - 1)
qpos = grasp_data[index]["qpos"]
scale = grasp_data[index]["scale"]
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

hand_pose = qpos_to_pose(
    qpos=qpos, joint_names=joint_names, unsqueeze_batch_dim=True
).to(device)
hand_model.set_parameters(hand_pose)

batch_idx = 0
hand_mesh = hand_model.get_trimesh_data(batch_idx)
object_mesh = object_model.object_mesh_list[batch_idx].copy().apply_scale(scale)

# %%
(hand_mesh + object_mesh).show()

# %%
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# %%
# Plotly fig
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

# %%
original_hand_pose = hand_model.hand_pose.detach().clone()
print(f"original_hand_pose[:, 9:] = {original_hand_pose[:, 9:]}")
joint_angle_targets_to_optimize = (
    original_hand_pose[:, 9:].detach().clone().requires_grad_(True)
)

# %%
from DUMMY import compute_loss_desired_penetration_dist, compute_loss_desired_dist_move
from enum import Enum, auto


class OptimizationMethod(Enum):
    DESIRED_PENETRATION_DIST = auto()
    DESIRED_DIST_MOVE_ONE_STEP = auto()
    DESIRED_DIST_MOVE_MULTIPLE_STEPS = auto()


optimization_method = OptimizationMethod.DESIRED_DIST_MOVE_ONE_STEP

losses = []
old_debug_info = None
if optimization_method == OptimizationMethod.DESIRED_PENETRATION_DIST:
    N_ITERS = 100
    for i in range(N_ITERS):
        loss, debug_info = compute_loss_desired_penetration_dist(
            joint_angle_targets_to_optimize=joint_angle_targets_to_optimize,
            hand_model=hand_model,
            object_model=object_model,
            batch_size=1,
            device=device,
            dist_thresh_to_move_finger=0.01,
            desired_penetration_dist=0.003,
            return_debug_info=True,
        )
        grad_step_size = 50

        if old_debug_info is None:
            old_debug_info = debug_info
        loss.backward(retain_graph=True)

        with torch.no_grad():
            joint_angle_targets_to_optimize -= (
                joint_angle_targets_to_optimize.grad * grad_step_size
            )
            joint_angle_targets_to_optimize.grad.zero_()
        losses.append(loss.item())

elif optimization_method == OptimizationMethod.DESIRED_DIST_MOVE_ONE_STEP:
    N_ITERS = 1
    for i in range(N_ITERS):
        loss, debug_info = compute_loss_desired_dist_move(
            joint_angle_targets_to_optimize=joint_angle_targets_to_optimize,
            hand_model=hand_model,
            object_model=object_model,
            batch_size=1,
            device=device,
            dist_thresh_to_move_finger=0.001,
            dist_move_link=0.001,
            return_debug_info=True,
        )
        grad_step_size = 500

        if old_debug_info is None:
            old_debug_info = debug_info
        loss.backward(retain_graph=True)

        with torch.no_grad():
            joint_angle_targets_to_optimize -= (
                joint_angle_targets_to_optimize.grad * grad_step_size
            )
            joint_angle_targets_to_optimize.grad.zero_()
        losses.append(loss.item())

elif optimization_method == OptimizationMethod.DESIRED_DIST_MOVE_MULTIPLE_STEPS:
    N_ITERS = 100
    # Use cached target and indices to continue moving the same points toward the same targets for each iter
    # Otherwise, would be moving different points to different targets each iter
    cached_target_points = None
    cached_contact_nearest_point_indexes = None
    for i in range(N_ITERS):
        loss, debug_info = compute_loss_desired_dist_move(
            joint_angle_targets_to_optimize=joint_angle_targets_to_optimize,
            hand_model=hand_model,
            object_model=object_model,
            batch_size=1,
            device=device,
            cached_target_points=cached_target_points,
            cached_contact_nearest_point_indexes=cached_contact_nearest_point_indexes,
            dist_thresh_to_move_finger=0.01,
            dist_move_link=0.01,
            return_debug_info=True,
        )
        if cached_target_points is None:
            cached_target_points = debug_info["target_points"]
        if cached_contact_nearest_point_indexes is None:
            cached_contact_nearest_point_indexes = debug_info[
                "contact_nearest_point_indexes"
            ]
        grad_step_size = 5

        if old_debug_info is None:
            old_debug_info = debug_info
        loss.backward(retain_graph=True)

        with torch.no_grad():
            joint_angle_targets_to_optimize -= (
                joint_angle_targets_to_optimize.grad * grad_step_size
            )
            joint_angle_targets_to_optimize.grad.zero_()
        losses.append(loss.item())

else:
    raise NotImplementedError

# %%
import plotly.express as px

px.line(y=losses)

# %%
print(f"joint_angle_targets_to_optimize = {joint_angle_targets_to_optimize}")


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
    title_text=f"Optimization Method: {optimization_method.name}",
    # scene1=dict(title="Original"),
    # scene2=dict(title="Optimized"),
)
fig.show()


# %%
