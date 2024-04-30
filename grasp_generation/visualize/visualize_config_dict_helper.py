import os
import sys

# os.chdir(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import plotly.graph_objects as go
from typing import Dict, Any, Optional
import numpy as np

from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.hand_model_type import handmodeltype_to_joint_names
from utils.pose_conversion import hand_config_to_pose
from utils.joint_angle_targets import (
    compute_fingertip_targets,
    compute_fingertip_mean_contact_positions,
    compute_optimized_joint_angle_targets_given_fingertip_targets,
)


def get_hand_config_dict_plotly_data_list(
    hand_model: HandModel,
    hand_pose: torch.Tensor,
    hand_pose_start: Optional[torch.Tensor],
) -> list:
    if hand_pose_start is not None:
        hand_model.set_parameters(hand_pose_start)
        hand_start_plotly = hand_model.get_plotly_data(
            i=0,
            opacity=0.5,
            color="lightblue",
            with_contact_points=False,
            with_contact_candidates=True,
            with_surface_points=True,
            with_penetration_keypoints=True,
        )
    else:
        hand_start_plotly = []

    hand_model.set_parameters(hand_pose)
    hand_plotly = hand_model.get_plotly_data(
        i=0,
        opacity=1,
        color="lightblue",
        with_contact_points=False,
        with_contact_candidates=True,
        with_surface_points=True,
        with_penetration_keypoints=False,
    )

    return hand_start_plotly + hand_plotly


def get_grasp_config_dict_plotly_data_list(
    hand_model: HandModel,
    hand_pose: torch.Tensor,
    config_dict: Dict[str, np.ndarray],
    idx_to_visualize: int,
    device: str,
) -> list:
    if "grasp_orientations" not in config_dict:
        print(
            f"This is not a grasp_config_dict, skipping get_grasp_config_dict_plotly_data_list"
        )
        return []

    wrist_pose = hand_pose[:, :9]
    joint_angles = hand_pose[:, 9:]

    # fingertips
    fingertip_mean_positions = compute_fingertip_mean_contact_positions(
        joint_angles=joint_angles,
        hand_model=hand_model,
    )
    assert fingertip_mean_positions.shape == (1, hand_model.num_fingers, 3)
    fingertips_plotly = [
        go.Scatter3d(
            x=fingertip_mean_positions[0, :, 0],
            y=fingertip_mean_positions[0, :, 1],
            z=fingertip_mean_positions[0, :, 2],
            mode="markers",
            marker=dict(size=7, color="goldenrod"),
            name="fingertip mean positions",
        ),
    ]

    # fingertip targets
    grasp_orientations = torch.tensor(
        config_dict["grasp_orientations"], dtype=torch.float, device=device
    )[idx_to_visualize]
    assert grasp_orientations.shape == (hand_model.num_fingers, 3, 3)
    fingertip_targets = compute_fingertip_targets(
        joint_angles_start=joint_angles,
        hand_model=hand_model,
        grasp_orientations=grasp_orientations.unsqueeze(dim=0),
    )
    assert fingertip_targets.shape == (1, hand_model.num_fingers, 3)
    fingertip_targets_plotly = [
        go.Scatter3d(
            x=fingertip_targets[0, :, 0],
            y=fingertip_targets[0, :, 1],
            z=fingertip_targets[0, :, 2],
            mode="markers",
            marker=dict(size=10, color="magenta"),
            name="fingertip targets",
        ),
    ]

    # grasp_orientations
    grasp_orientations_plotly = []
    for i in range(hand_model.num_fingers):
        origin = fingertip_mean_positions[0, i]
        line_length = 0.01
        x_dir = grasp_orientations[i, :, 0] * line_length
        y_dir = grasp_orientations[i, :, 1] * line_length
        z_dir = grasp_orientations[i, :, 2] * line_length
        grasp_orientations_plotly += [
            go.Scatter3d(
                x=[origin[0], origin[0] + x_dir[0]],
                y=[origin[1], origin[1] + x_dir[1]],
                z=[origin[2], origin[2] + x_dir[2]],
                mode="lines",
                marker=dict(size=5, color="red"),
                name=f"x_dir for finger {i}",
            ),
            go.Scatter3d(
                x=[origin[0], origin[0] + y_dir[0]],
                y=[origin[1], origin[1] + y_dir[1]],
                z=[origin[2], origin[2] + y_dir[2]],
                mode="lines",
                marker=dict(size=5, color="green"),
                name=f"y_dir for finger {i}",
            ),
            go.Scatter3d(
                x=[origin[0], origin[0] + z_dir[0]],
                y=[origin[1], origin[1] + z_dir[1]],
                z=[origin[2], origin[2] + z_dir[2]],
                mode="lines",
                marker=dict(size=5, color="blue"),
                name=f"z_dir for finger {i}",
            ),
        ]

    # joint angle targets
    (
        joint_angle_targets,
        _,
    ) = compute_optimized_joint_angle_targets_given_fingertip_targets(
        joint_angles_start=joint_angles,
        hand_model=hand_model,
        fingertip_targets=fingertip_targets,
    )
    hand_pose_target = torch.cat(
        [
            wrist_pose,
            joint_angle_targets,
        ],
        dim=1,
    )
    hand_model.set_parameters(hand_pose_target)
    hand_target_plotly = hand_model.get_plotly_data(
        i=0,
        opacity=0.5,
        color="lightblue",
        with_contact_points=False,
        with_contact_candidates=False,
        with_surface_points=False,
        with_penetration_keypoints=False,
    )
    return (
        fingertips_plotly
        + fingertip_targets_plotly
        + grasp_orientations_plotly
        + hand_target_plotly
    )


def create_config_dict_fig(
    config_dict: Dict[str, Any],
    hand_model: HandModel,
    object_model: Optional[ObjectModel],
    skip_visualize_qpos_start: bool,
    skip_visualize_grasp_config_dict: bool,
    title: str,
    idx_to_visualize: int,
    concise_title: bool = False,
) -> go.Figure:
    if object_model is not None:
        object_plotly = object_model.get_plotly_data(
            i=0, color="lightgreen", opacity=0.5, with_surface_points=True, with_table=True
        )
    else:
        object_plotly = []

    # hand pose
    hand_pose = (
        hand_config_to_pose(
            trans=config_dict["trans"],
            rot=config_dict["rot"],
            joint_angles=config_dict["joint_angles"],
        )[idx_to_visualize]
        .to(hand_model.device)
        .unsqueeze(0)
    )

    # hand pose start
    if "joint_angles_start" in config_dict and not skip_visualize_qpos_start:
        hand_pose_start = hand_config_to_pose(
            trans=config_dict["trans_start"],
            rot=config_dict["rot_start"],
            joint_angles=config_dict["joint_angles_start"],
        ).to(hand_model.device)
    else:
        hand_pose_start = None

    # hand config dict
    hand_config_dict_plotly_data_list = get_hand_config_dict_plotly_data_list(
        hand_model=hand_model,
        hand_pose=hand_pose,
        hand_pose_start=hand_pose_start,
    )

    # grasp config dict
    if not skip_visualize_grasp_config_dict:
        # Slowest part of this function
        grasp_config_dict_plotly_data_list = get_grasp_config_dict_plotly_data_list(
            hand_model=hand_model,
            hand_pose=hand_pose,
            config_dict=config_dict,
            idx_to_visualize=idx_to_visualize,
            device=hand_model.device,
        )
    else:
        grasp_config_dict_plotly_data_list = []

    # Create fig
    fig = go.Figure(
        data=(
            object_plotly
            + hand_config_dict_plotly_data_list
            + grasp_config_dict_plotly_data_list
        )
    )

    # energy
    if "energy" in config_dict:
        energy = config_dict["energy"]
        energy_terms_to_values = {
            key: round(value[idx_to_visualize], 3)
            for key, value in config_dict.items()
            if key.startswith("E_")
        }

        energy_terms_str = "\n  ".join(
            [f"{key}: {value}" for key, value in energy_terms_to_values.items()]
        )
        energy_str = f"Energy: {energy}\n  {energy_terms_str}"
        fig.add_annotation(text=energy_str, x=0.5, y=0.1, xref="paper", yref="paper")

    # passed_eval
    if "passed_eval" in config_dict:
        passed_eval = config_dict["passed_eval"][idx_to_visualize]
        passed_eval_str = (
            f"Passed eval: {passed_eval}" if not concise_title else f"E: {passed_eval}"
        )
        fig.add_annotation(
            text=passed_eval_str, x=0.5, y=0.05, xref="paper", yref="paper"
        )
        # For some reason, annotations not showing up in the multi fig plot
        title += f" | {passed_eval_str}"

    if "passed_penetration_threshold" in config_dict:
        passed_penetration_threshold = config_dict["passed_penetration_threshold"][
            idx_to_visualize
        ]
        passed_penetration_threshold_str = (
            f"Passed penetration threshold: {passed_penetration_threshold}"
            if not concise_title
            else f"P: {passed_penetration_threshold}"
        )
        fig.add_annotation(
            text=passed_penetration_threshold_str,
            x=0.5,
            y=0.1,
            xref="paper",
            yref="paper",
        )
        # For some reason, annotations not showing up in the multi fig plot
        title += f" | {passed_penetration_threshold_str}"

    if "penetration" in config_dict:
        penetration = config_dict["penetration"][idx_to_visualize]
        penetration_str = (
            f"Penetration: {round(penetration, 5)}" if not concise_title else ""
        )
        fig.add_annotation(
            text=penetration_str, x=0.5, y=0.15, xref="paper", yref="paper"
        )
        # For some reason, annotations not showing up in the multi fig plot
        title += f" | {penetration_str}"

    if "passed_simulation" in config_dict:
        passed_simulation = config_dict["passed_simulation"][idx_to_visualize]
        passed_simulation_str = (
            f"Passed simulation: {passed_simulation}"
            if not concise_title
            else f"S: {passed_simulation}"
        )
        fig.add_annotation(
            text=passed_simulation_str, x=0.5, y=0.2, xref="paper", yref="paper"
        )
        # For some reason, annotations not showing up in the multi fig plot
        title += f" | {passed_simulation_str}"

    # loss
    if "loss" in config_dict:
        loss = round(config_dict["loss"][idx_to_visualize], 3)
        predicted_score = 1 - loss
        predicted_score_str = f"predicted: {predicted_score}"
        fig.add_annotation(text=predicted_score_str, x=0.5, y=0.25, xref="paper", yref="paper")
        title += f" | {predicted_score_str}"

    # yup
    yup_camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.0, y=1.0, z=0.0)
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z"),
            aspectmode="data",
        ),
        scene_camera=yup_camera,
    )
    return fig
