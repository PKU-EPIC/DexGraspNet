"""
Last modified date: 2023.02.23
Author: Jialiang Zhang
Description: visualize grasp result using plotly.graph_objects
"""

import os
import sys

# os.chdir(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from tap import Tap
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, List, Tuple

from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.hand_model_type import handmodeltype_to_joint_names, HandModelType
from utils.qpos_pose_conversion import qpos_to_pose
import pathlib
from utils.joint_angle_targets import (
    computer_fingertip_targets,
    compute_fingertip_mean_contact_positions,
    compute_optimized_joint_angle_targets_given_fingertip_targets,
)

from utils.parse_object_code_and_scale import (
    parse_object_code_and_scale,
)


class VisualizeGraspConfigDictArgumentParser(Tap):
    hand_model_type: HandModelType = HandModelType.ALLEGRO_HAND
    input_grasp_config_dicts_path: pathlib.Path = pathlib.Path(
        "../data/grasp_config_dicts"
    )
    object_code_and_scale_str: str = "sem-Ipod-4b6c6248d5c01b3e4eee8d1cb32988b_0_10"
    idx_to_visualize: int = 0
    visualize_joint_angle_targets: bool = False
    save_to_html: bool = False


def main(args: VisualizeGraspConfigDictArgumentParser):
    device = "cpu"

    joint_names = handmodeltype_to_joint_names[args.hand_model_type]
    object_code, object_scale = parse_object_code_and_scale(
        args.object_code_and_scale_str
    )

    # load results
    grasp_config_dicts: List[Dict[str, Any]] = np.load(
        args.input_grasp_config_dicts_path / f"{args.object_code_and_scale_str}.npy",
        allow_pickle=True,
    )
    grasp_config_dict = grasp_config_dicts[args.idx_to_visualize]
    hand_pose = qpos_to_pose(
        qpos=grasp_config_dict["qpos"], joint_names=joint_names, unsqueeze_batch_dim=True
    ).to(device)
    hand_pose_start = (
        qpos_to_pose(
            qpos=grasp_config_dict["qpos_start"],
            joint_names=joint_names,
            unsqueeze_batch_dim=True,
        ).to(device)
        if "qpos_start" in grasp_config_dict
        else None
    )

    # hand model
    hand_model = HandModel(hand_model_type=args.hand_model_type, device=device)

    # object model
    object_model = ObjectModel(
        meshdata_root_path="../data/meshdata",
        batch_size_each=1,
        scale=object_scale,
        num_samples=2000,
        device=device,
    )
    object_model.initialize(object_code)

    # visualize
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
    object_plotly = object_model.get_plotly_data(
        i=0, color="lightgreen", opacity=0.5, with_surface_points=True
    )

    # Add grasp_orientations
    grasp_orientations = torch.tensor(
        grasp_config_dict["grasp_orientations"], dtype=torch.float, device=device
    )
    assert grasp_orientations.shape == (hand_model.num_fingers, 3, 3)
    fingertip_mean_positions = compute_fingertip_mean_contact_positions(
        joint_angles=hand_pose[:, 9:],
        hand_model=hand_model,
    )
    assert fingertip_mean_positions.shape == (1, hand_model.num_fingers, 3)
    fingertip_targets = computer_fingertip_targets(
        joint_angles_start=hand_pose[:, 9:],
        hand_model=hand_model,
        grasp_orientations=grasp_orientations.unsqueeze(dim=0),
    )
    assert fingertip_targets.shape == (1, hand_model.num_fingers, 3)
    fingertips_plotly = [
        go.Scatter3d(
            x=fingertip_mean_positions[0, :, 0],
            y=fingertip_mean_positions[0, :, 1],
            z=fingertip_mean_positions[0, :, 2],
            mode="markers",
            marker=dict(size=7, color="goldenrod"),
            name="fingertip mean positions",
        ),
        go.Scatter3d(
            x=fingertip_targets[0, :, 0],
            y=fingertip_targets[0, :, 1],
            z=fingertip_targets[0, :, 2],
            mode="markers",
            marker=dict(size=10, color="magenta"),
            name="fingertip targets",
        ),
    ]
    for i in range(hand_model.num_fingers):
        origin = fingertip_mean_positions[0, i]
        line_length = 0.01
        x_dir = grasp_orientations[i, :, 0] * line_length
        y_dir = grasp_orientations[i, :, 1] * line_length
        z_dir = grasp_orientations[i, :, 2] * line_length
        fingertips_plotly += [
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

    # Add joint angle targets
    if args.visualize_joint_angle_targets:
        (
            joint_angle_targets,
            _,
        ) = compute_optimized_joint_angle_targets_given_fingertip_targets(
            joint_angles_start=hand_pose[:, 9:],
            hand_model=hand_model,
            fingertip_targets=fingertip_targets,
        )

        hand_pose_target = torch.cat(
            [
                hand_pose[:, :9],
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
    else:
        hand_target_plotly = []

    fig = go.Figure(
        hand_start_plotly
        + hand_plotly
        + object_plotly
        + fingertips_plotly
        + hand_target_plotly
    )
    if "energy" in grasp_config_dict:
        energy = grasp_config_dict["energy"]
        E_fc = round(grasp_config_dict["E_fc"], 3)
        E_dis = round(grasp_config_dict["E_dis"], 5)
        E_pen = round(grasp_config_dict["E_pen"], 5)
        E_spen = round(grasp_config_dict["E_spen"], 5)
        E_joints = round(grasp_config_dict["E_joints"], 5)
        result = (
            f"Index {args.idx_to_visualize}  E_fc {E_fc}  E_dis {E_dis}  E_pen {E_pen}"
        )
        fig.add_annotation(text=result, x=0.5, y=0.1, xref="paper", yref="paper")
    fig.update_layout(scene_aspectmode="data")

    if args.save_to_html:
        output_folder = "../html_outputs"
        os.makedirs(output_folder, exist_ok=True)
        output_filepath = os.path.join(
            output_folder,
            f"result_{args.object_code_and_scale_str}-{args.idx_to_visualize}.html",
        )
        print(f"Saving to {output_filepath}")
        fig.write_html(output_filepath)
    else:
        print("Showing figure...")
        fig.show()


if __name__ == "__main__":
    args = VisualizeGraspConfigDictArgumentParser().parse_args()
    main(args)
