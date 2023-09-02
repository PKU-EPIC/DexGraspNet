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
from typing import Dict, Any, List, Tuple, Optional

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


class VisualizeConfigDictArgumentParser(Tap):
    hand_model_type: HandModelType = HandModelType.ALLEGRO_HAND
    input_config_dicts_path: pathlib.Path = pathlib.Path(
        "../data/config_dicts"
    )  # SHOULD be able to hand most types of config dicts
    meshdata_root_path: pathlib.Path = pathlib.Path("../data/meshdata")
    object_code_and_scale_str: str = "sem-Ipod-4b6c6248d5c01b3e4eee8d1cb32988b_0_10"
    idx_to_visualize: int = 0
    save_to_html: bool = False
    device: str = "cpu"


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
    config_dict: Dict[str, Any],
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
    )
    assert grasp_orientations.shape == (hand_model.num_fingers, 3, 3)
    fingertip_targets = computer_fingertip_targets(
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


def main(args: VisualizeConfigDictArgumentParser):
    object_code, object_scale = parse_object_code_and_scale(
        args.object_code_and_scale_str
    )

    # load results
    config_dicts: List[Dict[str, Any]] = np.load(
        args.input_config_dicts_path / f"{args.object_code_and_scale_str}.npy",
        allow_pickle=True,
    )
    config_dict = config_dicts[args.idx_to_visualize]

    # hand model: be careful with this, as it is stateful
    hand_model = HandModel(hand_model_type=args.hand_model_type, device=args.device)

    # object model
    object_model = ObjectModel(
        meshdata_root_path=str(args.meshdata_root_path),
        batch_size_each=1,
        scale=object_scale,
        num_samples=2000,
        device=args.device,
    )
    object_model.initialize(object_code)
    object_plotly = object_model.get_plotly_data(
        i=0, color="lightgreen", opacity=0.5, with_surface_points=True
    )

    # hand pose
    joint_names = handmodeltype_to_joint_names[args.hand_model_type]
    hand_pose = qpos_to_pose(
        qpos=config_dict["qpos"],
        joint_names=joint_names,
        unsqueeze_batch_dim=True,
    ).to(args.device)

    # hand pose start
    if "qpos_start" in config_dict:
        hand_pose_start = qpos_to_pose(
            qpos=config_dict["qpos_start"],
            joint_names=joint_names,
            unsqueeze_batch_dim=True,
        ).to(args.device)
    else:
        hand_pose_start = None

    # hand config dict
    print("Plotting hand config dict part")
    hand_config_dict_plotly_data_list = get_hand_config_dict_plotly_data_list(
        hand_model=hand_model,
        hand_pose=hand_pose,
        hand_pose_start=hand_pose_start,
    )
    print("Done")

    # grasp config dict
    print("Plotting grasp config dict part")
    grasp_config_dict_plotly_data_list = get_grasp_config_dict_plotly_data_list(
        hand_model=hand_model,
        hand_pose=hand_pose,
        config_dict=config_dict,
        device=args.device,
    )
    print("Done")

    # Create fig
    print("Creating fig")
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
            key: round(value, 3)
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
        passed_eval = config_dict["passed_eval"]
        passed_eval_str = f"Passed eval: {passed_eval}"
        fig.add_annotation(
            text=passed_eval_str, x=0.5, y=0.05, xref="paper", yref="paper"
        )

    # score
    if "score" in config_dict:
        score = config_dict["score"]
        score_str = f"Score: {score}"
        fig.add_annotation(text=score_str, x=0.5, y=0.0, xref="paper", yref="paper")

    fig.update_layout()
    fig.update_layout(
        title=f"{args.object_code_and_scale_str} {args.idx_to_visualize}",
        scene=dict(
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z"),
            aspectmode="data",
        ),
    )

    # Output
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
    args = VisualizeConfigDictArgumentParser().parse_args()
    main(args)
