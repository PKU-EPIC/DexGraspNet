"""
Last modified date: 2023.02.23
Author: Jialiang Zhang
Description: visualize grasp result using plotly.graph_objects
"""

import os
import sys

# os.chdir(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tap import Tap
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, List, Tuple

from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.hand_model_type import handmodeltype_to_joint_names, HandModelType
from utils.qpos_pose_conversion import qpos_to_pose
import pathlib
from utils.parse_object_code_and_scale import (
    parse_object_code_and_scale,
)


class VisualizeHandConfigDictArgumentParser(Tap):
    hand_model_type: HandModelType = HandModelType.ALLEGRO_HAND
    input_hand_config_dicts_path: pathlib.Path = pathlib.Path(
        "../data/hand_config_dicts"
    )
    object_code_and_scale_str: str = "sem-Ipod-4b6c6248d5c01b3e4eee8d1cb32988b_0_10"
    idx_to_visualize: int = 0
    save_to_html: bool = False


def main(args: VisualizeHandConfigDictArgumentParser):
    device = "cpu"

    joint_names = handmodeltype_to_joint_names[args.hand_model_type]
    object_code, object_scale = parse_object_code_and_scale(
        args.object_code_and_scale_str
    )

    # load results
    data_dicts: List[Dict[str, Any]] = np.load(
        args.input_hand_config_dicts_path / f"{args.object_code_and_scale_str}.npy",
        allow_pickle=True,
    )
    data_dict = data_dicts[args.idx_to_visualize]
    hand_pose = qpos_to_pose(
        qpos=data_dict["qpos"], joint_names=joint_names, unsqueeze_batch_dim=True
    ).to(device)
    hand_pose_start = (
        qpos_to_pose(
            qpos=data_dict["qpos_start"],
            joint_names=joint_names,
            unsqueeze_batch_dim=True,
        ).to(device)
        if "qpos_start" in data_dict
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
    hand_en_plotly = hand_model.get_plotly_data(
        i=0,
        opacity=1,
        color="lightblue",
        with_contact_points=False,
        with_contact_candidates=True,
        with_surface_points=True,
        with_penetration_keypoints=True,
    )
    object_plotly = object_model.get_plotly_data(
        i=0, color="lightgreen", opacity=1, with_surface_points=True
    )
    fig = go.Figure(hand_start_plotly + hand_en_plotly + object_plotly)
    if "energy" in data_dict:
        energy = data_dict["energy"]
        E_fc = round(data_dict["E_fc"], 3)
        E_dis = round(data_dict["E_dis"], 5)
        E_pen = round(data_dict["E_pen"], 5)
        E_spen = round(data_dict["E_spen"], 5)
        E_joints = round(data_dict["E_joints"], 5)
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
    args = VisualizeHandConfigDictArgumentParser().parse_args()
    main(args)
