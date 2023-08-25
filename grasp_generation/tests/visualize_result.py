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
import torch
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, List

from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.hand_model_type import handmodeltype_to_joint_names, HandModelType
from utils.qpos_pose_conversion import qpos_to_pose


class VisualizeResultArgumentParser(Tap):
    hand_model_type: HandModelType = HandModelType.ALLEGRO_HAND
    object_code: str = "sem-Xbox360-d0dff348985d4f8e65ca1b579a4b8d2"
    idx_to_visualize: int = 0
    result_path: str = "../data/dataset"
    save_to_html: bool = False


def get_contact_candidates_plot(link_name_to_contact_candidates: Dict[str, Any], plot_name: str, color: str) -> List[go.Scatter3d]:
    all_contact_candidates = np.concatenate(
        [
            contact_candidates
            for _, contact_candidates in link_name_to_contact_candidates.items()
        ],
        axis=0,
    )
    num_points = all_contact_candidates.shape[0]
    assert all_contact_candidates.shape == (num_points, 3)
    plotly_list = [
        go.Scatter3d(
            x=all_contact_candidates[:, 0],
            y=all_contact_candidates[:, 1],
            z=all_contact_candidates[:, 2],
            mode="markers",
            marker=dict(size=2, color=color),
            name=plot_name,
        )
    ]
    return plotly_list


def main(args: VisualizeResultArgumentParser):
    device = "cpu"

    joint_names = handmodeltype_to_joint_names[args.hand_model_type]

    # load results
    data_dict = np.load(
        os.path.join(args.result_path, args.object_code + ".npy"), allow_pickle=True
    )[args.idx_to_visualize]
    hand_pose = qpos_to_pose(
        qpos=data_dict["qpos"], joint_names=joint_names, unsqueeze_batch_dim=True
    ).to(device)
    hand_pose_st = (
        qpos_to_pose(
            qpos=data_dict["qpos_st"], joint_names=joint_names, unsqueeze_batch_dim=True
        ).to(device)
        if "qpos_st" in data_dict
        else None
    )

    if "link_name_to_contact_candidates" in data_dict:
        contact_candidates_plotly = get_contact_candidates_plot(link_name_to_contact_candidates=data_dict["link_name_to_contact_candidates"], plot_name="contact_candidates", color="red")
    else:
        contact_candidates_plotly = []
    if "link_name_to_target_contact_candidates" in data_dict:
        target_contact_candidates_plotly = get_contact_candidates_plot(link_name_to_contact_candidates=data_dict["link_name_to_target_contact_candidates"], plot_name="target_contact_candidates", color="green")
    else:
        target_contact_candidates_plotly = []

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
    object_model.initialize(args.object_code)

    # visualize
    if hand_pose_st is not None:
        hand_model.set_parameters(hand_pose_st)
        hand_st_plotly = hand_model.get_plotly_data(
            i=0, opacity=0.5, color="lightblue", with_contact_points=False
        )
    else:
        hand_st_plotly = []

    hand_model.set_parameters(hand_pose)
    hand_en_plotly = hand_model.get_plotly_data(
        i=0, opacity=1, color="lightblue", with_contact_points=False
    )
    object_plotly = object_model.get_plotly_data(i=0, color="lightgreen", opacity=1)
    fig = go.Figure(hand_st_plotly + hand_en_plotly + object_plotly + contact_candidates_plotly + target_contact_candidates_plotly)
    if "energy" in data_dict:
        energy = data_dict["energy"]
        E_fc = round(data_dict["E_fc"], 3)
        E_dis = round(data_dict["E_dis"], 5)
        E_pen = round(data_dict["E_pen"], 5)
        E_spen = round(data_dict["E_spen"], 5)
        E_joints = round(data_dict["E_joints"], 5)
        result = f"Index {args.idx_to_visualize}  E_fc {E_fc}  E_dis {E_dis}  E_pen {E_pen}"
        fig.add_annotation(text=result, x=0.5, y=0.1, xref="paper", yref="paper")
    fig.update_layout(scene_aspectmode="data")

    if args.save_to_html:
        output_folder = "../html_outputs"
        os.makedirs(output_folder, exist_ok=True)
        output_filepath = os.path.join(
            output_folder,
            f"result_{args.object_code}-{args.idx_to_visualize}.html",
        )
        print(f"Saving to {output_filepath}")
        fig.write_html(output_filepath)
    else:
        print("Showing figure...")
        fig.show()

if __name__ == "__main__":
    args = VisualizeResultArgumentParser().parse_args()
    main(args)
