"""
Last modified date: 2023.07.01
Author: Tyler Lum
Description: visualize hand model grasp optimization
"""

import os
import sys
from tqdm import tqdm

sys.path.append(os.path.realpath("."))

import torch
import numpy as np
import plotly.graph_objects as go
from typing import List, Tuple, Optional
from tap import Tap
from visualize_optimization_helper import (
    create_figure_with_buttons_and_slider,
)
from utils.qpos_pose_conversion import qpos_to_pose
from utils.hand_model_type import (
    HandModelType,
    handmodeltype_to_joint_names,
)
from utils.hand_model import HandModel
from utils.object_model import ObjectModel


class VisualizeOptimizationFromFolderArgumentParser(Tap):
    """Expects a folder with the following structure:
    - input_folder
        - 0
            - object_code.npy
        - x
            - object_code.npy
        - 2x
            - object_code.npy
        - 3x
            - object_code.npy
        ...
    """

    input_folder: str = "../data/2023-07-01_debug_graspdata_extra/"
    object_code: str = "core-pistol-ad72857d0fd2ad2d44a52d2e669c8daa"
    frame_duration: int = 200
    transition_duration: int = 100
    save_to_html: bool = False


def get_visualization_freq_from_folder(input_folder: str) -> int:
    folders = os.listdir(input_folder)
    for folder in folders:
        assert folder.isdigit(), f"Folder {folder} is not a number"

    visualization_iters = sorted([int(f) for f in folders])
    visualization_freq = visualization_iters[1] - visualization_iters[0]
    assert (
        visualization_iters[-1] - visualization_iters[0]
        == (len(visualization_iters) - 1) * visualization_freq
    ), f"Visualization iters are not evenly spaced: {visualization_iters}"
    return visualization_freq


def create_grasp_fig(
    hand_model: HandModel, object_model: ObjectModel, idx_to_visualize: int
) -> go.Figure:
    fig = go.Figure(
        layout=go.Layout(
            scene=dict(
                xaxis=dict(title="X"),
                yaxis=dict(title="Y"),
                zaxis=dict(title="Z"),
                aspectmode="data",
            ),
            showlegend=True,
            title="Grasp Visualization",
        )
    )
    plots = [
        *hand_model.get_plotly_data(
            i=idx_to_visualize,
            opacity=1.0,
            with_contact_points=False,  # No contact points after optimization
            with_contact_candidates=True,
        ),
        *object_model.get_plotly_data(i=idx_to_visualize, opacity=0.5),
    ]
    for plot in plots:
        fig.add_trace(plot)
    return fig


def get_hand_and_object_model_from_data_dict(
    data_dict: np.ndarray, object_code: str
) -> Tuple[HandModel, ObjectModel]:
    HAND_MODEL_TYPE = HandModelType.ALLEGRO_HAND
    MESH_PATH = "../data/meshdata"

    joint_names = handmodeltype_to_joint_names[HAND_MODEL_TYPE]
    batch_size = data_dict.shape[0]
    scale_array = []
    hand_pose_array = []
    for i in range(batch_size):
        qpos = data_dict[i]["qpos"]
        hand_pose_array.append(
            qpos_to_pose(qpos=qpos, joint_names=joint_names, unsqueeze_batch_dim=False)
        )
        scale = data_dict[i]["scale"]
        scale_array.append(scale)

    GPU = 0
    device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
    batch_size = len(hand_pose_array)

    # hand model
    hand_model = HandModel(hand_model_type=HAND_MODEL_TYPE, device=device)
    hand_model.set_parameters(torch.stack(hand_pose_array).to(device))

    # object model
    object_model = ObjectModel(
        data_root_path=MESH_PATH,
        batch_size_each=batch_size,
        num_samples=0,
        device=device,
    )
    object_model.initialize(object_code)
    object_model.object_scale_tensor = (
        torch.tensor(scale_array).reshape(1, -1).to(device)
    )  # 1 because 1 object code
    return hand_model, object_model


def get_grasps_from_folder(input_folder: str, object_code: str) -> List[go.Figure]:
    figs = []
    sorted_mid_folders = sorted(os.listdir(input_folder), key=int)
    for mid_folder in tqdm(sorted_mid_folders, desc="Going through folders..."):
        filepath = os.path.join(input_folder, mid_folder, f"{object_code}.npy")
        data_dict = np.load(os.path.join(filepath), allow_pickle=True)
        hand_model, object_model = get_hand_and_object_model_from_data_dict(
            data_dict=data_dict, object_code=object_code
        )
        fig = create_grasp_fig(
            hand_model=hand_model,
            object_model=object_model,
            idx_to_visualize=0,
        )
        figs.append(fig)
    return figs


def get_figs_from_folder(
    input_folder: str, object_code: str
) -> Tuple[List[go.Figure], int]:
    grasps = get_grasps_from_folder(input_folder=input_folder, object_code=object_code)
    visualization_freq = get_visualization_freq_from_folder(input_folder=input_folder)
    return grasps, visualization_freq


def main(args: VisualizeOptimizationFromFolderArgumentParser):
    input_figs, visualization_freq = get_figs_from_folder(
        input_folder=args.input_folder, object_code=args.object_code
    )

    print("Making figure with buttons and slider...")
    new_fig = create_figure_with_buttons_and_slider(
        input_figs=input_figs,
        visualization_freq=visualization_freq,
        frame_duration=args.frame_duration,
        transition_duration=args.transition_duration,
    )
    print("Done making figure with buttons and slider")

    if args.save_to_html:
        output_folder = "../html_outputs"
        os.makedirs(output_folder, exist_ok=True)
        output_filepath = os.path.join(output_folder, f"{args.object_code}.html")
        print(f"Saving to {output_filepath}")
        new_fig.write_html(output_filepath)
    else:
        new_fig.show()


if __name__ == "__main__":
    main(VisualizeOptimizationFromFolderArgumentParser().parse_args())
