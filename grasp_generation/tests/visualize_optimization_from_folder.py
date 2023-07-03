"""
Last modified date: 2023.07.01
Author: Tyler Lum
Description: visualize hand model grasp optimization
"""

import os
import sys
from tqdm import tqdm

sys.path.append(os.path.realpath("."))

import plotly.graph_objects as go
from typing import List, Tuple
from tap import Tap
import numpy as np
from visualize_optimization_helper import (
    create_figure_with_buttons_and_slider,
    get_hand_and_object_model_from_data_dict,
    create_grasp_fig,
)


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

    input_folder: str = "../data/2023-07-01_dryrun_graspdata_mid_optimization/"
    object_code: str = "sem-ToyFigure-47204f6aaa776c7bf8208b6313b1ffa0"
    idx_to_visualize: int = 0
    frame_duration: int = 200
    transition_duration: int = 100
    save_to_html: bool = False


def get_grasps_from_folder(
    input_folder: str, object_code: str, idx_to_visualize: int
) -> List[go.Figure]:
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
            idx_to_visualize=idx_to_visualize,
        )
        figs.append(fig)
    return figs


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


def main(args: VisualizeOptimizationFromFolderArgumentParser):
    print(f"args = {args}")
    input_figs = get_grasps_from_folder(
        input_folder=args.input_folder,
        object_code=args.object_code,
        idx_to_visualize=args.idx_to_visualize,
    )

    visualization_freq = get_visualization_freq_from_folder(
        input_folder=args.input_folder
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
        output_filepath = os.path.join(
            output_folder, f"optimization_{args.object_code}_{args.idx_to_visualize}.html"
        )
        print(f"Saving to {output_filepath}")
        new_fig.write_html(output_filepath)
    else:
        print("Showing figure...")
        new_fig.show()


if __name__ == "__main__":
    main(VisualizeOptimizationFromFolderArgumentParser().parse_args())
