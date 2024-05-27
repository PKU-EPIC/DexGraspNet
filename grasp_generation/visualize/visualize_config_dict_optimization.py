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
from typing import List, Tuple, Dict, Any, Optional
from tap import Tap
import numpy as np
from visualize_optimization_helper import (
    create_figure_with_buttons_and_slider,
)
from visualize_config_dict_helper import create_config_dict_fig

import pathlib

import numpy as np
from utils.pose_conversion import hand_config_to_pose
from utils.hand_model_type import (
    HandModelType,
)
from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.parse_object_code_and_scale import parse_object_code_and_scale


class VisualizeConfigDictOptimizationArgumentParser(Tap):
    """Expects a folder with the following structure:
    - <input_config_dicts_mid_optimization_path>
        - 0
            - <object_code_and_scale_str>.npy
        - x
            - <object_code_and_scale_str>.npy
        - 2x
            - <object_code_and_scale_str>.npy
        - 3x
            - <object_code_and_scale_str>.npy
        ...
    """

    input_config_dicts_mid_optimization_path: pathlib.Path = pathlib.Path(
        "../data/config_dicts_mid_optimization"
    )
    meshdata_root_path: pathlib.Path = pathlib.Path("../data/rotated_meshdata_stable")
    object_code_and_scale_str: str = (
        "mujoco-Olive_Kids_Butterfly_Garden_Pencil_Case_0_10"
    )
    idx_to_visualize: int = 0
    frame_duration: int = 200
    transition_duration: int = 100
    device: str = "cpu"
    save_to_html: bool = False
    skip_visualize_grasp_config_dict: bool = False
    uneven_diffs_ok: bool = False


def get_hand_model_from_config_dicts(
    config_dict: Dict[str, Any],
    device: str,
    idx_to_visualize: int,
    hand_model_type: HandModelType = HandModelType.ALLEGRO_HAND,
) -> HandModel:
    hand_pose = hand_config_to_pose(
        trans=config_dict["trans"][idx_to_visualize],
        rot=config_dict["rot"][idx_to_visualize],
        joint_angles=config_dict["joint_angles"][idx_to_visualize],
    ).to(device)
    hand_model = HandModel(hand_model_type=hand_model_type, device=device)
    hand_model.set_parameters(hand_pose)
    return hand_model


def get_object_model(
    meshdata_root_path: pathlib.Path,
    object_code_and_scale_str: str,
    device: str,
) -> Optional[ObjectModel]:
    try:
        object_code, object_scale = parse_object_code_and_scale(
            object_code_and_scale_str
        )

        object_model = ObjectModel(
            meshdata_root_path=str(meshdata_root_path),
            batch_size_each=1,
            device=device,
        )
        object_model.initialize(object_code, object_scale)
    except Exception as e:
        print("=" * 80)
        print(f"Exception: {e}")
        print("=" * 80)
        print(f"Skipping {object_code_and_scale_str} and continuing")
        object_model = None
    return object_model


def create_config_dict_figs_from_folder(
    input_config_dicts_mid_optimization_path: pathlib.Path,
    meshdata_root_path: pathlib.Path,
    object_code_and_scale_str: str,
    idx_to_visualize: int,
    device: str,
    skip_visualize_grasp_config_dict: bool,
    uneven_diffs_ok: bool,
) -> Tuple[List[go.Figure], int]:
    filename = f"{object_code_and_scale_str}.npy"

    sorted_mid_folders = sorted(
        [
            path.name
            for path in input_config_dicts_mid_optimization_path.iterdir()
            if path.is_dir() and path.name.isdigit() and (path / filename).exists()
        ],
        key=int,
    )

    # Check that the folders are evenly spaced
    diffs = []
    for first, second in zip(sorted_mid_folders[:-1], sorted_mid_folders[1:]):
        assert first.isdigit() and second.isdigit(), f"{first}, {second}"
        diffs.append(int(second) - int(first))
    if not uneven_diffs_ok:
        assert all(diff == diffs[0] for diff in diffs), f"diffs = {diffs}"

    figs = []
    for mid_folder in tqdm(sorted_mid_folders, desc="Going through folders..."):
        filepath = input_config_dicts_mid_optimization_path / mid_folder / filename
        assert filepath.exists(), f"{filepath} does not exist"

        # Read in data
        config_dict = np.load(filepath, allow_pickle=True).item()

        hand_model = get_hand_model_from_config_dicts(
            config_dict=config_dict, device=device, idx_to_visualize=idx_to_visualize
        )
        object_model = get_object_model(
            meshdata_root_path=meshdata_root_path,
            object_code_and_scale_str=object_code_and_scale_str,
            device=device,
        )

        # Create figure
        fig = create_config_dict_fig(
            config_dict=config_dict,
            hand_model=hand_model,
            object_model=object_model,
            skip_visualize_qpos_start=True,
            skip_visualize_grasp_config_dict=skip_visualize_grasp_config_dict,
            idx_to_visualize=idx_to_visualize,
            title=f"{object_code_and_scale_str} {idx_to_visualize}",
        )
        figs.append(fig)

    visualization_freq = diffs[0]
    return figs, visualization_freq


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


def main(args: VisualizeConfigDictOptimizationArgumentParser):
    print("=" * 80)
    print(f"args = {args}")
    print("=" * 80 + "\n")

    input_figs, visualization_freq = create_config_dict_figs_from_folder(
        input_config_dicts_mid_optimization_path=args.input_config_dicts_mid_optimization_path,
        meshdata_root_path=args.meshdata_root_path,
        object_code_and_scale_str=args.object_code_and_scale_str,
        idx_to_visualize=args.idx_to_visualize,
        device=args.device,
        skip_visualize_grasp_config_dict=args.skip_visualize_grasp_config_dict,
        uneven_diffs_ok=args.uneven_diffs_ok,
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
            output_folder,
            f"optimization_{args.object_code_and_scale_str}_{args.idx_to_visualize}.html",
        )
        print(f"Saving to {output_filepath}")
        new_fig.write_html(output_filepath)
    else:
        print("Showing figure...")
        new_fig.show()


if __name__ == "__main__":
    main(VisualizeConfigDictOptimizationArgumentParser().parse_args())
