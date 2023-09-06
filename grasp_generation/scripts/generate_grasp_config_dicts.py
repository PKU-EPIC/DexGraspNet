"""
Last modified date: 2023.08.23
Author: Tyler Lum
Description: Read in hand_config_dicts and generate grasp_config_dicts by computing grasp directions
"""

import os
import sys

sys.path.append(os.path.realpath("."))

import pathlib
from tap import Tap
import torch
import numpy as np
import random
from tqdm import tqdm
from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.hand_model_type import (
    HandModelType,
    handmodeltype_to_joint_names,
)
from utils.pose_conversion import (
    hand_config_to_pose,
)
from typing import List, Dict, Any, Tuple
from utils.seed import set_seed
from utils.joint_angle_targets import (
    compute_grasp_orientations as compute_grasp_orientations_external,
)
from utils.energy import _cal_hand_object_penetration
from utils.parse_object_code_and_scale import (
    parse_object_code_and_scale,
)


class GenerateGraspConfigDictsArgumentParser(Tap):
    hand_model_type: HandModelType = HandModelType.ALLEGRO_HAND
    gpu: int = 0
    meshdata_root_path: pathlib.Path = pathlib.Path("../data/meshdata")
    input_hand_config_dicts_path: pathlib.Path = pathlib.Path(
        "../data/hand_config_dicts"
    )
    output_grasp_config_dicts_path: pathlib.Path = pathlib.Path(
        "../data/grasp_config_dicts"
    )
    mid_optimization_steps: List[int] = []
    seed: int = 1


def compute_grasp_orientations(
    args: GenerateGraspConfigDictsArgumentParser,
    hand_pose: torch.Tensor,
    object_code: str,
    object_scale: float,
) -> torch.Tensor:
    batch_size = hand_pose.shape[0]

    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else "cpu"
    hand_pose = hand_pose.to(device)

    # hand model
    hand_model = HandModel(
        hand_model_type=args.hand_model_type, device=hand_pose.device
    )
    hand_model.set_parameters(hand_pose)

    # object model
    object_model = ObjectModel(
        meshdata_root_path=str(args.meshdata_root_path),
        batch_size_each=batch_size,
        scale=object_scale,
        num_samples=0,
        device=hand_pose.device,
    )
    object_model.initialize(object_code)
    grasp_orientations = compute_grasp_orientations_external(
        joint_angles_start=hand_model.hand_pose[:, 9:],
        hand_model=hand_model,
        object_model=object_model,
    )
    assert grasp_orientations.shape == (batch_size, hand_model.num_fingers, 3, 3)
    return grasp_orientations


def generate_grasp_config_dicts(
    args: GenerateGraspConfigDictsArgumentParser,
    input_hand_config_dicts_path: pathlib.Path,
    output_grasp_config_dicts_path: pathlib.Path,
) -> None:
    hand_config_dict_filepaths = [
        path for path in list(input_hand_config_dicts_path.glob("*.npy"))
    ]
    print(f"len(input_hand_config_dict_filepaths): {len(hand_config_dict_filepaths)}")
    print(f"First 10: {[path for path in hand_config_dict_filepaths[:10]]}")
    random.Random(args.seed).shuffle(hand_config_dict_filepaths)

    set_seed(42)  # Want this fixed so deterministic computation
    pbar = tqdm(
        hand_config_dict_filepaths,
        desc="Generating grasp_config_dicts",
        dynamic_ncols=True,
    )
    for hand_config_dict_filepath in pbar:
        object_code_and_scale_str = hand_config_dict_filepath.stem
        object_code, object_scale = parse_object_code_and_scale(
            object_code_and_scale_str
        )

        # Read in data
        hand_config_dict: Dict[str, np.ndarray] = np.load(
            hand_config_dict_filepath, allow_pickle=True
        ).item()

        hand_pose = hand_config_to_pose(
            hand_config_dict["trans"],
            hand_config_dict["rot"],
            hand_config_dict["joint_angles"],
        )

        # Compute grasp_orientations
        grasp_orientations = compute_grasp_orientations(
            args=args,
            hand_pose=hand_pose,
            object_code=object_code,
            object_scale=object_scale,
        )  # shape = (batch_size, num_fingers, 3, 3)

        grasp_config_dict = hand_config_dict.copy()
        grasp_config_dict["grasp_orientations"] = (
            grasp_orientations.detach().cpu().numpy()
        )

        # Save grasp_config_dict

        output_grasp_config_dicts_path.mkdir(parents=True, exist_ok=True)
        np.save(
            output_grasp_config_dicts_path / f"{object_code_and_scale_str}.npy",
            grasp_config_dict,
            allow_pickle=True,
        )


def main(args: GenerateGraspConfigDictsArgumentParser):
    print("=" * 80)
    print(f"args = {args}")
    print("=" * 80 + "\n")

    os.environ.pop("CUDA_VISIBLE_DEVICES")

    generate_grasp_config_dicts(
        args=args,
        input_hand_config_dicts_path=args.input_hand_config_dicts_path,
        output_grasp_config_dicts_path=args.output_grasp_config_dicts_path,
    )

    for mid_optimization_step in args.mid_optimization_steps:
        print("!" * 80)
        print(f"Running mid_optimization_step: {mid_optimization_step}")
        print("!" * 80 + "\n")
        mid_optimization_input_hand_config_dicts_path = (
            args.input_hand_config_dicts_path
            / "mid_optimization"
            / f"{mid_optimization_step}"
        )
        mid_optimization_output_grasp_config_dicts_path = (
            args.output_grasp_config_dicts_path
            / "mid_optimization"
            / f"{mid_optimization_step}"
        )
        generate_grasp_config_dicts(
            args=args,
            input_hand_config_dicts_path=mid_optimization_input_hand_config_dicts_path,
            output_grasp_config_dicts_path=mid_optimization_output_grasp_config_dicts_path,
        )


if __name__ == "__main__":
    args = GenerateGraspConfigDictsArgumentParser().parse_args()
    main(args)
