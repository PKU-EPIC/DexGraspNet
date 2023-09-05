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
from utils.qpos_pose_conversion import (
    qpos_to_pose,
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
    hand_pose_array: List[torch.Tensor],
    object_code: str,
    object_scale: float,
) -> torch.Tensor:
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    batch_size = len(hand_pose_array)

    # hand model
    hand_model = HandModel(hand_model_type=args.hand_model_type, device=device)
    hand_model.set_parameters(torch.stack(hand_pose_array).to(device))

    # object model
    object_model = ObjectModel(
        meshdata_root_path=str(args.meshdata_root_path),
        batch_size_each=batch_size,
        scale=object_scale,
        num_samples=0,
        device=device,
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
    input_hand_config_dict_paths: List[pathlib.Path],
    output_grasp_config_dicts_path: pathlib.Path,
) -> None:
    joint_names = handmodeltype_to_joint_names[args.hand_model_type]
    print(f"len(input_hand_config_dict_paths): {len(input_hand_config_dict_paths)}")
    print(f"First 10: {[path for path in input_hand_config_dict_paths[:10]]}")
    random.Random(args.seed).shuffle(input_hand_config_dict_paths)

    set_seed(42)  # Want this fixed so deterministic computation
    pbar = tqdm(
        input_hand_config_dict_paths,
        desc="Generating grasp_config_dicts",
        dynamic_ncols=True,
    )
    for hand_config_dict_path in pbar:
        object_code_and_scale_str = hand_config_dict_path.stem
        object_code, object_scale = parse_object_code_and_scale(
            object_code_and_scale_str
        )

        # Read in data
        hand_config_dicts: List[Dict[str, Any]] = list(
            np.load(hand_config_dict_path, allow_pickle=True)
        )

        batch_size = len(hand_config_dicts)
        hand_pose_array = []
        for i in range(batch_size):
            qpos = hand_config_dicts[i]["qpos"]
            hand_pose_array.append(
                qpos_to_pose(
                    qpos=qpos, joint_names=joint_names, unsqueeze_batch_dim=False
                )
            )

        # Compute grasp_orientations
        grasp_orientations = compute_grasp_orientations(
            args=args,
            hand_pose_array=hand_pose_array,
            object_code=object_code,
            object_scale=object_scale,
        )

        # Save grasp_config_dicts
        grasp_config_dicts = []
        for i in range(batch_size):
            grasp_config_dicts.append(
                {
                    **hand_config_dicts[i],
                    "grasp_orientations": grasp_orientations[i].tolist(),
                }
            )

        output_grasp_config_dicts_path.mkdir(parents=True, exist_ok=True)
        np.save(
            output_grasp_config_dicts_path / f"{object_code_and_scale_str}.npy",
            grasp_config_dicts,
            allow_pickle=True,
        )


def main(args: GenerateGraspConfigDictsArgumentParser):
    print("=" * 80)
    print(f"args = {args}")
    print("=" * 80 + "\n")

    os.environ.pop("CUDA_VISIBLE_DEVICES")

    input_hand_config_dict_paths = [
        path for path in list(args.input_hand_config_dicts_path.glob("*.npy"))
    ]
    generate_grasp_config_dicts(
        args=args,
        input_hand_config_dict_paths=input_hand_config_dict_paths,
        output_grasp_config_dicts_path=args.output_grasp_config_dicts_path,
    )

    for mid_optimization_step in args.mid_optimization_steps:
        mid_optimization_input_hand_config_dict_paths = [
            hand_config_dict_path.parent
            / "mid_optimization"
            / f"{mid_optimization_step}"
            for hand_config_dict_path in input_hand_config_dict_paths
        ]
        mid_optimization_output_grasp_config_dicts_path = (
            args.output_grasp_config_dicts_path
            / "mid_optimization"
            / f"{mid_optimization_step}"
        )
        generate_grasp_config_dicts(
            args=args,
            input_hand_config_dict_paths=mid_optimization_input_hand_config_dict_paths,
            output_grasp_config_dicts_path=mid_optimization_output_grasp_config_dicts_path,
        )


if __name__ == "__main__":
    args = GenerateGraspConfigDictsArgumentParser().parse_args()
    main(args)
