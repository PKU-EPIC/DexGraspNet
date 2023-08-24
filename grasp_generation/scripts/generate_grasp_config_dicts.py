"""
Last modified date: 2023.08.23
Author: Tyler Lum
Description: Read in hand_config_dicts and generate grasp_config_dicts by computing grasp directions
"""

import os
import sys

sys.path.append(os.path.realpath("."))

from utils.isaac_validator import IsaacValidator, ValidationType
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
    qpos_to_translation_quaternion_jointangles,
    pose_to_qpos,
)
from typing import List, Optional, Dict, Any
import math
from utils.seed import set_seed
from utils.joint_angle_targets import (
    compute_optimized_joint_angle_targets,
    OptimizationMethod,
)
from utils.energy import _cal_hand_object_penetration


class GenerateGraspConfigDictsArgumentParser(Tap):
    hand_model_type: HandModelType = HandModelType.ALLEGRO_HAND
    optimization_method: OptimizationMethod = (
        OptimizationMethod.DESIRED_DIST_TOWARDS_OBJECT_SURFACE_MULTIPLE_STEPS
    )
    gpu: int = 0
    meshdata_root_path: pathlib.Path = pathlib.Path("../data/meshdata")
    input_hand_config_dicts_path: pathlib.Path = pathlib.Path(
        "../data/hand_config_dicts"
    )
    output_grasp_config_dicts_path: pathlib.Path = pathlib.Path("../data/dataset")
    seed: int = 1


def compute_joint_angle_targets(
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
        num_samples=0,
        device=device,
    )
    object_model.initialize(object_code)
    object_model.object_scale_tensor = (
        torch.tensor([object_scale] * batch_size).reshape(1, batch_size).to(device)
    )  # 1 because 1 object code

    # Optimization
    (
        optimized_joint_angle_targets,
        losses,
        debug_infos,
    ) = compute_optimized_joint_angle_targets(
        optimization_method=args.optimization_method,
        hand_model=hand_model,
        object_model=object_model,
        device=device,
    )

    return optimized_joint_angle_targets


def compute_link_name_to_all_contact_candidates(
    args: GenerateGraspConfigDictsArgumentParser,
    hand_pose_array: List[torch.Tensor],
    joint_angles: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # hand model
    hand_model = HandModel(hand_model_type=args.hand_model_type, device=device)
    hand_model.set_parameters(torch.stack(hand_pose_array).to(device))
    batch_size = len(hand_pose_array)

    current_status = hand_model.chain.forward_kinematics(joint_angles.to(device))
    link_name_to_contact_candidates = {}
    for i, link_name in enumerate(hand_model.mesh):
        contact_candidates = hand_model.mesh[link_name]["contact_candidates"]
        if len(contact_candidates) == 0:
            continue

        contact_candidates = (
            current_status[link_name]
            .transform_points(contact_candidates)
            .expand(batch_size, -1, 3)
        )
        contact_candidates = contact_candidates @ hand_model.global_rotation.transpose(
            1, 2
        ) + hand_model.global_translation.unsqueeze(1)

        link_name_to_contact_candidates[link_name] = contact_candidates
    return link_name_to_contact_candidates


def split_object_code_and_scale(object_code_and_scale: str) -> (str, float):
    keyword = "_0_"
    idx = object_code_and_scale.rfind(keyword)
    object_code = object_code_and_scale[:idx]
    object_scale = float(object_code_and_scale[idx + len(keyword) :].replace("_", "."))
    return object_code, object_scale


def main(args: GenerateGraspConfigDictsArgumentParser):
    joint_names = handmodeltype_to_joint_names[args.hand_model_type]
    os.environ.pop("CUDA_VISIBLE_DEVICES")

    hand_config_dict_paths = [path for path in args.input_hand_config_dicts_path.iterdir()]
    print(f"len(hand_config_dict_paths): {len(hand_config_dict_paths)}")
    print(f"First 10: {[path.name for path in hand_config_dict_paths[:10]]}")
    random.Random(args.seed).shuffle(hand_config_dict_paths)

    set_seed(42)  # Want this fixed so deterministic computation
    pbar = tqdm(
        hand_config_dict_paths,
        desc="Generating grasp_config_dicts",
        dynamic_ncols=True,
    )
    for hand_config_dict_path in pbar:
        object_code_and_scale = hand_config_dict_path.name
        object_code, object_scale = split_object_code_and_scale(object_code_and_scale)

        # Read in data
        data_dicts: List[Dict[str, Any]] = np.load(
            hand_config_dict_path, allow_pickle=True
        )
        batch_size = len(data_dicts)
        joint_angles_array = []
        hand_pose_array = []
        for i in range(batch_size):
            qpos = data_dicts[i]["qpos"]
            (
                _,
                _,
                joint_angles,
            ) = qpos_to_translation_quaternion_jointangles(
                qpos=qpos, joint_names=joint_names
            )
            joint_angles_array.append(joint_angles)
            hand_pose_array.append(
                qpos_to_pose(
                    qpos=qpos, joint_names=joint_names, unsqueeze_batch_dim=False
                )
            )

        # Compute joint angle targets
        joint_angle_targets_array = (
            compute_joint_angle_targets(
                args=args,
                hand_pose_array=hand_pose_array,
                object_code=object_code,
                object_scale=object_scale,
            )
            .detach()
            .cpu()
            .numpy()
        )
        success_data_dicts = []
        link_name_to_all_contact_candidates = (
            compute_link_name_to_all_contact_candidates(
                args=args,
                hand_pose_array=hand_pose_array,
                joint_angles=torch.from_numpy(np.stack(joint_angles_array, axis=0)),
            )
        )
        link_name_to_all_target_contact_candidates = (
            compute_link_name_to_all_contact_candidates(
                args=args,
                hand_pose_array=hand_pose_array,
                joint_angles=torch.from_numpy(
                    np.stack(joint_angle_targets_array, axis=0)
                ),
            )
        )
        for i in range(batch_size):
            success_data_dicts.append(
                {
                    "qpos": pose_to_qpos(
                        hand_pose=hand_pose_array[i], joint_names=joint_names
                    ),
                    "scale": object_scale,
                    "valid": valid[i],
                    "link_name_to_contact_candidates": {
                        link_name: all_contact_candidates[i].cpu().numpy()
                        for link_name, all_contact_candidates in link_name_to_all_contact_candidates.items()
                    },
                    "link_name_to_target_contact_candidates": {
                        link_name: all_target_contact_candidates[i].cpu().numpy()
                        for link_name, all_target_contact_candidates in link_name_to_all_target_contact_candidates.items()
                    },
                    "joint_angles": joint_angles_array[i],
                }
            )

        os.makedirs(args.output_grasp_config_dicts_path, exist_ok=True)
        np.save(
            os.path.join(args.output_grasp_config_dicts_path, object_code + ".npy"),
            success_data_dicts,
            allow_pickle=True,
        )


if __name__ == "__main__":
    args = GenerateGraspConfigDictsArgumentParser().parse_args()
    main(args)
