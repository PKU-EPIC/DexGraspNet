"""
Last modified date: 2023.08.19
Author: Tyler Lum
Description: eval grasps on Isaac simulator
"""

import os
import sys

sys.path.append(os.path.realpath("."))

from utils.isaac_validator import IsaacValidator, ValidationType
from tap import Tap
import torch
import numpy as np
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
from typing import List, Optional, Dict
import math
from utils.seed import set_seed
from utils.joint_angle_targets import (
    compute_optimized_joint_angle_targets_given_directions,
    OptimizationMethod,
)


class EvalGraspArgumentParser(Tap):
    hand_model_type: HandModelType = HandModelType.ALLEGRO_HAND
    validation_type: ValidationType = ValidationType.NO_GRAVITY_SHAKING
    gpu: int = 0
    val_batch: int = 500
    mesh_path: str = "../data/meshdata"
    grasp_path: str = "../data/graspdata"
    result_path: str = "../data/dataset"
    object_code: str = "sem-Xbox360-d0dff348985d4f8e65ca1b579a4b8d2"
    # if debug_index is received, then the debug mode is on
    debug_index: Optional[int] = None
    start_with_step_mode: bool = False
    penetration_threshold: Optional[float] = None


def compute_joint_angle_targets(
    args: EvalGraspArgumentParser,
    hand_pose_array: List[torch.Tensor],
    grasp_dirs_array: List[torch.Tensor],
) -> torch.Tensor:
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    grasp_dirs_array = torch.cat(grasp_dirs_array, dim=0)

    # hand model
    hand_model = HandModel(hand_model_type=args.hand_model_type, device=device)
    hand_model.set_parameters(torch.stack(hand_pose_array).to(device))

    # Optimization
    (
        optimized_joint_angle_targets,
        losses,
        debug_infos,
    ) = compute_optimized_joint_angle_targets_given_directions(
        hand_model=hand_model,
        grasp_dirs_array=grasp_dirs_array,
    )

    return optimized_joint_angle_targets


def main(args: EvalGraspArgumentParser):
    set_seed(42)
    joint_names = handmodeltype_to_joint_names[args.hand_model_type]
    os.environ.pop("CUDA_VISIBLE_DEVICES")

    if args.debug_index is not None:
        sim = IsaacValidator(
            hand_model_type=args.hand_model_type,
            gpu=args.gpu,
            validation_type=args.validation_type,
            mode="gui",
            start_with_step_mode=args.start_with_step_mode,
        )
    else:
        sim = IsaacValidator(
            hand_model_type=args.hand_model_type,
            gpu=args.gpu,
            validation_type=args.validation_type,
        )

    # Read in data
    # TODO: Figure out details of grasp_configs
    grasp_configs = AllegroGraspConfig.from_path(args.grasp_path)
    batch_size = grasp_configs.batch_size
    translation_array = []
    quaternion_array = []
    joint_angles_array = []
    scale_array = []
    hand_pose_array = []
    grasp_dirs_array = []
    for i in range(batch_size):
        qpos = grasp_configs.qpos_array[i]  # TODO make this conversion
        (
            translation,
            quaternion,
            joint_angles,
        ) = qpos_to_translation_quaternion_jointangles(
            qpos=qpos, joint_names=joint_names
        )
        translation_array.append(translation)
        quaternion_array.append(quaternion)
        joint_angles_array.append(joint_angles)
        hand_pose_array.append(
            qpos_to_pose(qpos=qpos, joint_names=joint_names, unsqueeze_batch_dim=False)
        )

        scale = grasp_configs.scale_array[i]
        scale_array.append(scale)

    # Compute joint angle targets
    joint_angle_targets_array = compute_joint_angle_targets(
        args=args,
        hand_pose_array=hand_pose_array,
        grasp_dirs_array=grasp_dirs_array,
    )

    # Debug with single grasp
    if args.debug_index is not None:
        sim.set_obj_asset(
            obj_root=os.path.join(args.mesh_path, args.object_code, "coacd"),
            obj_file="coacd.urdf",
        )
        index = args.debug_index
        sim.add_env_single_test_rotation(
            hand_quaternion=quaternion_array[index],
            hand_translation=translation_array[index],
            hand_qpos=joint_angles_array[index],
            obj_scale=scale_array[index],
            target_qpos=(
                joint_angle_targets_array[index]
                if joint_angle_targets_array is not None
                else None
            ),
        )
        successes = sim.run_sim()
        print(f"successes = {successes}")

    # Run validation on all grasps
    else:
        passed_simulation = np.zeros(batch_size, dtype=np.bool8)
        successes = []
        num_val_batches = math.ceil(batch_size / args.val_batch)
        for val_batch_idx in range(num_val_batches):
            start_offset = val_batch_idx * args.val_batch
            end_offset = min(start_offset + args.val_batch, batch_size)

            sim.set_obj_asset(
                obj_root=os.path.join(args.mesh_path, args.object_code, "coacd"),
                obj_file="coacd.urdf",
            )
            for index in range(start_offset, end_offset):
                sim.add_env_all_test_rotations(
                    hand_quaternion=quaternion_array[index],
                    hand_translation=translation_array[index],
                    hand_qpos=joint_angles_array[index],
                    obj_scale=scale_array[index],
                    target_qpos=(
                        joint_angle_targets_array[index]
                        if joint_angle_targets_array is not None
                        else None
                    ),
                )
            successes.extend([*sim.run_sim()])
            sim.reset_simulator()

        num_envs_per_grasp = len(sim.test_rotations)
        for i in range(batch_size):
            passed_simulation[i] = np.array(
                sum(successes[i * num_envs_per_grasp : (i + 1) * num_envs_per_grasp])
                == num_envs_per_grasp
            )

        # TODO: add penetration check E_pen
        passed_penetration_threshold = (
            0 < args.penetration_threshold
            if args.penetration_threshold is not None
            else np.ones(batch_size, dtype=np.bool8)
        )
        valid = passed_simulation * passed_penetration_threshold
        print("=" * 80)
        print(
            f"passed_penetration_threshold: {passed_penetration_threshold.sum().item()}/{batch_size}, "
            f"passed_simulation: {passed_simulation.sum().item()}/{batch_size}, "
            f"valid = passed_simulation * passed_penetration_threshold: {valid.sum().item()}/{batch_size}"
        )
        print("=" * 80)
        success_data_dicts = []
        for i in range(batch_size):
            success_data_dicts.append(
                {
                    "qpos": pose_to_qpos(
                        hand_pose=hand_pose_array[i], joint_names=joint_names
                    ),
                    "scale": scale_array[i],
                    "valid": valid[i],
                }
            )

        os.makedirs(args.result_path, exist_ok=True)
        np.save(
            os.path.join(args.result_path, args.object_code + ".npy"),
            success_data_dicts,
            allow_pickle=True,
        )
    sim.destroy()


if __name__ == "__main__":
    args = EvalGraspArgumentParser().parse_args()
    main(args)

