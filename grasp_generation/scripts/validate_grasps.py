"""
Last modified date: 2023.02.23
Author: Ruicheng Wang
Description: validate grasps on Isaac simulator
"""

import os
import sys

sys.path.append(os.path.realpath("."))

from utils.isaac_validator import IsaacValidator
import argparse
import torch
import numpy as np
import transforms3d
from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.hand_model_type import (
    HandModelType,
    handmodeltype_to_joint_names,
    handmodeltype_to_hand_root_hand_file,
)
from utils.qpos_pose_conversion import qpos_to_pose, qpos_to_translation_rot_jointangles
from typing import List
import math
from utils.seed import set_seed
from utils.joint_angle_targets import compute_optimized_joint_angle_targets, OptimizationMethod


def compute_joint_angle_targets(
    args: argparse.Namespace,
    joint_names: List[str],
    data_dict: np.ndarray,
    optimization_method: OptimizationMethod,
):
    # Read in hand state and scale tensor
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    batch_size = data_dict.shape[0]
    hand_poses = []
    scale_tensor = []
    for i in range(batch_size):
        qpos = data_dict[i]["qpos"]
        scale = data_dict[i]["scale"]
        hand_pose = qpos_to_pose(
            qpos=qpos, joint_names=joint_names, unsqueeze_batch_dim=False
        ).to(device)
        hand_poses.append(hand_pose)
        scale_tensor.append(scale)
    hand_poses = torch.stack(hand_poses).to(device).requires_grad_()
    scale_tensor = torch.tensor(scale_tensor).reshape(1, -1).to(device)

    # hand model
    hand_model = HandModel(
        hand_model_type=args.hand_model_type, device=device
    )
    hand_model.set_parameters(hand_poses)

    # object model
    object_model = ObjectModel(
        data_root_path=args.mesh_path,
        batch_size_each=batch_size,
        num_samples=0,
        device=device,
    )
    object_model.initialize(args.object_code)
    object_model.object_scale_tensor = scale_tensor

    # Optimization
    joint_angle_targets_to_optimize, losses, debug_infos = compute_optimized_joint_angle_targets(
        optimization_method=optimization_method,
        hand_model=hand_model,
        object_model=object_model,
        device=device,
    )

    return joint_angle_targets_to_optimize


def main(args):
    set_seed(42)
    joint_names = handmodeltype_to_joint_names[args.hand_model_type]
    os.environ.pop("CUDA_VISIBLE_DEVICES")

    if args.index is not None:
        sim = IsaacValidator(
            hand_model_type=args.hand_model_type,
            gpu=args.gpu,
            mode="gui",
            start_with_step_mode=args.start_with_step_mode,
        )
    else:
        sim = IsaacValidator(hand_model_type=args.hand_model_type, gpu=args.gpu)

    # Read in data
    data_dict = np.load(
        os.path.join(args.grasp_path, args.object_code + ".npy"), allow_pickle=True
    )
    batch_size = data_dict.shape[0]
    translations = []
    rotations = []
    joint_angles_array = []
    scale_array = []
    E_pen_array = []
    for i in range(batch_size):
        qpos = data_dict[i]["qpos"]
        translation, rot, joint_angles = qpos_to_translation_rot_jointangles(
            qpos=qpos, joint_names=joint_names
        )
        translations.append(translation)
        rotations.append(transforms3d.euler.euler2quat(*rot))
        joint_angles_array.append(joint_angles)

        scale = data_dict[i]["scale"]
        scale_array.append(scale)

        if "E_pen" in data_dict[i]:
            E_pen_array.append(data_dict[i]["E_pen"])
        # Note: Will not do penetration check if E_pen is not found
        else:
            print(f"Warning: E_pen not found in data_dict[{i}]")
            print(
                "This is expected behavior if you are validating already validated grasps"
            )
            E_pen_array.append(0)
    E_pen_array = np.array(E_pen_array)

    if not args.no_force:
        joint_angle_targets_array = (
            compute_joint_angle_targets(
                args=args,
                joint_names=joint_names,
                data_dict=data_dict,
                optimization_method=OptimizationMethod.DESIRED_DIST_MOVE_ONE_STEP,
            )
            .detach()
            .cpu()
            .numpy()
        )
    else:
        joint_angle_targets_array = None

    # Debug with single grasp
    if args.index is not None:
        sim.set_obj_asset(
            obj_root=os.path.join(args.mesh_path, args.object_code, "coacd"),
            obj_file="coacd.urdf",
        )
        index = args.index
        sim.add_env_single_test_rotation(
            hand_rotation=rotations[index],
            hand_translation=translations[index],
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
        print(
            " = ".join(
                [
                    "E_pen < args.penetration_threshold",
                    f"{E_pen_array[index]:.7f} < {args.penetration_threshold:.7f}",
                    f"{E_pen_array[index] < args.penetration_threshold}",
                ]
            )
        )

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
                    hand_rotation=rotations[index],
                    hand_translation=translations[index],
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

        passed_penetration_threshold = E_pen_array < args.penetration_threshold
        valid = passed_simulation * passed_penetration_threshold
        print(
            f"passed_penetration_threshold: {passed_penetration_threshold.sum().item()}/{batch_size}, "
            f"passed_simulation: {passed_simulation.sum().item()}/{batch_size}, "
            f"valid = passed_simulation * passed_penetration_threshold: {valid.sum().item()}/{batch_size}"
        )
        success_data_dicts = []
        for i in range(batch_size):
            if valid[i]:
                new_data_dict = {}
                new_data_dict["qpos"] = data_dict[i]["qpos"]
                new_data_dict["scale"] = data_dict[i]["scale"]
                success_data_dicts.append(new_data_dict)

        os.makedirs(args.result_path, exist_ok=True)
        np.save(
            os.path.join(args.result_path, args.object_code + ".npy"),
            success_data_dicts,
            allow_pickle=True,
        )
    sim.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hand_model_type",
        default=HandModelType.SHADOW_HAND,
        type=HandModelType.from_string,
        choices=list(HandModelType),
    )
    parser.add_argument("--gpu", default=3, type=int)
    parser.add_argument("--val_batch", default=500, type=int)
    parser.add_argument("--mesh_path", default="../data/meshdata", type=str)
    parser.add_argument("--grasp_path", default="../data/graspdata", type=str)
    parser.add_argument("--result_path", default="../data/dataset", type=str)
    parser.add_argument(
        "--object_code", default="sem-Xbox360-d0dff348985d4f8e65ca1b579a4b8d2", type=str
    )
    # if index is received, then the debug mode is on
    parser.add_argument("--index", type=int)
    parser.add_argument("--start_with_step_mode", action="store_true")
    parser.add_argument("--no_force", action="store_true")
    parser.add_argument("--penetration_threshold", default=0.001, type=float)

    args = parser.parse_args()
    main(args)
