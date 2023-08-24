"""
Last modified date: 2023.02.23
Author: Ruicheng Wang
Description: validate grasps on Isaac simulator
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
    compute_optimized_joint_angle_targets,
    OptimizationMethod,
)
from utils.energy import _cal_hand_object_penetration


class ValidateGraspArgumentParser(Tap):
    hand_model_type: HandModelType = HandModelType.ALLEGRO_HAND
    optimization_method: OptimizationMethod = (
        OptimizationMethod.DESIRED_DIST_TOWARDS_OBJECT_SURFACE_MULTIPLE_STEPS
    )
    validation_type: ValidationType = ValidationType.NO_GRAVITY_SHAKING
    gpu: int = 0
    val_batch: int = 500
    mesh_path: str = "../data/meshdata"
    grasp_path: str = "../data/graspdata"
    result_path: str = "../data/dataset"
    object_code: str = "sem-Xbox360-d0dff348985d4f8e65ca1b579a4b8d2"
    # if debug_index is received, then the debug mode is on
    debug_index: Optional[int] = None
    debug_only_valid_grasps: bool = False
    start_with_step_mode: bool = False
    no_force: bool = False
    penetration_threshold: Optional[float] = None


def compute_joint_angle_targets(
    args: ValidateGraspArgumentParser,
    hand_pose_array: List[torch.Tensor],
    scale_array: List[float],
) -> torch.Tensor:
    assert len(hand_pose_array) == len(scale_array)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    batch_size = len(hand_pose_array)

    # hand model
    hand_model = HandModel(hand_model_type=args.hand_model_type, device=device)
    hand_model.set_parameters(torch.stack(hand_pose_array).to(device))

    # object model
    object_model = ObjectModel(
        meshdata_root_path=args.mesh_path,
        batch_size_each=batch_size,
        num_samples=0,
        device=device,
    )
    object_model.initialize(args.object_code)
    object_model.object_scale_tensor = (
        torch.tensor(scale_array).reshape(1, -1).to(device)
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


def compute_E_pen(
    args: ValidateGraspArgumentParser,
    hand_pose_array: List[torch.Tensor],
    scale_array: List[float],
) -> torch.Tensor:
    assert len(hand_pose_array) == len(scale_array)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    batch_size = len(hand_pose_array)

    # hand model
    hand_model = HandModel(hand_model_type=args.hand_model_type, device=device)
    hand_model.set_parameters(torch.stack(hand_pose_array).to(device))

    # object model
    object_model = ObjectModel(
        meshdata_root_path=args.mesh_path,
        batch_size_each=batch_size,
        num_samples=2000,
        device=device,
    )
    object_model.initialize(args.object_code)
    object_model.object_scale_tensor = (
        torch.tensor(scale_array).reshape(1, -1).to(device)
    )  # 1 because 1 object code

    E_pen = _cal_hand_object_penetration(hand_model, object_model)
    return E_pen


def compute_link_name_to_all_contact_candidates(
    args: ValidateGraspArgumentParser,
    hand_pose_array: List[torch.Tensor],
    joint_angles: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # hand model
    hand_model = HandModel(hand_model_type=args.hand_model_type, device=device)
    hand_model.set_parameters(torch.stack(hand_pose_array).to(device))
    batch_size = len(hand_pose_array)

    current_status = hand_model.chain.forward_kinematics(
        joint_angles.to(device)
    )
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

def main(args: ValidateGraspArgumentParser):
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
    data_dicts = np.load(
        os.path.join(args.grasp_path, args.object_code + ".npy"), allow_pickle=True
    )
    batch_size = data_dicts.shape[0]
    translation_array = []
    quaternion_array = []
    joint_angles_array = []
    scale_array = []
    E_pen_array = []
    hand_pose_array = []
    for i in range(batch_size):
        qpos = data_dicts[i]["qpos"]
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

        scale = data_dicts[i]["scale"]
        scale_array.append(scale)

        if "E_pen" in data_dicts[i]:
            E_pen_array.append(data_dicts[i]["E_pen"])
        # Note: Will not do penetration check if E_pen is not found
        else:
            if i == 0:
                print(f"Warning: E_pen not found in data_dicts[{i}]")
                print(
                    "This is expected behavior if you are validating already validated grasps"
                )
            E_pen_array.append(0)
    E_pen_array = np.array(E_pen_array)

    # Compute joint angle targets
    if not args.no_force:
        joint_angle_targets_array = (
            compute_joint_angle_targets(
                args=args,
                hand_pose_array=hand_pose_array,
                scale_array=scale_array,
            )
            .detach()
            .cpu()
            .numpy()
        )
    else:
        joint_angle_targets_array = None

    # Debug with single grasp
    if args.debug_index is not None:
        sim.set_obj_asset(
            obj_root=os.path.join(args.mesh_path, args.object_code, "coacd"),
            obj_file="coacd.urdf",
        )
        index = args.debug_index
        if args.debug_only_valid_grasps:
            # Try to give a valid grasp
            valid_list = [
                data_dicts[i]["valid"]
                for i in range(batch_size)
                if "valid" in data_dicts[i]
            ]
            if len(valid_list) == 0:
                raise ValueError("No valid labels found")

            valid_idxs = [i for i, valid in enumerate(valid_list) if valid]
            print(f"valid_idxs = {valid_idxs}, len(valid_idxs) = {len(valid_idxs)}")
            if len(valid_idxs) == 0:
                raise ValueError("No valid grasps found")

            index = (
                valid_idxs[args.debug_index]
                if len(valid_idxs) > args.debug_index
                else valid_idxs[-1]
            )

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
        print(f"E_pen = {E_pen_array[index]:.7f}")
        if args.penetration_threshold is not None:
            print(f"args.penetration_threshold = {args.penetration_threshold:.7f}")
            print(
                f"E_pen < args.penetration_threshold = {E_pen_array[index] < args.penetration_threshold}"
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

        passed_penetration_threshold = (
            E_pen_array < args.penetration_threshold
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
        link_name_to_all_contact_candidates = compute_link_name_to_all_contact_candidates(args=args, hand_pose_array=hand_pose_array, joint_angles=torch.from_numpy(np.stack(joint_angles_array, axis=0)),)
        link_name_to_all_target_contact_candidates = compute_link_name_to_all_contact_candidates(args=args, hand_pose_array=hand_pose_array, joint_angles=torch.from_numpy(np.stack(joint_angle_targets_array, axis=0)),)
        for i in range(batch_size):
            success_data_dicts.append(
                {
                    "qpos": pose_to_qpos(
                        hand_pose=hand_pose_array[i], joint_names=joint_names
                    ),
                    "scale": scale_array[i],
                    "valid": valid[i],
                    "link_name_to_contact_candidates": {link_name: all_contact_candidates[i].cpu().numpy() for link_name, all_contact_candidates in link_name_to_all_contact_candidates.items()},
                    "link_name_to_target_contact_candidates": {link_name: all_target_contact_candidates[i].cpu().numpy() for link_name, all_target_contact_candidates in link_name_to_all_target_contact_candidates.items()},
                    "joint_angles": joint_angles_array[i],
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
    args = ValidateGraspArgumentParser().parse_args()
    main(args)
