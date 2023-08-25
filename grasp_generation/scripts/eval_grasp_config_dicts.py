"""
Last modified date: 2023.08.24
Author: Tyler Lum
Description: eval grasps on Isaac simulator
"""

import os
import sys

sys.path.append(os.path.realpath("."))

import random
from utils.isaac_validator import IsaacValidator, ValidationType
from tap import Tap
import torch
import numpy as np
from tqdm import tqdm
from utils.hand_model import HandModel
from utils.hand_model_type import (
    HandModelType,
    handmodeltype_to_joint_names,
)
from utils.qpos_pose_conversion import (
    qpos_to_pose,
    qpos_to_translation_quaternion_jointangles,
)
from typing import List, Optional, Tuple, Dict, Any
import math
from utils.seed import set_seed
from utils.joint_angle_targets import (
    compute_optimized_joint_angle_targets_given_grasp_orientations,
)
import pathlib


class EvalGraspConfigDictsArgumentParser(Tap):
    hand_model_type: HandModelType = HandModelType.ALLEGRO_HAND
    validation_type: ValidationType = ValidationType.NO_GRAVITY_SHAKING
    gpu: int = 0
    seed: int = 1
    meshdata_root_path: pathlib.Path = pathlib.Path("../data/meshdata")
    input_grasp_config_dicts_path: pathlib.Path = pathlib.Path(
        "../data/grasp_config_dicts"
    )
    object_code_and_scale_str: str = "core-bottle-2722bec1947151b86e22e2d2f64c8cef_0_10"
    output_evaled_grasp_config_dicts_path: pathlib.Path = pathlib.Path(
        "../data/evaled_grasp_config_dicts"
    )
    # if debug_index is received, then the debug mode is on
    debug_index: Optional[int] = None
    start_with_step_mode: bool = False
    penetration_threshold: Optional[float] = None


def compute_joint_angle_targets(
    args: EvalGraspConfigDictsArgumentParser,
    hand_pose_array: List[torch.Tensor],
    grasp_orientations_array: List[torch.Tensor],
) -> np.ndarray:
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    grasp_orientations = torch.stack(grasp_orientations_array, dim=0).to(device)

    # hand model
    hand_model = HandModel(hand_model_type=args.hand_model_type, device=device)
    hand_model.set_parameters(torch.stack(hand_pose_array).to(device))

    # Optimization
    optimized_joint_angle_targets, _ = (
        compute_optimized_joint_angle_targets_given_grasp_orientations(
            joint_angles_start=hand_model.hand_pose[:, 9:],
            hand_model=hand_model,
            grasp_orientations=grasp_orientations,
        )
    )

    num_joints = len(handmodeltype_to_joint_names[hand_model.hand_model_type])
    assert optimized_joint_angle_targets.shape == (hand_model.batch_size, num_joints)

    return optimized_joint_angle_targets.detach().cpu().numpy()


def split_object_code_and_scale(object_code_and_scale_str: str) -> Tuple[str, float]:
    keyword = "_0_"
    idx = object_code_and_scale_str.rfind(keyword)
    object_code = object_code_and_scale_str[:idx]

    idx_offset_for_scale = keyword.index("0")
    object_scale = float(
        object_code_and_scale_str[idx + idx_offset_for_scale :].replace("_", ".")
    )
    return object_code, object_scale


def get_data(
    args: EvalGraspConfigDictsArgumentParser, grasp_config_dicts: List[Dict[str, Any]]
) -> Tuple[
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[torch.Tensor],
    List[torch.Tensor],
]:
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    joint_names = handmodeltype_to_joint_names[args.hand_model_type]
    batch_size = len(grasp_config_dicts)
    translation_array = []
    quaternion_array = []
    joint_angles_array = []
    hand_pose_array = []
    grasp_orientations_array = []
    for i in range(batch_size):
        data_dict = grasp_config_dicts[i]
        qpos = data_dict["qpos"]

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
        grasp_orientations_array.append(
            torch.tensor(
                data_dict["grasp_orientations"], dtype=torch.float, device=device
            )
        )
    return (
        translation_array,
        quaternion_array,
        joint_angles_array,
        hand_pose_array,
        grasp_orientations_array,
    )


def main(args: EvalGraspConfigDictsArgumentParser):
    print("=" * 80)
    print(f"args = {args}")
    print("=" * 80 + "\n")

    os.environ.pop("CUDA_VISIBLE_DEVICES")

    object_code, object_scale = split_object_code_and_scale(
        args.object_code_and_scale_str
    )
    set_seed(42)  # Want this fixed so deterministic computation

    # Read in data
    grasp_config_dict_path = (
        args.input_grasp_config_dicts_path / f"{args.object_code_and_scale_str}.npy"
    )
    grasp_config_dicts: List[Dict[str, Any]] = np.load(
        grasp_config_dict_path, allow_pickle=True
    )
    (
        translation_array,
        quaternion_array,
        joint_angles_array,
        hand_pose_array,
        grasp_orientations_array,
    ) = get_data(
        args=args,
        grasp_config_dicts=grasp_config_dicts,
    )

    # Compute joint angle targets
    joint_angle_targets_array = compute_joint_angle_targets(
        args=args,
        hand_pose_array=hand_pose_array,
        grasp_orientations_array=grasp_orientations_array,
    )

    # Debug with single grasp
    if args.debug_index is not None:
        sim = IsaacValidator(
            hand_model_type=args.hand_model_type,
            gpu=args.gpu,
            validation_type=args.validation_type,
            mode="gui",
            start_with_step_mode=args.start_with_step_mode,
        )
        sim.set_obj_asset(
            obj_root=str(args.meshdata_root_path / object_code / "coacd"),
            obj_file="coacd.urdf",
        )
        index = args.debug_index
        sim.add_env_single_test_rotation(
            hand_quaternion=quaternion_array[index],
            hand_translation=translation_array[index],
            hand_qpos=joint_angles_array[index],
            obj_scale=object_scale,
            target_qpos=joint_angle_targets_array[index],
        )
        successes = sim.run_sim()
        print(f"successes = {successes}")
        print("Ending...")
        return

    sim = IsaacValidator(
        hand_model_type=args.hand_model_type,
        gpu=args.gpu,
        validation_type=args.validation_type,
    )
    # Run validation on all grasps
    batch_size = len(grasp_config_dicts)

    # TODO: All rotations should be the same since no gravity, so this is meaningless
    num_envs_per_grasp = len(sim.test_rotations)
    sim.set_obj_asset(
        obj_root=str(args.meshdata_root_path / object_code / "coacd"),
        obj_file="coacd.urdf",
    )
    for index in range(batch_size):
        sim.add_env_all_test_rotations(
            hand_quaternion=quaternion_array[index],
            hand_translation=translation_array[index],
            hand_qpos=joint_angles_array[index],
            obj_scale=object_scale,
            target_qpos=joint_angle_targets_array[index],
        )
    successes = sim.run_sim()
    sim.reset_simulator()

    # Aggregate results
    assert len(successes) == batch_size * num_envs_per_grasp
    successes = np.array(successes).reshape(batch_size, num_envs_per_grasp)
    passed_simulation = successes.all(axis=1)

    # TODO: add penetration check E_pen
    print("WARNING: penetration check is not implemented yet")
    passed_penetration_threshold = np.ones(batch_size, dtype=np.bool8)

    passed_eval = passed_simulation * passed_penetration_threshold
    print("=" * 80)
    print(
        f"passed_penetration_threshold: {passed_penetration_threshold.sum().item()}/{batch_size}, "
        f"passed_simulation: {passed_simulation.sum().item()}/{batch_size}, "
        f"passed_eval = passed_simulation * passed_penetration_threshold: {passed_eval.sum().item()}/{batch_size}"
    )
    print("=" * 80)
    evaled_grasp_config_dicts = []
    for i in range(batch_size):
        evaled_grasp_config_dicts.append(
            {
                **grasp_config_dicts[i],
                "passed_eval": passed_eval[i],
            }
        )

    args.output_evaled_grasp_config_dicts_path.mkdir(parents=True, exist_ok=True)
    np.save(
        args.output_evaled_grasp_config_dicts_path
        / f"{args.object_code_and_scale_str}.npy",
        evaled_grasp_config_dicts,
        allow_pickle=True,
    )

    sim.destroy()

    # NOTE: Tried making this run in a loop over objects, but had issues with simulator


if __name__ == "__main__":
    args = EvalGraspConfigDictsArgumentParser().parse_args()
    main(args)
