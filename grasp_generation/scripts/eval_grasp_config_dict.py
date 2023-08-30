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
from utils.parse_object_code_and_scale import (
    parse_object_code_and_scale,
)
import pathlib


class EvalGraspConfigDictArgumentParser(Tap):
    hand_model_type: HandModelType = HandModelType.ALLEGRO_HAND
    validation_type: ValidationType = ValidationType.NO_GRAVITY_SHAKING
    gpu: int = 0
    seed: int = 1
    meshdata_root_path: pathlib.Path = pathlib.Path("../data/meshdata")
    input_grasp_config_dicts_path: pathlib.Path = pathlib.Path(
        "../data/grasp_config_dicts"
    )
    max_grasps_per_batch: int = 2500
    object_code_and_scale_str: str = "core-bottle-2722bec1947151b86e22e2d2f64c8cef_0_10"
    output_evaled_grasp_config_dicts_path: pathlib.Path = pathlib.Path(
        "../data/evaled_grasp_config_dicts"
    )
    # if debug_index is received, then the debug mode is on
    debug_index: Optional[int] = None
    start_with_step_mode: bool = False
    use_gui: bool = False
    penetration_threshold: Optional[float] = None
    record_indices: List[int] = []
    optimized: bool = False


def compute_joint_angle_targets(
    args: EvalGraspConfigDictArgumentParser,
    hand_pose_array: List[torch.Tensor],
    grasp_orientations_array: List[torch.Tensor],
) -> np.ndarray:
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    grasp_orientations = torch.stack(grasp_orientations_array, dim=0).to(device)

    # hand model
    hand_model = HandModel(hand_model_type=args.hand_model_type, device=device)
    hand_model.set_parameters(torch.stack(hand_pose_array).to(device))

    # Optimization
    (
        optimized_joint_angle_targets,
        _,
    ) = compute_optimized_joint_angle_targets_given_grasp_orientations(
        joint_angles_start=hand_model.hand_pose[:, 9:],
        hand_model=hand_model,
        grasp_orientations=grasp_orientations,
    )

    num_joints = len(handmodeltype_to_joint_names[hand_model.hand_model_type])
    assert optimized_joint_angle_targets.shape == (hand_model.batch_size, num_joints)

    return optimized_joint_angle_targets.detach().cpu().numpy()


def get_data(
    args: EvalGraspConfigDictArgumentParser, grasp_config_dicts: List[Dict[str, Any]]
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
        grasp_config_dict = grasp_config_dicts[i]
        qpos = grasp_config_dict["qpos"]

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
                grasp_config_dict["grasp_orientations"],
                dtype=torch.float,
                device=device,
            )
        )
    return (
        translation_array,
        quaternion_array,
        joint_angles_array,
        hand_pose_array,
        grasp_orientations_array,
    )


def main(args: EvalGraspConfigDictArgumentParser):
    print("=" * 80)
    print(f"args = {args}")
    print("=" * 80 + "\n")

    os.environ.pop("CUDA_VISIBLE_DEVICES")

    object_code, object_scale = parse_object_code_and_scale(
        args.object_code_and_scale_str
    )
    set_seed(42)  # Want this fixed so deterministic computation

    # Read in data
    if args.optimized:
        grasp_config_dict_path = (
            args.input_grasp_config_dicts_path
            / f"{args.object_code_and_scale_str}_optimized.npy"  # BRITTLE AF.
        )
    else:
        grasp_config_dict_path = (
            args.input_grasp_config_dicts_path / f"{args.object_code_and_scale_str}.npy"
        )

    print(f"Loading grasp config dicts from: {grasp_config_dict_path}")

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
            record=index in args.record_indices,
        )
        successes = sim.run_sim()
        print(f"successes = {successes}")
        print("Ending...")
        return

    sim = IsaacValidator(
        hand_model_type=args.hand_model_type,
        gpu=args.gpu,
        validation_type=args.validation_type,
        mode="gui" if args.use_gui else "headless",
    )
    # Run validation on all grasps
    batch_size = len(grasp_config_dicts)

    # Run for loop over minibatches of grasps.
    successes = []
    for i in tqdm(range(math.ceil(batch_size / args.max_grasps_per_batch))):
        start_index = i * args.max_grasps_per_batch
        end_index = min((i + 1) * args.max_grasps_per_batch, batch_size)
        sim.set_obj_asset(
            obj_root=str(args.meshdata_root_path / object_code / "coacd"),
            obj_file="coacd.urdf",
        )
        for index in range(start_index, end_index):
            sim.add_env_single_test_rotation(
                hand_quaternion=quaternion_array[index],
                hand_translation=translation_array[index],
                hand_qpos=joint_angles_array[index],
                obj_scale=object_scale,
                target_qpos=joint_angle_targets_array[index],
                record=index in args.record_indices,
            )
        batch_successes = sim.run_sim()
        successes.append(batch_successes)
        sim.reset_simulator()

    # Aggregate results
    successes = np.concatenate(successes, axis=0)
    assert len(successes) == batch_size
    passed_simulation = np.array(successes)

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
    args = EvalGraspConfigDictArgumentParser().parse_args()
    main(args)
