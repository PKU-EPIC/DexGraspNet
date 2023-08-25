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
    sim_batch_size: int = 500
    seed: int = 1
    meshdata_root_path: pathlib.Path = pathlib.Path("../data/meshdata")
    input_grasp_config_dicts_path: pathlib.Path = pathlib.Path(
        "../data/grasp_config_dicts"
    )
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
) -> torch.Tensor:
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    grasp_orientations = torch.stack(grasp_orientations_array, dim=0).to(device)

    # hand model
    hand_model = HandModel(hand_model_type=args.hand_model_type, device=device)
    hand_model.set_parameters(torch.stack(hand_pose_array).to(device))

    # Optimization
    optimized_joint_angle_targets = (
        compute_optimized_joint_angle_targets_given_grasp_orientations(
            joint_angles_start=hand_model.hand_pose[:, 9:],
            hand_model=hand_model,
            grasp_orientations=grasp_orientations,
        )
    )

    return optimized_joint_angle_targets


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
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
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
    os.environ.pop("CUDA_VISIBLE_DEVICES")

    grasp_config_dict_paths = [
        path for path in args.input_grasp_config_dicts_path.iterdir()
    ]
    print(f"len(grasp_config_dict_paths): {len(grasp_config_dict_paths)}")
    print(f"First 10: {[path.name for path in grasp_config_dict_paths[:10]]}")
    random.Random(args.seed).shuffle(grasp_config_dict_paths)

    pbar = tqdm(
        grasp_config_dict_paths,
        desc="Generating evaled_grasp_config_dicts",
        dynamic_ncols=True,
    )
    for grasp_config_dict_path in pbar:
        object_code_and_scale_str = grasp_config_dict_path.stem
        object_code, object_scale = split_object_code_and_scale(
            object_code_and_scale_str
        )

        set_seed(42)  # Want this fixed so deterministic computation

        # Read in data
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

        # Run validation on all grasps
        sim = IsaacValidator(
            hand_model_type=args.hand_model_type,
            gpu=args.gpu,
            validation_type=args.validation_type,
        )

        batch_size = len(grasp_config_dicts)
        assert (
            batch_size % args.sim_batch_size == 0
        ), f"{batch_size} % {args.sim_batch_size} == {batch_size % args.sim_batch_size}"

        # TODO: All rotations should be the same since no gravity, so this is meaningless
        num_envs_per_grasp = len(sim.test_rotations)
        all_successes = []
        num_sim_batches = math.ceil(batch_size / args.sim_batch_size)
        for sim_batch_idx in range(num_sim_batches):
            start_offset = sim_batch_idx * args.sim_batch_size
            end_offset = (sim_batch_idx + 1) * args.sim_batch_size

            sim.set_obj_asset(
                obj_root=str(args.meshdata_root_path / object_code / "coacd"),
                obj_file="coacd.urdf",
            )
            for index in range(start_offset, end_offset):
                sim.add_env_all_test_rotations(
                    hand_quaternion=quaternion_array[index],
                    hand_translation=translation_array[index],
                    hand_qpos=joint_angles_array[index],
                    obj_scale=object_scale,
                    target_qpos=joint_angle_targets_array[index],
                )
            successes = sim.run_sim()

            assert len(successes) == args.sim_batch_size * num_envs_per_grasp
            all_successes.append(successes)
            sim.reset_simulator()

        # Aggregate results
        all_successes = np.array(all_successes)
        assert all_successes.shape == (
            num_sim_batches,
            args.sim_batch_size * num_envs_per_grasp,
        )
        all_successes = all_successes.reshape(batch_size, num_envs_per_grasp)
        passed_simulation = np.zeros(batch_size, dtype=np.bool8)
        for i in range(batch_size):
            passed_simulation[i] = np.array(
                sum(all_successes[i, :]) == num_envs_per_grasp
            )

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
            / f"{object_code_and_scale_str}.npy",
            evaled_grasp_config_dicts,
            allow_pickle=True,
        )

        # sim.destroy()


if __name__ == "__main__":
    args = EvalGraspConfigDictsArgumentParser().parse_args()
    main(args)
