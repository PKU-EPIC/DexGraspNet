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
from utils.hand_model_type import (
    HandModelType,
    handmodeltype_to_joint_names,
)
from utils.qpos_pose_conversion import (
    qpos_to_pose,
    qpos_to_translation_quaternion_jointangles,
    pose_to_qpos,
)
from typing import List, Optional
import math
from utils.seed import set_seed
from utils.joint_angle_targets import (
    compute_optimized_joint_angle_targets_given_directions,
)
import pathlib


class EvalGraspArgumentParser(Tap):
    hand_model_type: HandModelType = HandModelType.ALLEGRO_HAND
    validation_type: ValidationType = ValidationType.NO_GRAVITY_SHAKING
    gpu: int = 0
    val_batch: int = 500
    mesh_path: str = "../data/meshdata"
    orig_grasp_path: str = "../data/graspdata"
    grasp_path: str = "/afs/cs.stanford.edu/u/tylerlum/github_repos/nerf_grasping/dexgraspnet_dicts"  # HACK
    filename: str = "sem-Wii-effdc659515ff747eb2c6725049f8f_0_15000000596046448.npy"
    result_path: str = "../data/dataset"
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
    grasp_dirs_array = torch.stack(grasp_dirs_array, dim=0).to(device)

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
    data_dicts = np.load(
        pathlib.Path(args.grasp_path) / args.filename, allow_pickle=True
    )
    # HACK: assume same scale for all grasps
    object_code_and_scale, _ = os.path.splitext(args.filename)
    tmp_idx = object_code_and_scale.rfind("_0_")
    object_code = object_code_and_scale[:tmp_idx]
    object_scale = float(object_code_and_scale[tmp_idx + 1 :].replace("_", "."))

    # CURRENT
    # grasp_generation  qpos (HandConfig)
    # grasp_validation (input: qpos) (compute closets points => fingertip targets => joint angle targets => start and end fingertips were)
    # later (convert start and end fingertips => grasp directions => sample nerfs with this dir)
    # eval_grasps (input: qpos, grasp_dirs) => fingertip targets => joint angle targets => start and end fingertips were

    # hand_config_dicts (keys: qpos, maybe random other if needed only)
    # np.save("hand_config_dicts/sem-Wii-1232111_0_05.npy")  # grasp_config_dict

    # NEW
    # grasp_generation (output: qpos (HandConfig)) hand_config_dicts
    # grasp_directions (input: [qpos, mesh], output: GraspConfig [qpos, directions]) (compute closets points => fingertip targets)  grasp_config_dicts
    # eval_grasp (input: GraspConfig [qpos, direction]): (fingertip targets => joint angle targets) use this both for validation to label and evaluation to see how good GaG and baselines are in sim (grasp_config_dicts)

    # config =AllegroGraspConfig()
    # grasp_config_dicts (keys: qpos, grasp_dirs)
    # qpos = dict("joint_1.0": 1.0, "TRx": 1.0, )  # joint_angles, wrist_eulers, wrist_translations
    # grasp_dirs: list[list[]] torch.tensor(grasp_dirs).shape == (4, 3)
    # np.save("grasp_config_dicts/sem-Wii-1232111_0_05.npy")  # grasp_config_dict

    # Baselines:
    #  * FC CEM
    #  * Classifier from depth images from mesh

    # convert_qpos+dirs_to_joint_angle_targets
    # IK impl

    # CRITICAL PATH: Get full pipeline to work on 1 object and 1 scale
    # 1. Generate large amount of data (10k - 100k) grasps for 1 object and 1 scale: inputs: (mesh, compute, DGN)  outputs: grasp_config_dicts
    # 2. Train a NeRF on this object: inputs: (nerf data, compute to train) => outputs (nerf checkpoints)
    # 3. Generate metric data with this dataset (sampling nerf): inputs: (nerf checkpoints, grasp_config_dicts) => outputs (classifier data)
    # 4. Training classifier inputs: classifier data, compute, => outputs: nn weights, config with nn shape
    # 5. Run Optimizer: input: nerf checkpoint, nn weights => outputs: grasp_config_dicts
    # 6. Run eval: input: grasp_config_dicts => success rate

    orig_data_dicts = np.load(
        pathlib.Path(args.orig_grasp_path) / f"{object_code}.npy", allow_pickle=True
    )
    batch_size = len(data_dicts)
    translation_array = []
    quaternion_array = []
    joint_angles_array = []
    scale_array = []
    hand_pose_array = []
    grasp_dirs_array = []
    for i in range(batch_size):
        data_dict = data_dicts[i]
        qpos = data_dict["qpos"]

        # Verify that qpos is set up correctly
        orig_qpos = orig_data_dicts[i]["qpos"]
        qpos_keys = list(qpos.keys())
        for key in set([*qpos_keys, *orig_qpos.keys()]):
            if key not in qpos_keys or key not in orig_qpos.keys():
                continue
            assert np.allclose(
                qpos[key], orig_qpos[key]
            ), f"{key}: {qpos[key]} != {orig_qpos[key]}"

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
        grasp_dirs_array.append(
            torch.tensor(data_dict["grasp_dirs"], dtype=torch.float)
        )

        # TODO: Figure out how we interface the scale and object with the config
        scale_array.append(object_scale)

    # Compute joint angle targets
    joint_angle_targets_array = compute_joint_angle_targets(
        args=args,
        hand_pose_array=hand_pose_array,
        grasp_dirs_array=grasp_dirs_array,
    )

    # Debug with single grasp
    if args.debug_index is not None:
        sim.set_obj_asset(
            obj_root=os.path.join(args.mesh_path, object_code, "coacd"),
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
                obj_root=os.path.join(args.mesh_path, object_code, "coacd"),
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
            os.path.join(args.result_path, object_code + ".npy"),
            success_data_dicts,
            allow_pickle=True,
        )
    sim.destroy()


if __name__ == "__main__":
    args = EvalGraspArgumentParser().parse_args()
    main(args)
