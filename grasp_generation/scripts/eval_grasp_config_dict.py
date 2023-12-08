"""
Last modified date: 2023.08.24
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
from tqdm import tqdm
from utils.hand_model import HandModel
from utils.hand_model_type import (
    HandModelType,
    handmodeltype_to_joint_names,
)
from utils.pose_conversion import (
    hand_config_to_pose,
)
from pytorch3d.transforms import matrix_to_quaternion
from typing import List, Optional, Dict, Any
import math
from utils.seed import set_seed
from utils.joint_angle_targets import (
    compute_optimized_joint_angle_targets_given_grasp_orientations,
    compute_init_joint_angles_given_grasp_orientations,
)
from utils.parse_object_code_and_scale import (
    parse_object_code_and_scale,
)
from utils.energy import _cal_hand_object_penetration
from utils.object_model import ObjectModel
import pathlib


class EvalGraspConfigDictArgumentParser(Tap):
    hand_model_type: HandModelType = HandModelType.ALLEGRO_HAND
    validation_type: ValidationType = ValidationType.NO_GRAVITY_SHAKING
    gpu: int = 0
    meshdata_root_path: pathlib.Path = pathlib.Path("../data/meshdata")
    input_grasp_config_dicts_path: pathlib.Path = pathlib.Path(
        "../data/grasp_config_dicts"
    )
    max_grasps_per_batch: int = 5000
    object_code_and_scale_str: str = "core-bottle-2722bec1947151b86e22e2d2f64c8cef_0_10"
    output_evaled_grasp_config_dicts_path: pathlib.Path = pathlib.Path(
        "../data/evaled_grasp_config_dicts"
    )
    num_random_pose_noise_samples_per_grasp: Optional[int] = None
    move_fingers_back_at_init: bool = False

    # if debug_index is received, then the debug mode is on
    debug_index: Optional[int] = None
    start_with_step_mode: bool = False  # with use_gui, starts sim paused in step mode, press S to step 1 sim step, press space to toggle pause
    use_gui: bool = False
    use_cpu: bool = False  # NOTE: Tyler has had big discrepancy between using GPU vs CPU, hypothesize that CPU is safer
    penetration_threshold: Optional[float] = 5e-3  # From original DGN
    record_indices: List[int] = []


def compute_joint_angle_targets(
    args: EvalGraspConfigDictArgumentParser,
    hand_pose: torch.Tensor,
    grasp_orientations: torch.Tensor,
) -> np.ndarray:
    grasp_orientations = grasp_orientations.to(hand_pose.device)

    # hand model
    hand_model = HandModel(
        hand_model_type=args.hand_model_type, device=hand_pose.device
    )
    hand_model.set_parameters(hand_pose)

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


def compute_init_joint_angles(
    args: EvalGraspConfigDictArgumentParser,
    hand_pose: torch.Tensor,
    grasp_orientations: torch.Tensor,
) -> np.ndarray:
    grasp_orientations = grasp_orientations.to(hand_pose.device)

    # hand model
    hand_model = HandModel(
        hand_model_type=args.hand_model_type, device=hand_pose.device
    )
    hand_model.set_parameters(hand_pose)

    # Optimization
    (
        init_joint_angles,
        _,
    ) = compute_init_joint_angles_given_grasp_orientations(
        joint_angles_start=hand_model.hand_pose[:, 9:],
        hand_model=hand_model,
        grasp_orientations=grasp_orientations,
    )

    num_joints = len(handmodeltype_to_joint_names[hand_model.hand_model_type])
    assert init_joint_angles.shape == (hand_model.batch_size, num_joints)

    return init_joint_angles.detach().cpu().numpy()


def main(args: EvalGraspConfigDictArgumentParser):
    print("=" * 80)
    print(f"args = {args}")
    print("=" * 80 + "\n")

    os.environ.pop("CUDA_VISIBLE_DEVICES")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    add_random_pose_noise = args.num_random_pose_noise_samples_per_grasp is not None

    object_code, object_scale = parse_object_code_and_scale(
        args.object_code_and_scale_str
    )
    set_seed(42)  # Want this fixed so deterministic computation

    # Read in data
    grasp_config_dict_path = (
        args.input_grasp_config_dicts_path / f"{args.object_code_and_scale_str}.npy"
    )
    print(f"Loading grasp config dicts from: {grasp_config_dict_path}")
    grasp_config_dict: Dict[str, Any] = np.load(
        grasp_config_dict_path, allow_pickle=True
    ).item()
    trans: np.ndarray = grasp_config_dict["trans"]
    rot: np.ndarray = grasp_config_dict["rot"]
    joint_angles: np.ndarray = grasp_config_dict["joint_angles"]
    grasp_orientations: np.ndarray = grasp_config_dict["grasp_orientations"]

    # Compute hand pose
    quat_wxyz = matrix_to_quaternion(torch.from_numpy(rot)).numpy()
    hand_pose = hand_config_to_pose(trans, rot, joint_angles).to(device)

    # Compute joint angle targets
    joint_angle_targets_array = compute_joint_angle_targets(
        args=args,
        hand_pose=hand_pose,
        grasp_orientations=torch.from_numpy(grasp_orientations).float().to(device),
    )
    init_joint_angles = (
        compute_init_joint_angles(
            args=args,
            hand_pose=hand_pose,
            grasp_orientations=torch.from_numpy(grasp_orientations).float().to(device),
        )
        if args.move_fingers_back_at_init
        else joint_angles
    )

    # Debug with single grasp
    if args.debug_index is not None:
        sim = IsaacValidator(
            hand_model_type=args.hand_model_type,
            gpu=args.gpu,
            validation_type=args.validation_type,
            mode="gui" if args.use_gui else "headless",
            start_with_step_mode=args.start_with_step_mode,
            use_cpu=args.use_cpu,
        )
        sim.set_obj_asset(
            obj_root=str(args.meshdata_root_path / object_code / "coacd"),
            obj_file="coacd.urdf",
        )
        index = args.debug_index
        sim.add_env_single_test_rotation(
            hand_quaternion_wxyz=quat_wxyz[index],
            hand_translation=trans[index],
            hand_qpos=init_joint_angles[index],
            obj_scale=object_scale,
            target_qpos=joint_angle_targets_array[index],
            add_random_pose_noise=add_random_pose_noise,
            record=index in args.record_indices,
        )
        passed_simulation, passed_new_penetration_test = sim.run_sim()
        sim.reset_simulator()
        print(
            f"passed_simulation = {passed_simulation} ({np.mean(passed_simulation) * 100:.2f}%)"
        )
        print(
            f"passed_new_penetration_test = {passed_new_penetration_test} ({np.mean(passed_new_penetration_test) * 100:.2f}%)"
        )
        print("Ending...")
        return

    sim = IsaacValidator(
        hand_model_type=args.hand_model_type,
        gpu=args.gpu,
        validation_type=args.validation_type,
        mode="gui" if args.use_gui else "headless",
        use_cpu=args.use_cpu,
    )
    # Run validation on all grasps
    batch_size = trans.shape[0]
    hand_model = HandModel(hand_model_type=args.hand_model_type, device=device)

    # Some final shape checking.
    assert quat_wxyz.shape == (batch_size, 4)
    assert joint_angles.shape == (batch_size, 16)
    assert hand_pose.shape == (batch_size, 3 + 6 + 16)
    assert grasp_orientations.shape == (batch_size, hand_model.num_fingers, 3, 3)
    assert joint_angle_targets_array.shape == (batch_size, 16)
    assert init_joint_angles.shape == (batch_size, 16)

    # Run for loop over minibatches of grasps.
    passed_simulation_array = []
    passed_new_penetration_test_array = []
    E_pen_array = []
    max_grasps_per_batch = (
        args.max_grasps_per_batch
        if args.num_random_pose_noise_samples_per_grasp is None
        else args.max_grasps_per_batch
        // (args.num_random_pose_noise_samples_per_grasp + 1)
    )
    pbar = tqdm(range(math.ceil(batch_size / max_grasps_per_batch)))
    for i in pbar:
        start_index = i * max_grasps_per_batch
        end_index = min((i + 1) * max_grasps_per_batch, batch_size)
        sim.set_obj_asset(
            obj_root=str(args.meshdata_root_path / object_code / "coacd"),
            obj_file="coacd.urdf",
        )
        for index in range(start_index, end_index):
            sim.add_env_single_test_rotation(
                hand_quaternion_wxyz=quat_wxyz[index],
                hand_translation=trans[index],
                hand_qpos=init_joint_angles[index],
                obj_scale=object_scale,
                target_qpos=joint_angle_targets_array[index],
                add_random_pose_noise=add_random_pose_noise,
                record=index in args.record_indices,
            )

            if args.num_random_pose_noise_samples_per_grasp is not None:
                for _ in range(args.num_random_pose_noise_samples_per_grasp):
                    sim.add_env_single_test_rotation(
                        hand_quaternion_wxyz=quat_wxyz[index],
                        hand_translation=trans[index],
                        hand_qpos=init_joint_angles[index],
                        obj_scale=object_scale,
                        target_qpos=joint_angle_targets_array[index],
                        add_random_pose_noise=add_random_pose_noise,
                        record=index in args.record_indices,
                    )

        passed_simulation, passed_new_penetration_test = sim.run_sim()
        passed_simulation_array.extend(passed_simulation)
        passed_new_penetration_test_array.extend(passed_new_penetration_test)
        sim.reset_simulator()
        pbar.set_description(f"mean_success = {np.mean(passed_simulation_array)}")

        hand_model.set_parameters(hand_pose[start_index:end_index])

        object_model = ObjectModel(
            meshdata_root_path=str(args.meshdata_root_path),
            batch_size_each=end_index - start_index,
            num_samples=2000,
            device=device,
        )
        object_model.initialize(object_code, object_scale)

        # TODO: Do we need to use thres_pen param here? Does threshold change? Do we even need passed_penetration_threshold now?
        batch_E_pen_array = _cal_hand_object_penetration(
            hand_model=hand_model, object_model=object_model, reduction="max"
        )
        E_pen_array.extend(batch_E_pen_array.flatten().tolist())

    # Aggregate results
    passed_simulation_array = np.array(passed_simulation_array)
    passed_new_penetration_test_array = np.array(passed_new_penetration_test_array)
    E_pen_array = np.array(E_pen_array)

    if args.num_random_pose_noise_samples_per_grasp is not None:
        passed_simulation_array = passed_simulation_array.reshape(
            batch_size, args.num_random_pose_noise_samples_per_grasp + 1
        )
        passed_simulation_without_noise = passed_simulation_array[:, 0]
        passed_simulation_with_noise = passed_simulation_array[:, 1:]
        # Use mean of all noise samples
        mean_passed_simulation_with_noise = passed_simulation_with_noise.mean(axis=1)
        passed_simulation_array = mean_passed_simulation_with_noise

        passed_new_penetration_test_array = passed_new_penetration_test_array.reshape(
            batch_size, args.num_random_pose_noise_samples_per_grasp + 1
        )
        passed_new_penetration_test_without_noise = passed_new_penetration_test_array[
            :, 0
        ]
        passed_new_penetration_test_with_noise = passed_new_penetration_test_array[
            :, 1:
        ]
        # Use mean of all noise samples
        mean_passed_new_penetration_test_with_noise = (
            passed_new_penetration_test_with_noise.mean(axis=1)
        )
        passed_new_penetration_test_array = mean_passed_new_penetration_test_with_noise

    assert passed_simulation_array.shape == (batch_size,)
    assert passed_new_penetration_test_array.shape == (batch_size,)
    assert E_pen_array.shape == (batch_size,)

    if args.penetration_threshold is None:
        print("WARNING: penetration check skipped")
        passed_penetration_threshold_array = np.ones(batch_size, dtype=np.bool8)
    else:
        passed_penetration_threshold_array = E_pen_array < args.penetration_threshold

    # TODO: Remove these prints
    DEBUG = True
    if DEBUG:
        print(
            f"passed_simulation_array = {passed_simulation_array} ({passed_simulation_array.mean() * 100:.2f}%)"
        )
        print(
            f"passed_new_penetration_test_array = {passed_new_penetration_test_array} ({passed_new_penetration_test_array.mean() * 100:.2f}%)"
        )
        print(
            f"passed_penetration_threshold_array = {passed_penetration_threshold_array} ({passed_penetration_threshold_array.mean() * 100:.2f}%)"
        )
        print(f"E_pen_array = {E_pen_array}")

    passed_eval = (
        passed_simulation_array
        * passed_penetration_threshold_array
        * passed_new_penetration_test_array
    )
    sim_frac = np.mean(passed_simulation_array)
    new_pen_frac = np.mean(passed_new_penetration_test_array)
    pen_frac = np.mean(passed_penetration_threshold_array)
    eval_frac = np.mean(passed_eval)
    print("=" * 80)
    print(
        f"passed_penetration_threshold: {passed_penetration_threshold_array.sum().item()}/{batch_size} ({100*pen_frac:.2f}%),"
        f"passed_simulation: {passed_simulation_array.sum().item()}/{batch_size} ({100 * sim_frac:.2f}%),"
        f"passed_new_penetration_test: {passed_new_penetration_test_array.sum().item()}/{batch_size} ({100 * new_pen_frac:.2f}%),"
        f"passed_eval = passed_simulation * passed_penetration_threshold * passed_new_penetration_test: {passed_eval.sum().item()}/{batch_size} ({100 * eval_frac:.2f}%)"
    )
    print("=" * 80)

    # TODO: passed_penetration_threshold_array, passed_new_penetration_test_array: decide if we want to:
    #  1. Store it separately
    #  2. Replace it in the "passed_penetration_threshold" key
    #  3. And it with passed_penetration_threshold_array
    evaled_grasp_config_dict = {
        **grasp_config_dict,
        "passed_penetration_threshold": passed_penetration_threshold_array,
        "passed_new_penetration_test": passed_new_penetration_test_array,
        "passed_simulation": passed_simulation_array,
        "passed_eval": passed_eval,
        "penetration": E_pen_array,
    }

    args.output_evaled_grasp_config_dicts_path.mkdir(parents=True, exist_ok=True)
    np.save(
        args.output_evaled_grasp_config_dicts_path
        / f"{args.object_code_and_scale_str}.npy",
        evaled_grasp_config_dict,
        allow_pickle=True,
    )

    sim.destroy()

    # NOTE: Tried making this run in a loop over objects, but had issues with simulator


if __name__ == "__main__":
    args = EvalGraspConfigDictArgumentParser().parse_args()
    main(args)
