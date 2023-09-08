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
from utils.pose_conversion import (
    hand_config_to_pose,
)
from pytorch3d.transforms import matrix_to_quaternion
from typing import List, Optional, Tuple, Dict, Any
import math
from utils.seed import set_seed
from utils.joint_angle_targets import (
    compute_optimized_joint_angle_targets_given_grasp_orientations,
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
    max_grasps_per_batch: int = 2500
    object_code_and_scale_str: str = "core-bottle-2722bec1947151b86e22e2d2f64c8cef_0_10"
    output_evaled_grasp_config_dicts_path: pathlib.Path = pathlib.Path(
        "../data/evaled_grasp_config_dicts"
    )
    # if debug_index is received, then the debug mode is on
    debug_index: Optional[int] = None
    start_with_step_mode: bool = False  # with use_gui, starts sim paused in step mode, press S to step 1 sim step, press space to toggle pause
    use_gui: bool = False
    use_cpu: bool = False  # NOTE: Tyler has had big discrepancy between using GPU vs CPU, hypothesize that CPU is safer
    penetration_threshold: Optional[float] = 0.001  # From original DGN
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


def get_data(
    args: EvalGraspConfigDictArgumentParser, grasp_config_dict: Dict[str, np.ndarray]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,]:
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    trans = torch.tensor(grasp_config_dict["trans"], device=device, dtype=torch.float)
    rot = torch.tensor(grasp_config_dict["rot"], device=device, dtype=torch.float)
    quat_wxyz = matrix_to_quaternion(rot)  #
    joint_angles = torch.tensor(
        grasp_config_dict["joint_angles"], device=device, dtype=torch.float
    )
    hand_pose = hand_config_to_pose(trans, rot, joint_angles)
    grasp_orientations = torch.tensor(
        grasp_config_dict["grasp_orientations"],
        dtype=torch.float,
        device=device,
    )
    return (
        trans,
        quat_wxyz,
        joint_angles,
        hand_pose,
        grasp_orientations,
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
    grasp_config_dict_path = (
        args.input_grasp_config_dicts_path / f"{args.object_code_and_scale_str}.npy"
    )

    print(f"Loading grasp config dicts from: {grasp_config_dict_path}")

    grasp_config_dict: Dict[str, Any] = np.load(
        grasp_config_dict_path, allow_pickle=True
    ).item()
    (
        trans,
        quat_wxyz,
        joint_angles,
        hand_pose,
        grasp_orientations,
    ) = get_data(
        args=args,
        grasp_config_dict=grasp_config_dict,
    )

    # Compute joint angle targets
    joint_angle_targets_array = compute_joint_angle_targets(
        args=args,
        hand_pose=hand_pose,
        grasp_orientations=grasp_orientations,
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
            hand_quaternion=quat_wxyz[index],
            hand_translation=trans[index],
            hand_qpos=joint_angles[index],
            obj_scale=object_scale,
            target_qpos=joint_angle_targets_array[index],
            record=index in args.record_indices,
        )
        successes = sim.run_sim()
        sim.reset_simulator()
        print(f"successes = {successes}")
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

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    hand_model = HandModel(hand_model_type=args.hand_model_type, device=device)

    # Some final shape checking.
    assert quat_wxyz.shape == (batch_size, 4)
    assert joint_angles.shape == (batch_size, 16)
    assert hand_pose.shape == (batch_size, 3 + 6 + 16)
    assert grasp_orientations.shape == (batch_size, hand_model.num_fingers, 3, 3)

    # Run for loop over minibatches of grasps.
    successes = []
    E_pen_array = []
    pbar = tqdm(range(math.ceil(batch_size / args.max_grasps_per_batch)))
    for i in pbar:
        start_index = i * args.max_grasps_per_batch
        end_index = min((i + 1) * args.max_grasps_per_batch, batch_size)
        sim.set_obj_asset(
            obj_root=str(args.meshdata_root_path / object_code / "coacd"),
            obj_file="coacd.urdf",
        )
        for index in range(start_index, end_index):
            sim.add_env_single_test_rotation(
                hand_quaternion=quat_wxyz[index],
                hand_translation=trans[index],
                hand_qpos=joint_angles[index],
                obj_scale=object_scale,
                target_qpos=joint_angle_targets_array[index],
                record=index in args.record_indices,
            )
        batch_successes = sim.run_sim()
        successes.extend(batch_successes)
        sim.reset_simulator()
        pbar.set_description(f"mean_success = {np.mean(successes)}")

        hand_model.set_parameters(hand_pose[start_index:end_index])

        object_model = ObjectModel(
            meshdata_root_path=str(args.meshdata_root_path),
            batch_size_each=end_index - start_index,
            num_samples=2000,
            device=device,
        )
        object_model.initialize(object_code, object_scale)

        batch_E_pen_array = _cal_hand_object_penetration(
            hand_model=hand_model, object_model=object_model, reduction="max"
        )
        E_pen_array.extend(batch_E_pen_array.flatten().tolist())

    # Aggregate results
    successes = np.array(successes)
    assert len(successes) == batch_size
    passed_simulation = np.array(successes)
    E_pen_array = np.array(E_pen_array)
    assert len(E_pen_array) == batch_size

    if args.penetration_threshold is None:
        print("WARNING: penetration check skipped")
        passed_penetration_threshold = np.ones(batch_size, dtype=np.bool8)
    else:
        passed_penetration_threshold = E_pen_array < args.penetration_threshold

    passed_eval = passed_simulation * passed_penetration_threshold
    print("=" * 80)
    print(
        f"passed_penetration_threshold: {passed_penetration_threshold.sum().item()}/{batch_size}, "
        f"passed_simulation: {passed_simulation.sum().item()}/{batch_size}, "
        f"passed_eval = passed_simulation * passed_penetration_threshold: {passed_eval.sum().item()}/{batch_size}"
    )
    print("=" * 80)
    evaled_grasp_config_dict = {
        **grasp_config_dict,
        "passed_penetration_threshold": passed_penetration_threshold,
        "passed_simulation": passed_simulation,
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
