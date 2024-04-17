import os
import sys

sys.path.append(os.path.realpath("."))

import pathlib
from tap import Tap
import numpy as np
from typing import Optional, Dict, List
from utils.seed import set_seed
from scipy.spatial.transform import Rotation as R


class ArgParser(Tap):
    """
    Command line arguments for this script.
    """

    input_grasp_config_dicts_path: pathlib.Path = pathlib.Path(
        "../data/2023-09-05_grasp_config_dicts_trial"
    )
    output_grasp_config_dicts_path: Optional[pathlib.Path] = None
    mid_optimization_steps: List[int] = []

    num_augmentations_per_grasp: int = 10000
    trans_noise_max: float = 0.03
    rot_deg_noise_max: float = 0.0
    augment_only_successes: bool = False
    no_continue: bool = False


def generate_augmented_pose_grasps(
    grasp_config_dict: Dict[str, np.ndarray],
    num_augmentations_per_grasp: int,
    trans_noise_max: float,
    rot_deg_noise_max: float,
    augment_only_successes: bool,
) -> Dict[str, np.ndarray]:
    orig_batch_size = grasp_config_dict["grasp_orientations"].shape[0]
    if augment_only_successes:
        assert "passed_simulation" in grasp_config_dict.keys()
        PASSED_THRESHOLD = 0.5
        inds = np.argwhere(grasp_config_dict["passed_simulation"] > PASSED_THRESHOLD)

        if inds.size == 0:
            print(f"WARNING: No successful grasps found, using first one")
            inds = np.arange(1).reshape(-1, 1)
    else:
        inds = np.arange(orig_batch_size).reshape(-1, 1)

    assert inds.shape == (inds.size, 1)

    # Repeat inds to get desired number of augmentations per grasp.
    repeated_inds = np.repeat(
        inds, repeats=num_augmentations_per_grasp, axis=-1
    ).flatten()
    assert repeated_inds.shape == (inds.size * num_augmentations_per_grasp,)

    # Build new grasp config dict.
    aug_grasp_config_dict = {}
    for key, val in grasp_config_dict.items():
        aug_grasp_config_dict[key] = val[repeated_inds]

    print(f"Adding {len(repeated_inds)} grasps with trans_noise_max = {trans_noise_max}, rot_deg_noise_max = {rot_deg_noise_max}")

    # Sample trans perturbations
    trans_noise_shape = aug_grasp_config_dict["trans"].shape
    trans_noise = np.random.uniform(low=-trans_noise_max, high=trans_noise_max, size=trans_noise_shape)
    aug_grasp_config_dict["trans"] += trans_noise

    # Sample rot perturbations
    rot_noise_shape = trans_noise_shape
    rot_deg_noise = np.random.uniform(low=-rot_deg_noise_max, high=rot_deg_noise_max, size=rot_noise_shape)
    orig_rpy = R.from_matrix(aug_grasp_config_dict["rot"]).as_euler('xyz', degrees=True)
    new_rpy = orig_rpy + rot_deg_noise
    aug_grasp_config_dict["rot"][:] = R.from_euler('xyz', new_rpy, degrees=True).as_matrix()

    return aug_grasp_config_dict


def augment_pose_grasp_config_dicts(
    args: ArgParser,
    input_grasp_config_dicts_path: pathlib.Path,
    output_grasp_config_dicts_path: pathlib.Path,
) -> None:
    # Load desired grasp config dict.
    grasp_config_dict_paths = list(input_grasp_config_dicts_path.glob("*.npy"))

    existing_grasp_config_dicts = (
        list(output_grasp_config_dicts_path.glob("*.npy")) if output_grasp_config_dicts_path.exists() else []
    )

    existing_object_code_and_scale_strs = [
        path.stem for path in existing_grasp_config_dicts
    ]

    if args.no_continue and len(existing_object_code_and_scale_strs) > 0:
        raise ValueError(
            f"Found {len(existing_object_code_and_scale_strs)} existing grasp config dicts in {output_grasp_config_dicts_path}."
            + " Set no_continue to False to continue training on these objects, or change output path."
        )
    elif len(existing_object_code_and_scale_strs) > 0:
        print(
            f"Found {len(existing_object_code_and_scale_strs)} existing grasp config dicts in {output_grasp_config_dicts_path}."
            + " Continuing training on these objects."
        )
        grasp_config_dict_paths = [
            pp
            for pp in grasp_config_dict_paths
            if pp.stem not in existing_object_code_and_scale_strs
        ]

        print(f"Found {len(grasp_config_dict_paths)} new grasp config dicts to add.")

    for grasp_config_dict_path in grasp_config_dict_paths:
        print(f"Loading grasp config dicts from: {grasp_config_dict_path}")
        grasp_config_dict = np.load(grasp_config_dict_path, allow_pickle=True).item()

        # Check that grasp config dict has grasp directions.
        if "grasp_orientations" not in grasp_config_dict:
            raise ValueError(
                f"grasp_config_dict at {grasp_config_dict_path} does not have grasp_orientations. Run generate_grasp_config_dicts.py first."
            )

        # Add augmented pose grasps
        output_grasp_config_dict = generate_augmented_pose_grasps(
            grasp_config_dict=grasp_config_dict,
            num_augmentations_per_grasp=args.num_augmentations_per_grasp,
            trans_noise_max=args.trans_noise_max,
            rot_deg_noise_max=args.rot_deg_noise_max,
            augment_only_successes=args.augment_only_successes,
        )

        output_grasp_config_dict_path = (
            output_grasp_config_dicts_path / grasp_config_dict_path.name
        )

        output_grasp_config_dict_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Saving output grasps to {output_grasp_config_dict_path}")
        np.save(
            output_grasp_config_dict_path,
            output_grasp_config_dict,
            allow_pickle=True,
        )


def main(args: ArgParser) -> None:
    print("=" * 80)
    print(f"args = {args}")
    print("=" * 80 + "\n")

    set_seed(-1)

    # Create output path.
    if args.output_grasp_config_dicts_path is None:
        output_grasp_config_dicts_path = args.input_grasp_config_dicts_path
    else:
        output_grasp_config_dicts_path = args.output_grasp_config_dicts_path

    augment_pose_grasp_config_dicts(
        args=args,
        input_grasp_config_dicts_path=args.input_grasp_config_dicts_path,
        output_grasp_config_dicts_path=output_grasp_config_dicts_path,
    )

    for mid_optimization_step in args.mid_optimization_steps:
        print("!" * 80)
        print(f"Running mid_optimization_step: {mid_optimization_step}")
        print("!" * 80 + "\n")
        mid_optimization_input_grasp_config_dicts_path = (
            args.input_grasp_config_dicts_path
            / "mid_optimization"
            / f"{mid_optimization_step}"
        )
        mid_optimization_output_path = (
            output_grasp_config_dicts_path / "mid_optimization" / f"{mid_optimization_step}"
        )
        augment_pose_grasp_config_dicts(
            args=args,
            input_grasp_config_dicts_path=mid_optimization_input_grasp_config_dicts_path,
            output_grasp_config_dicts_path=mid_optimization_output_path,
        )


if __name__ == "__main__":
    args = ArgParser().parse_args()
    main(args)
