import pathlib
from tap import Tap
import numpy as np
from typing import Optional


class ArgParser(Tap):
    """
    Command line arguments for this script.
    """

    input_grasp_config_dicts_path: pathlib.Path = pathlib.Path(
        "../data/2023-09-05_grasp_config_dicts_trial"
    )
    output_grasp_config_dicts_path: Optional[pathlib.Path] = None
    object_code_and_scale_str: str = "core-bottle-2722bec1947151b86e22e2d2f64c8cef_0_10"
    add_open_grasps: bool = True
    frac_open_grasps: float = 1.0
    """Relative fraction of grasps to add data for - e.g., frac_open_grasps=0.5 means add data for 50% of grasps."""
    add_closed_grasps: bool = True
    frac_closed_grasps: float = 1.0
    open_grasp_var: float = 0.075
    closed_grasp_var: float = 0.05


def main(args: ArgParser):
    print("=" * 80)
    print(f"args = {args}")
    print("=" * 80 + "\n")

    # Create output path.
    if args.output_grasp_config_dicts_path is None:
        args.output_grasp_config_dicts_path = args.input_grasp_config_dicts_path

    # Load desired grasp config dict.
    grasp_config_dict_path = (
        args.input_grasp_config_dicts_path / f"{args.object_code_and_scale_str}.npy"
    )

    print(f"Loading grasp config dicts from: {grasp_config_dict_path}")
    grasp_config_dict = np.load(grasp_config_dict_path, allow_pickle=True).item()

    # Check that grasp config dict has grasp directions.
    if "grasp_orientations" not in grasp_config_dict:
        raise ValueError(
            f"grasp_config_dict at {grasp_config_dict_path} does not have grasp_orientations. Run generate_grasp_config_dicts.py first."
        )

    orig_batch_size = grasp_config_dict["grasp_orientations"].shape[0]
    # Add open grasps.
    if args.add_open_grasps:
        # Compute how many times we need to copy the dataset to get the desired fraction of open grasps.
        num_copies = 1 + int(args.frac_open_grasps)
        sample_inds = np.random.choice(
            np.repeat(np.arange(orig_batch_size), num_copies),
            int(args.frac_open_grasps * orig_batch_size),
        )

        # Build new grasp config dict.
        open_grasp_config_dict = {}
        for key, val in grasp_config_dict.items():
            open_grasp_config_dict[key] = val[sample_inds]

        # Now sample joint angle perturbations to open hand.
        print(
            f"Adding {len(sample_inds)} open grasps with variance {args.open_grasp_var}"
        )
        orig_joint_angles = open_grasp_config_dict["joint_angles"]
        deltas = args.open_grasp_var * (np.random.rand(*orig_joint_angles.shape))

        open_grasp_config_dict["joint_angles"] = orig_joint_angles - deltas

        open_grasp_config_dict_path = (
            args.output_grasp_config_dicts_path
            / "opened_hand"
            / f"{args.object_code_and_scale_str}.npy"
        )

        open_grasp_config_dict_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Saving open grasps to {open_grasp_config_dict_path}")
        np.save(open_grasp_config_dict_path, open_grasp_config_dict, allow_pickle=True)

    # Add closed grasps.
    if args.add_closed_grasps:
        num_copies = 1 + int(args.frac_closed_grasps)
        sample_inds = np.random.choice(
            np.repeat(np.arange(orig_batch_size), num_copies),
            int(args.frac_closed_grasps * orig_batch_size),
        )

        # Build new grasp config dict.
        closed_grasp_config_dict = {}
        for key, val in grasp_config_dict.items():
            closed_grasp_config_dict[key] = val[sample_inds]

        print(
            f"Adding {len(sample_inds)} closed grasps with variance {args.closed_grasp_var}"
        )
        orig_joint_angles = closed_grasp_config_dict["joint_angles"]
        deltas = args.closed_grasp_var * (np.random.rand(*orig_joint_angles.shape))
        closed_grasp_config_dict["joint_angles"] = orig_joint_angles + deltas

        closed_grasp_config_dict_path = (
            args.output_grasp_config_dicts_path
            / "closed_hand"
            / f"{args.object_code_and_scale_str}.npy"
        )

        closed_grasp_config_dict_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Saving closed grasps to {closed_grasp_config_dict_path}")
        np.save(
            closed_grasp_config_dict_path,
            closed_grasp_config_dict,
            allow_pickle=True,
        )


if __name__ == "__main__":
    args = ArgParser().parse_args()
    main(args)
