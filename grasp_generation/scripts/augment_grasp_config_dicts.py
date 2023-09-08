import pathlib
from tap import Tap
import numpy as np
from typing import Optional, Dict


class ArgParser(Tap):
    """
    Command line arguments for this script.
    """

    input_grasp_config_dicts_path: pathlib.Path = pathlib.Path(
        "../data/2023-09-05_grasp_config_dicts_trial"
    )
    output_grasp_config_dicts_path: Optional[pathlib.Path] = None
    add_open_grasps: bool = True
    frac_open_grasps: float = 4.0
    """Relative fraction of grasps to add data for - e.g., frac_open_grasps=0.5 means add data for 50% of grasps."""
    add_closed_grasps: bool = True
    frac_closed_grasps: float = 2.0
    open_grasp_var: float = 0.075
    closed_grasp_var: float = 0.05


def generate_open_or_closed_grasps(
    grasp_config_dict: Dict[str, np.ndarray],
    frac_grasps: float,
    grasp_var: float,
    open_grasp: bool,
) -> Dict[str, np.ndarray]:
    # Compute how many times we need to copy the dataset to get the desired fraction of open grasps.
    orig_batch_size = grasp_config_dict["grasp_orientations"].shape[0]
    num_copies = 1 + int(frac_grasps)
    sample_inds = np.random.choice(
        np.repeat(np.arange(orig_batch_size), num_copies),
        int(frac_grasps * orig_batch_size),
    )

    # Build new grasp config dict.
    aug_grasp_config_dict = {}
    for key, val in grasp_config_dict.items():
        aug_grasp_config_dict[key] = val[sample_inds]

    dir_str = "open" if open_grasp else "closed"

    # Now sample joint angle perturbations to open hand.
    print(f"Adding {len(sample_inds)} {dir_str} grasps with variance {grasp_var}")
    orig_joint_angles = aug_grasp_config_dict["joint_angles"]
    deltas = grasp_var * (np.random.rand(*orig_joint_angles.shape))

    if open_grasp:
        aug_grasp_config_dict["joint_angles"] = orig_joint_angles - deltas
    else:
        aug_grasp_config_dict["joint_angles"] = orig_joint_angles + deltas

    return aug_grasp_config_dict


def main(args: ArgParser):
    print("=" * 80)
    print(f"args = {args}")
    print("=" * 80 + "\n")

    # Create output path.
    if args.output_grasp_config_dicts_path is None:
        args.output_grasp_config_dicts_path = args.input_grasp_config_dicts_path

    # Load desired grasp config dict.
    grasp_config_dict_paths = list(args.input_grasp_config_dicts_path.glob("*.npy"))

    for grasp_config_dict_path in grasp_config_dict_paths:
        print(f"Loading grasp config dicts from: {grasp_config_dict_path}")
        grasp_config_dict = np.load(grasp_config_dict_path, allow_pickle=True).item()

        # Check that grasp config dict has grasp directions.
        if "grasp_orientations" not in grasp_config_dict:
            raise ValueError(
                f"grasp_config_dict at {grasp_config_dict_path} does not have grasp_orientations. Run generate_grasp_config_dicts.py first."
            )

        # Add open grasps.
        if args.add_open_grasps:
            open_grasp_config_dict = generate_open_or_closed_grasps(
                grasp_config_dict=grasp_config_dict,
                frac_grasps=args.frac_open_grasps,
                grasp_var=args.open_grasp_var,
                open_grasp=True,
            )

            open_grasp_config_dict_path = (
                args.output_grasp_config_dicts_path
                / "opened_hand"
                / grasp_config_dict_path.name
            )

            open_grasp_config_dict_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"Saving open grasps to {open_grasp_config_dict_path}")
            np.save(
                open_grasp_config_dict_path, open_grasp_config_dict, allow_pickle=True
            )

        # Add closed grasps.
        if args.add_closed_grasps:
            closed_grasp_config_dict = generate_open_or_closed_grasps(
                grasp_config_dict=grasp_config_dict,
                frac_grasps=args.frac_closed_grasps,
                grasp_var=args.closed_grasp_var,
                open_grasp=False,
            )

            closed_grasp_config_dict_path = (
                args.output_grasp_config_dicts_path
                / "closed_hand"
                / grasp_config_dict_path.name
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
