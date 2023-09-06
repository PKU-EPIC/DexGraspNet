import numpy as np
import pathlib
from tap import Tap
from collections import defaultdict


class MergeConfigDictsArgumentsParser(Tap):
    input_config_dicts_path: pathlib.Path = pathlib.Path(
        "../data/unmerged_config_dicts"
    )
    output_config_dicts_path: pathlib.Path = pathlib.Path("../data/config_dicts")


def main(args: MergeConfigDictsArgumentsParser) -> None:
    # Create dir
    args.output_config_dicts_path.mkdir()

    # Get list of all filenames
    config_dict_filepaths = [
        path for path in list(args.input_config_dicts_path.rglob("*.npy"))
    ]
    filenames = list(set([path.name for path in config_dict_filepaths]))

    for filename in filenames:
        filepaths_with_filename = [
            path for path in config_dict_filepaths if path.name == filename
        ]
        print(f"Found {len(filepaths_with_filename)} files with filename {filename}")

        # Append all
        num_grasps = 0
        combined_grasp_config_dict = defaultdict(list)
        for filepath in filepaths_with_filename:
            grasp_config_dict = np.load(filepath, allow_pickle=True).item()
            num_grasps += grasp_config_dict["trans"].shape[0]

            for key, value in grasp_config_dict.items():
                combined_grasp_config_dict[key].append(value)
        combined_grasp_config_dict = dict(combined_grasp_config_dict)

        # Merge
        for key, value in combined_grasp_config_dict.items():
            combined_grasp_config_dict[key] = np.concatenate(value, axis=0)

        # Shape checks.
        for key, value in combined_grasp_config_dict.items():
            assert value.shape[0] == num_grasps, f"{key}: {value.shape}, {num_grasps}"

        new_filepath = args.output_config_dicts_path / filename
        print(f"Saving {num_grasps} grasps to {new_filepath}")
        print()
        np.save(new_filepath, combined_grasp_config_dict, allow_pickle=True)


if __name__ == "__main__":
    args = MergeConfigDictsArgumentsParser().parse_args()
    main(args)
