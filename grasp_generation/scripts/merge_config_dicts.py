import numpy as np
import pathlib
from tap import Tap
from typing import List, Optional


class MergeConfigDictsArgumentsParser(Tap):
    input_config_dicts_paths: List[pathlib.Path]
    output_config_dicts_path: pathlib.Path = pathlib.Path("../data/config_dicts")


def main(args: MergeConfigDictsArgumentsParser) -> None:
    # Create dir
    args.output_config_dicts_path.mkdir(parents=True, exist_ok=True)

    # Get list of all filenames
    input_config_dict_filepaths = []
    for input_config_dicts_path in args.input_config_dicts_paths:
        input_config_dict_filepaths += [
            path for path in list(input_config_dicts_path.rglob("*.npy"))
        ]
    filenames = list(set([path.name for path in input_config_dict_filepaths]))

    for filename in filenames:
        filepaths_with_filename = [
            path for path in input_config_dict_filepaths if path.name == filename
        ]
        print(f"Found {len(filepaths_with_filename)} files with filename {filename}")

        # Append all
        num_grasps = 0
        combined_grasp_config_dict = None
        for filepath in filepaths_with_filename:
            if combined_grasp_config_dict is None:
                combined_grasp_config_dict = np.load(filepath, allow_pickle=True).item()
                num_grasps += combined_grasp_config_dict["trans"].shape[0]
            else:
                grasp_config_dict = np.load(filepath, allow_pickle=True).item()
                num_grasps += grasp_config_dict["trans"].shape[0]
                for key in combined_grasp_config_dict.keys():
                    combined_grasp_config_dict[key] = np.concatenate(
                        [
                            combined_grasp_config_dict[key],
                            grasp_config_dict[key],
                        ],
                        axis=0,
                    )

        # Shape checks.
        assert combined_grasp_config_dict is not None
        for key in combined_grasp_config_dict.keys():
            assert combined_grasp_config_dict[key].shape[0] == num_grasps

        new_filepath = args.output_config_dicts_path / filename
        print(f"Saving {num_grasps} grasps to {new_filepath}")
        print()
        np.save(new_filepath, combined_grasp_config_dict, allow_pickle=True)


if __name__ == "__main__":
    args = MergeConfigDictsArgumentsParser().parse_args()
    main(args)
