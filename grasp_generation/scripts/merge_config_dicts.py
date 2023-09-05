import numpy as np
import pathlib
from tap import Tap

class MergeConfigDictsArgumentsParser(Tap):
    input_config_dicts_path: pathlib.Path = pathlib.Path("../data/unmerged_config_dicts")
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

        num_grasps = 0
        grasp_config_dicts_list = []
        for filepath in filepaths_with_filename:
            grasp_config_dicts = np.load(filepath, allow_pickle=True)
            num_grasps += len(grasp_config_dicts)
            grasp_config_dicts_list.append(grasp_config_dicts)

        new_grasp_config_dicts = np.concatenate(grasp_config_dicts_list, axis=0)
        assert len(new_grasp_config_dicts) == num_grasps

        new_filepath = args.output_config_dicts_path / filename
        print(f"Saving {num_grasps} grasps to {new_filepath}")
        print()
        np.save(new_filepath, new_grasp_config_dicts, allow_pickle=True)

if __name__ == "__main__":
    args = MergeConfigDictsArgumentsParser().parse_args()
    main(args)
