from tap import Tap
import os
import sys
import pathlib
from typing import List
from datetime import datetime

sys.path.append(os.path.realpath("."))
DATETIME_STR = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


class ArgParser(Tap):
    input_meshdata_path: pathlib.Path = pathlib.Path("../data/meshdata")
    base_data_path: pathlib.Path = pathlib.Path("../data")
    experiment_name: str = DATETIME_STR
    use_multiprocess: bool = True


def process_data(args: ArgParser):
    # Generate hand configs.
    hand_gen_command = (
        f"python scripts/generate_hand_config_dicts.py --meshdata_root_path {args.input_meshdata_path}"
        + f" --output_hand_config_dicts_path {args.base_data_path / args.experiment_name / 'hand_config_dicts'}"
        + " --rand_object_scale"
        + " --use_penetration_energy"
    )

    if args.use_multiprocess:
        hand_gen_command += " --use_multiprocess"

    print(f"Running: {hand_gen_command}")
    os.system(hand_gen_command)

    # Get resulting mid-opt steps
    mid_opt_path = (
        args.base_data_path
        / args.experiment_name
        / "hand_config_dicts"
        / "mid_optimization"
    )
    mid_opt_steps = [int(str(x.stem)) for x in mid_opt_path.iterdir()]

    # Generate grasp configs.
    grasp_gen_command = (
        f"python scripts/generate_grasp_config_dicts.py --meshdata_root_path {args.input_meshdata_path}"
        + f" --input_hand_config_dicts_path {args.base_data_path / args.experiment_name / 'hand_config_dicts'}"
        + f" --output_grasp_config_dicts_path {args.base_data_path / args.experiment_name / 'grasp_config_dicts'}"
        + f" --mid_optimization_steps {' '.join([str(x) for x in mid_opt_steps])}"
    )

    print(f"Running: {grasp_gen_command}")
    os.system(grasp_gen_command)


if __name__ == "__main__":
    args = ArgParser().parse_args()
    process_data(args)
